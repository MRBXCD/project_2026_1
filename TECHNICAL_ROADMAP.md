# SemEval 2026 Task A: Humor Generation — 技术路线与实现方案

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 技术架构总览](#2-技术架构总览)
- [3. 环境与依赖](#3-环境与依赖)
- [4. 数据流水线](#4-数据流水线)
  - [4.1 SFT 数据准备](#41-sft-数据准备)
  - [4.2 GRPO 训练数据与 Reward 设计](#42-grpo-训练数据与-reward-设计)
- [5. Stage 1: Supervised Fine-Tuning (SFT)](#5-stage-1-supervised-fine-tuning-sft)
  - [5.1 SFT 的目标](#51-sft-的目标)
  - [5.2 LoRA 配置](#52-lora-配置)
  - [5.3 训练配置与代码](#53-训练配置与代码)
- [6. Stage 2: Group Relative Policy Optimization (GRPO)](#6-stage-2-group-relative-policy-optimization-grpo)
  - [6.1 GRPO 原理速览](#61-grpo-原理速览)
  - [6.2 Reward Function 设计（核心）](#62-reward-function-设计核心)
  - [6.3 训练配置与代码](#63-训练配置与代码)
- [7. 推理与约束满足](#7-推理与约束满足)
  - [7.1 Rejection Sampling](#71-rejection-sampling)
  - [7.2 推理流水线](#72-推理流水线)
- [8. 评估方案](#8-评估方案)
- [9. 项目目录结构（建议）](#9-项目目录结构建议)
- [10. 时间线与里程碑](#10-时间线与里程碑)
- [附录 A: GRPO 数学原理](#附录-a-grpo-数学原理)
- [附录 B: 常见问题与调参建议](#附录-b-常见问题与调参建议)

---

## 1. 项目概览

| 项目 | 内容 |
|------|------|
| **任务** | SemEval 2026 Task A — 给定新闻标题（+ 可选关键词约束），生成幽默短文本 |
| **语言** | 英文、中文、西班牙语 |
| **基座模型** | Qwen3-8B（Apache 2.0 开源，支持 119 种语言） |
| **训练框架** | HuggingFace TRL + PEFT (LoRA) + Accelerate |
| **训练流程** | SFT → GRPO |
| **硬件** | 单张 80GB GPU (A100/H100) |
| **训练精度** | bf16 混合精度 |

---

## 2. 技术架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                     整体 Pipeline 架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐      ┌──────────────┐      ┌─────────────────┐   │
│  │  原始数据  │─────▶│  数据预处理    │─────▶│  SFT 数据集      │   │
│  │(rJokes,   │      │(清洗/格式化/  │      │({prompt,        │   │
│  │ CFun,     │      │ 合成任务格式)  │      │  completion})   │   │
│  │ HAHA, ...) │      └──────────────┘      └────────┬────────┘   │
│  └──────────┘                                       │            │
│                                                     ▼            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Stage 1: SFT (LoRA Fine-Tuning)             │   │
│  │  Qwen3-8B + LoRA → 学习幽默风格 + 任务格式映射              │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Stage 2: GRPO (在线 RL)                      │   │
│  │                                                          │   │
│  │  ┌─────────┐    ┌──────────────┐    ┌────────────────┐  │   │
│  │  │ Policy  │───▶│  Group       │───▶│  Reward        │  │   │
│  │  │ Model   │    │  Sampling    │    │  Function      │  │   │
│  │  │(SFT后)  │    │  (G=8条/组)  │    │  (规则+LLM)    │  │   │
│  │  └────┬────┘    └──────────────┘    └────────┬───────┘  │   │
│  │       │                                       │          │   │
│  │       │         ┌──────────────┐              │          │   │
│  │       └─────────│  GRPO Update │◀─────────────┘          │   │
│  │                 │  (组内相对    │                          │   │
│  │                 │   优势估计)   │                          │   │
│  │                 └──────────────┘                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Inference: Rejection Sampling                   │   │
│  │  生成 N 个候选 → 硬约束过滤 → 幽默评分 → 选最佳              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 环境与依赖

### 3.1 核心依赖

```txt
# requirements.txt
torch>=2.4.0
transformers>=4.51.0
accelerate>=1.2.0
peft>=0.14.0
trl>=0.18.0
bitsandbytes>=0.45.0
datasets>=3.0.0
vllm>=0.7.0
flash-attn>=2.7.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0

# 评估
rouge-score
nltk
scikit-learn

# 实验管理
wandb
tensorboard
tqdm

# 推理加速 (可选)
vllm>=0.7.0
```

> **注意**: `trl>=0.18.0` 是关键，因为 GRPOTrainer 在较新版本中才成熟。建议安装时不指定上界版本，直接用最新版。

### 3.2 Dockerfile 更新建议

现有 Dockerfile 基本可用，但建议更新 `trl` 和 `transformers` 版本以确保 GRPO 支持。具体版本在安装时用 `pip install --upgrade` 获取最新即可。

### 3.3 模型下载

```bash
# 使用 huggingface-cli 下载模型（在容器内执行）
huggingface-cli download Qwen/Qwen3-8B --local-dir /workspace/models/Qwen3-8B
```

---

## 4. 数据流水线

### 4.1 SFT 数据准备

SFT 数据的核心目标是让模型学会 **两件事**：
1. **幽默语言风格** — 来自真实幽默语料
2. **任务输入-输出映射** — 来自合成的任务格式化数据

#### 4.1.1 数据来源与用途

| 数据集 | 语言 | 用途 | 格式 |
|--------|------|------|------|
| rJokes (Reddit) | EN | 通用幽默 + 偏好排序（有评分） | TSV with scores |
| News Headlines Sarcasm | EN | **弃用**（讽刺≠幽默，格式不匹配）| - |
| CFun | ZH | 通用幽默 | Arrow (HuggingFace) |
| HAHA 2019 | ES | 通用幽默 | CSV with scores |
| **合成数据** | EN/ZH/ES | **任务格式化训练** | 用强模型生成 |

#### 4.1.2 SFT 数据格式

所有数据统一为 **对话格式（chat template）** 以匹配 Qwen3 的 chat format：

**类型 A：通用幽默数据（教风格）**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Tell me a short joke."
    },
    {
      "role": "assistant",
      "content": "I told my wife she was drawing her eyebrows too high. She looked surprised."
    }
  ]
}
```

**类型 B：任务格式化数据（教映射）— 这部分非常重要**

```json
{
  "messages": [
    {
      "role": "user",
      "content": "You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.\n\nHeadline: \"Tech Giants Face New Regulations on AI Safety\"\nRequired words: penguin, bankruptcy\n\nWrite a humorous one-liner that includes both required words."
    },
    {
      "role": "assistant",
      "content": "After the new AI regulations, even the penguin recognition startup filed for bankruptcy — turns out investors don't find Antarctic tech very liquid."
    }
  ]
}
```

> **关键**: 类型 B 的数据需要通过合成生成（见下方脚本）。建议 SFT 数据中类型 A 和类型 B 的比例大约为 **6:4** 或 **7:3**。

#### 4.1.3 数据合成脚本思路

任务格式化数据（类型 B）的合成流程：

```python
"""
合成任务格式化 SFT 数据的思路示例（伪代码）

核心思想：
1. 从新闻数据集（如 SemEval 提供的 headlines 或公开新闻数据集）中抽取标题
2. 从词表中随机配对两个低频词作为关键词约束
3. 调用强模型（Gemini / GPT-4）生成符合约束的幽默回复
4. 进行质量过滤后存储
"""
import random
import json


def build_prompt_for_synthesis(headline: str, word1: str, word2: str, lang: str) -> str:
    """构造让强模型生成幽默回复的 prompt"""
    if lang == "en":
        return (
            f"You are a professional comedy writer. "
            f"Write a short, clever one-liner joke inspired by the following news headline. "
            f"The joke MUST naturally include both words: '{word1}' and '{word2}'.\n\n"
            f"Headline: \"{headline}\"\n\n"
            f"Requirements:\n"
            f"- One sentence only\n"
            f"- Must contain '{word1}' and '{word2}'\n"
            f"- Should be genuinely funny, not forced\n\n"
            f"Joke:"
        )
    elif lang == "zh":
        return (
            f"你是一位专业的喜剧作家。"
            f"根据以下新闻标题，写一个简短而巧妙的笑话。"
            f"笑话中必须自然地包含以下两个词：'{word1}' 和 '{word2}'。\n\n"
            f"新闻标题：「{headline}」\n\n"
            f"要求：\n"
            f"- 仅一句话\n"
            f"- 必须包含'{word1}'和'{word2}'\n"
            f"- 要真正有趣\n\n"
            f"笑话："
        )
    elif lang == "es":
        return (
            f"Eres un escritor de comedia profesional. "
            f"Escribe un chiste corto e ingenioso inspirado en el siguiente titular de noticias. "
            f"El chiste DEBE incluir naturalmente ambas palabras: '{word1}' y '{word2}'.\n\n"
            f"Titular: \"{headline}\"\n\n"
            f"Requisitos:\n"
            f"- Solo una oración\n"
            f"- Debe contener '{word1}' y '{word2}'\n"
            f"- Debe ser genuinamente gracioso\n\n"
            f"Chiste:"
        )


def build_sft_example(headline: str, word1: str, word2: str, 
                       response: str, lang: str) -> dict:
    """将合成结果包装成 SFT 训练格式"""
    if lang == "en":
        user_msg = (
            f"You are a witty comedian. Given the following news headline, "
            f"write a short, funny one-liner joke inspired by it.\n\n"
            f"Headline: \"{headline}\"\n"
            f"Required words: {word1}, {word2}\n\n"
            f"Write a humorous one-liner that includes both required words."
        )
    elif lang == "zh":
        user_msg = (
            f"你是一位机智的喜剧演员。根据以下新闻标题，"
            f"写一个简短有趣的笑话。\n\n"
            f"新闻标题：「{headline}」\n"
            f"必须包含的词语：{word1}、{word2}\n\n"
            f"写一句包含以上两个词语的幽默段子。"
        )
    elif lang == "es":
        user_msg = (
            f"Eres un comediante ingenioso. Dado el siguiente titular de noticias, "
            f"escribe un chiste corto y divertido inspirado en él.\n\n"
            f"Titular: \"{headline}\"\n"
            f"Palabras requeridas: {word1}, {word2}\n\n"
            f"Escribe un chiste de una línea que incluya ambas palabras."
        )

    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ]
    }


# === 合成流程（概念代码，需要补充 API 调用逻辑）===
def synthesize_task_data(headlines: list, word_pairs: list, lang: str, 
                          api_call_fn, n_samples: int = 500) -> list:
    """
    headlines: 新闻标题列表
    word_pairs: [(word1, word2), ...] 关键词对列表
    lang: 语言代码 "en" / "zh" / "es"
    api_call_fn: 调用强模型的函数 (prompt) -> response
    """
    results = []
    for i in range(n_samples):
        headline = random.choice(headlines)
        w1, w2 = random.choice(word_pairs)
        
        synthesis_prompt = build_prompt_for_synthesis(headline, w1, w2, lang)
        response = api_call_fn(synthesis_prompt)
        
        # 质量过滤：检查关键词是否真的出现在回复中
        if w1.lower() in response.lower() and w2.lower() in response.lower():
            example = build_sft_example(headline, w1, w2, response, lang)
            results.append(example)
    
    return results
```

#### 4.1.4 rJokes 数据处理要点

rJokes 数据集自带评分，可以做两件事：
1. **SFT**: 筛选高评分（如 score > 10）的笑话作为回复语料
2. **后续 Reward Model 训练**（如果需要）: 用高评分 vs 低评分构造偏好对

```python
"""rJokes 数据预处理思路"""
import pandas as pd

# rJokes TSV 格式通常为: id, body (setup), score, ...
# 根据实际列名调整
df = pd.read_csv("data/sft/raw/rjoke/train.tsv.gz", sep="\t", compression="gzip")

# 筛选高质量笑话用于 SFT
high_quality = df[df["score"] > 10].copy()

# 转为 SFT 格式
sft_data = []
generic_prompts_en = [
    "Tell me a joke.",
    "Make me laugh with a short joke.",
    "Can you tell me something funny?",
    "I need a good laugh. Give me a joke.",
    "Share a humorous one-liner.",
]

for _, row in high_quality.iterrows():
    joke_text = row["body"]  # 根据实际列名调整
    prompt = random.choice(generic_prompts_en)
    sft_data.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": joke_text}
        ]
    })
```

### 4.2 GRPO 训练数据与 Reward 设计

GRPO 阶段的"训练数据"不是传统意义上的标注数据，而是 **prompt 集合**。模型对每个 prompt 自行生成多条回复（rollout），然后由 reward function 打分。

#### 4.2.1 GRPO Prompt 来源

| 来源 | 说明 |
|------|------|
| SemEval 训练 prompt | `data/semeval_task/task-a-{en,es,zh}.tsv` 中提供的标题和关键词 |
| 合成 prompt | 从新闻数据集中抽取额外标题 + 随机关键词配对 |

> **注意**: 从你的数据来看，SemEval TSV 文件中 word1/word2 列目前是 `-`（即尚未提供关键词约束）。这意味着当前阶段你可以先专注于**新闻标题子任务**，关键词约束等 SemEval 公布完整数据后再补充。

#### 4.2.2 Reward Function 设计（核心中的核心）

GRPO 的训练效果**高度依赖 reward function 的质量**。我们设计一个**复合 reward function**，包含规则项和模型打分项：

```python
"""
GRPO Reward Function 设计

Reward = R_format + R_keyword + R_relevance + R_humor

其中:
- R_format:  格式合规性（硬约束，规则检查）
- R_keyword: 关键词包含（硬约束，规则检查）
- R_relevance: 与新闻标题的相关性（软约束，可选）
- R_humor:   幽默程度（软约束，LLM-as-Judge 或 Reward Model）
"""
import re


def reward_format(response: str) -> float:
    """
    格式合规性检查
    
    规则:
    - 必须是非空文本
    - 长度在合理范围内（如 10-280 字符）
    - 不包含大量重复（degeneracy 检测）
    """
    if not response or not response.strip():
        return -2.0
    
    text = response.strip()
    
    # 长度检查
    if len(text) < 10:
        return -1.0
    if len(text) > 280:
        return -0.5
    
    # 重复检测 (简单的 n-gram degeneracy check)
    words = text.split()
    if len(words) >= 4:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.5:  # 超过一半的 trigram 是重复的
            return -1.5
    
    return 0.5  # 格式合规的基础奖励


def reward_keyword(response: str, keywords: list[str]) -> float:
    """
    关键词包含检查
    
    每包含一个关键词得 +1.0，全部包含额外 bonus +0.5
    未包含任何关键词扣 -1.0
    """
    if not keywords:  # 无关键词约束的 prompt
        return 0.0
    
    text = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    
    if hits == 0:
        return -1.0
    elif hits == len(keywords):
        return hits * 1.0 + 0.5  # 全部命中 bonus
    else:
        return hits * 1.0 - 0.5  # 部分命中


def reward_humor_llm_judge(prompt: str, response: str, 
                            judge_fn) -> float:
    """
    使用外部 LLM 评估幽默程度
    
    judge_fn: 调用外部 LLM 的函数，返回 1-5 的评分
    
    注意: 这个调用较慢且有成本，建议:
    - 训练初期可以降低调用频率（如每 N 步才用 LLM judge）
    - 或者用训练好的小 reward model 替代
    """
    judge_prompt = f"""Rate the following joke on a scale of 1-5 for humor.

Context/Prompt: {prompt}
Joke: {response}

Scoring criteria:
1 = Not funny at all, makes no sense
2 = Slightly amusing but weak
3 = Moderately funny
4 = Genuinely funny, clever wordplay or unexpected twist
5 = Hilarious, extremely witty

Reply with ONLY a single number (1-5)."""
    
    try:
        score_str = judge_fn(judge_prompt).strip()
        score = int(score_str)
        score = max(1, min(5, score))
        # 将 1-5 映射到 [-1, 1] 范围
        return (score - 3) / 2.0
    except:
        return 0.0  # 解析失败给中性分


def compute_reward(prompt: str, response: str, 
                   keywords: list[str] = None,
                   judge_fn=None) -> float:
    """
    复合 Reward Function
    
    各项权重可根据实验调整
    """
    r_format = reward_format(response)
    r_keyword = reward_keyword(response, keywords or [])
    
    # 如果格式严重不合规，直接返回低分（短路）
    if r_format <= -1.0:
        return r_format
    
    r_humor = 0.0
    if judge_fn is not None:
        r_humor = reward_humor_llm_judge(prompt, response, judge_fn)
    
    # 加权求和（权重是超参数，需要实验调整）
    total = (
        1.0 * r_format +    # 格式合规
        2.0 * r_keyword +    # 关键词包含（权重较高，因为是硬约束）
        1.5 * r_humor        # 幽默程度
    )
    
    return total
```

> **关于 LLM-as-Judge 的成本问题**: 在 GRPO 训练中，如果每个 rollout 都调用外部 API 打分，成本和延迟都很高。实际操作中有几种策略：
> 
> 1. **先训一个小的 Reward Model**（推荐）：用 rJokes 的评分数据训一个轻量 reward model（可以是 Qwen3-1.7B + classification head），然后在 GRPO 中用它打分。
> 2. **分阶段训练**: 初期只用规则 reward（format + keyword），训练稳定后再加入 humor reward。
> 3. **批量异步调用**: 累积一批 rollout 后批量调用 LLM judge，减少 API 开销。
> 
> **对于课程项目，推荐方案 2**——先用纯规则 reward 跑通 GRPO 流程，确认训练稳定后再逐步引入幽默评分。

---

## 5. Stage 1: Supervised Fine-Tuning (SFT)

### 5.1 SFT 的目标

| 目标 | 说明 |
|------|------|
| 学习幽默语言风格 | 从真实笑话语料中获取幽默表达模式 |
| 学习任务输入-输出映射 | 理解 "标题+关键词 → 笑话" 的生成格式 |
| 建立 GRPO 的初始策略 | 确保 GRPO 启动时有合理的起点，避免从随机策略开始 |

### 5.2 LoRA 配置

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                          # LoRA 秩，8B 模型建议 32-64
    lora_alpha=128,                # 通常设为 2*r
    lora_dropout=0.05,
    target_modules=[               # Qwen3 的注意力层
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    # 注意: 不要加 "lm_head"，在 SFT 阶段微调 attention + FFN 即可
)
```

> **LoRA rank 选择依据**: 对于 8B 模型，r=64 约产生 ~160M 可训练参数（占总参数 ~2%），在 80GB GPU 上完全没有显存压力。如果发现过拟合，可以降到 r=32。

### 5.3 训练配置与代码

```python
"""
SFT 训练脚本骨架

文件: scripts/train_sft.py
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer


def main():
    # ============================================================
    # 1. 加载模型和 Tokenizer
    # ============================================================
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",        # SFT 训练时 padding 在右侧
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",           # 单卡直接 auto
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # 使用 FlashAttention 2
    )

    # ============================================================
    # 2. LoRA 配置
    # ============================================================
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # ============================================================
    # 3. 加载数据集
    # ============================================================
    # 数据集应为 JSON/JSONL 格式，每条包含 "messages" 字段
    # 示例: {"messages": [{"role": "user", "content": "..."}, 
    #                      {"role": "assistant", "content": "..."}]}
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/sft/preprocessed/sft_train.jsonl",
            "validation": "data/sft/preprocessed/sft_val.jsonl",
        }
    )

    # ============================================================
    # 4. 训练配置
    # ============================================================
    training_args = SFTConfig(
        output_dir="checkpoints/sft",
        
        # --- 训练超参 ---
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,    # 有效 batch_size = 4 * 4 = 16
        learning_rate=2e-4,               # LoRA 学习率通常比全量微调高
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        # --- 精度 ---
        bf16=True,
        
        # --- 序列长度 ---
        max_seq_length=512,               # 笑话通常较短，512 足够
        
        # --- 日志与保存 ---
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        
        # --- 其他 ---
        report_to="wandb",               # 或 "tensorboard"
        seed=42,
        
        # --- PEFT ---
        peft_config=lora_config,          # TRL >= 0.18 直接传 LoRA config
    )

    # ============================================================
    # 5. 创建 Trainer 并训练
    # ============================================================
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    
    # 保存最终模型（只保存 LoRA adapter 权重）
    trainer.save_model("checkpoints/sft/final")
    tokenizer.save_pretrained("checkpoints/sft/final")


if __name__ == "__main__":
    main()
```

### 5.4 SFT 阶段关键注意事项

1. **Qwen3 的 thinking mode**: Qwen3 默认有 "thinking" 模式（会生成 `<think>...</think>` 标签的思考过程）。对于幽默生成任务，建议在推理时**关闭 thinking mode**（通过在 system prompt 中指定 `/no_think`，或者在生成参数中配置）。SFT 数据中的 assistant 回复不应包含 thinking 标签。

2. **多语言混合训练**: 将英/中/西三种语言的 SFT 数据混合在一起训练（不分开）。Qwen3 本身有强大的多语言能力，混合训练可以互相增益。

3. **数据量建议**: 总共约 3000-8000 条 SFT 数据即可（太多反而可能过拟合到特定笑话模式）。

---

## 6. Stage 2: Group Relative Policy Optimization (GRPO)

### 6.1 GRPO 原理速览

GRPO 的核心思想（来自 DeepSeek-R1 论文）：

1. **组采样 (Group Sampling)**: 对每个 prompt $x$，用当前策略 $\pi_\theta$ 生成一组 $G$ 个回复 $\{y_1, y_2, \ldots, y_G\}$
2. **奖励计算**: 对每个回复用 reward function 计算得分 $r_i = R(x, y_i)$
3. **组内归一化 (Group-Relative Advantage)**: 用组内均值和标准差归一化，得到优势估计：
   $$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\}) + \epsilon}$$
4. **策略更新**: 使用 PPO-clip 风格的目标函数更新策略，但不需要 value model：
   $$\mathcal{L} = -\mathbb{E}\left[\min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1\pm\epsilon)\hat{A}_i\right)\right] + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$
   其中 $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}$ 是重要性比率。

**相比 PPO 的优势**:
- **无需 value model**: 省去了训练 critic 的成本和不稳定性
- **无需单独的 reward model 前向传播**: reward 可以直接由规则函数计算
- **更稳定**: 组内归一化天然提供 baseline，减少方差

### 6.2 Reward Function 设计

参见 [4.2.2 节](#422-reward-function-设计核心中的核心) 的详细设计。

**分阶段训练策略（推荐）**:

| 阶段 | Reward 组成 | 训练步数 | 目标 |
|------|------------|---------|------|
| Phase 1 | `R_format + R_keyword` (纯规则) | ~200-500 steps | 学会满足硬约束 |
| Phase 2 | `R_format + R_keyword + R_humor` (规则+LLM) | ~300-800 steps | 在满足约束的基础上提升幽默质量 |

> 这种分阶段策略的好处是：先用便宜的规则 reward 跑通流程、调好超参，然后再引入昂贵的 LLM judge。避免一开始就烧钱调参。

### 6.3 训练配置与代码

```python
"""
GRPO 训练脚本骨架

文件: scripts/train_grpo.py

TRL 的 GRPOTrainer 封装了 GRPO 的核心逻辑:
- 自动处理 group sampling（每个 prompt 生成 G 个回复）
- 自动计算 group-relative advantage
- 自动处理 KL 散度约束
- 支持自定义 reward function
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# 1. Reward Function（传给 GRPOTrainer 的核心组件）
# ============================================================
def reward_fn(completions: list[str], prompts: list[str] = None, 
              **kwargs) -> list[float]:
    """
    GRPOTrainer 要求的 reward function 签名:
    - completions: 模型生成的回复列表
    - prompts: 对应的 prompt 列表 (TRL >= 0.18 支持)
    
    返回: 与 completions 等长的 float 列表
    
    注意: GRPOTrainer 会在内部将 prompt 和 completion 组合后传入，
    具体签名需要查阅你安装的 TRL 版本的文档。
    """
    rewards = []
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        
        # 从 prompt 中解析关键词（如果有）
        keywords = extract_keywords_from_prompt(prompt)
        
        # 计算复合 reward
        r = compute_reward(
            prompt=prompt,
            response=completion,
            keywords=keywords,
            judge_fn=None,  # Phase 1 不用 LLM judge
        )
        rewards.append(r)
    
    return rewards


def extract_keywords_from_prompt(prompt: str) -> list[str]:
    """从 prompt 文本中提取关键词约束"""
    import re
    # 匹配 "Required words: word1, word2" 这种格式
    match = re.search(r"Required words?:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
    if not match:
        # 尝试匹配中文格式
        match = re.search(r"必须包含的词语[：:]\s*(.+?)(?:\n|$)", prompt)
    if not match:
        return []
    
    words_str = match.group(1).strip()
    # 按逗号或顿号分割
    keywords = re.split(r"[,，、]", words_str)
    return [kw.strip() for kw in keywords if kw.strip()]


# ============================================================
# 2. 主训练流程
# ============================================================
def main():
    # --- 加载 SFT 阶段训练好的模型 ---
    # 方法: 加载基座模型 + 合并 SFT LoRA adapter
    base_model_name = "Qwen/Qwen3-8B"
    sft_adapter_path = "checkpoints/sft/final"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",          # GRPO 生成阶段需要 left padding
    )
    # 确保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    # 加载并合并 SFT adapter -> 作为 GRPO 的初始策略和参考模型
    # 注意: TRL GRPOTrainer 会自动处理 reference model
    # 你可以选择:
    #   A) 合并 SFT adapter 到基座，然后 GRPO 再套一层新的 LoRA
    #   B) 直接传入 SFT adapter 路径，让 GRPOTrainer 处理
    # 这里演示方案 A（更清晰）:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model = model.merge_and_unload()  # 合并 LoRA 到基座权重

    # GRPO 阶段的新 LoRA 配置（可以用更小的 rank）
    grpo_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # --- 加载 GRPO 训练 prompt ---
    # 数据集只需要 "prompt" 字段
    # 格式: {"prompt": [{"role": "user", "content": "..."}]}
    dataset = load_dataset(
        "json",
        data_files="data/grpo/grpo_prompts.jsonl",
        split="train"
    )

    # --- GRPO 训练配置 ---
    grpo_config = GRPOConfig(
        output_dir="checkpoints/grpo",
        
        # --- GRPO 核心超参 ---
        num_generations=8,            # G: 每个 prompt 生成的回复数量
        max_completion_length=256,    # 生成的最大长度
        
        # --- 训练超参 ---
        num_train_epochs=1,           # GRPO 通常跑 1-2 个 epoch
        per_device_train_batch_size=1,# GRPO 的 batch_size 通常设小
                                       # (因为每个 sample 会生成 G 条)
        gradient_accumulation_steps=8,# 有效 batch = 1 * 8 = 8 个 prompt
        learning_rate=5e-6,           # RL 阶段学习率要比 SFT 小很多
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        # --- KL 散度约束 ---
        beta=0.04,                    # KL penalty 系数，防止偏离 SFT 策略太远
                                       # 太大 → 学不到新东西; 太小 → reward hacking
        
        # --- 精度 ---
        bf16=True,
        
        # --- 日志与保存 ---
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        report_to="wandb",
        seed=42,
        
        # --- PEFT ---
        peft_config=grpo_lora_config,
    )

    # --- 创建 Trainer 并训练 ---
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,       # 自定义 reward function
    )

    trainer.train()
    
    # 保存
    trainer.save_model("checkpoints/grpo/final")
    tokenizer.save_pretrained("checkpoints/grpo/final")


if __name__ == "__main__":
    main()
```

### 6.4 GRPO 训练的关键超参数与调参指南

| 超参数 | 建议范围 | 说明 |
|--------|---------|------|
| `num_generations` (G) | 4 - 16 | 组大小。越大方差越低，但显存和时间消耗线性增长。**建议从 8 开始** |
| `beta` (KL 系数) | 0.01 - 0.1 | 核心超参。过大导致策略更新不动（模型不变），过小导致 reward hacking（模型发现 reward function 的漏洞）。**建议从 0.04 开始** |
| `learning_rate` | 1e-6 - 1e-5 | RL 阶段要比 SFT 小 1-2 个数量级。**建议 5e-6** |
| `per_device_train_batch_size` | 1 - 2 | 因为每个 sample 生成 G 条回复，实际处理量 = batch × G。设小以防 OOM |
| `gradient_accumulation_steps` | 4 - 16 | 通过累积增大有效 batch，稳定训练 |
| `max_completion_length` | 128 - 512 | 笑话较短，256 通常够用 |
| `temperature` (生成时) | 0.7 - 1.0 | GRPO 需要多样性，不要太低。**建议 0.9** |

### 6.5 GRPO 训练监控指标

训练过程中需要关注以下指标（通过 wandb/tensorboard 观察）：

| 指标 | 正常趋势 | 异常信号 |
|------|---------|---------|
| `reward/mean` | 缓慢上升 | 快速飙升 → reward hacking |
| `reward/std` | 逐渐下降 | 始终很高 → 训练不稳定 |
| `kl_divergence` | 缓慢增长，保持适度 | 爆炸 → 降低 lr 或增大 beta |
| `policy_loss` | 波动但总体下降 | 持续不动 → lr 太小或 reward 信号太弱 |
| `completion_length` | 相对稳定 | 趋向 max_length → 模型可能在 padding 或 rambling |

---

## 7. 推理与约束满足

### 7.1 Rejection Sampling

训练完成后，推理阶段使用 rejection sampling 来保证约束满足率：

```python
"""
推理阶段的 Rejection Sampling

文件: scripts/inference.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_path: str, adapter_path: str):
    """加载 GRPO 训练后的模型"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # 加载 GRPO adapter
    # 注意: 如果 GRPO 是在 SFT-merged 模型上训练的，
    # 需要先 merge SFT adapter 再加载 GRPO adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_candidates(model, tokenizer, prompt: str, 
                         n_candidates: int = 16,
                         max_new_tokens: int = 256,
                         temperature: float = 0.9,
                         top_p: float = 0.95) -> list[str]:
    """为单个 prompt 生成 N 个候选回复"""
    messages = [{"role": "user", "content": prompt}]
    
    # 使用 Qwen3 的 chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False,  # 关闭 thinking mode
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=n_candidates,  # 一次生成 N 个
    )
    
    # 解码（去掉 prompt 部分）
    prompt_len = inputs["input_ids"].shape[1]
    candidates = []
    for seq in outputs:
        text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
        candidates.append(text.strip())
    
    return candidates


def rejection_sample(model, tokenizer, prompt: str,
                      keywords: list[str] = None,
                      n_candidates: int = 16,
                      humor_scorer=None) -> dict:
    """
    Rejection Sampling 推理流程
    
    返回: {
        "best_response": str,
        "all_candidates": list[str],
        "constraint_pass_rate": float,
        "scores": list[float]
    }
    """
    # Step 1: 生成候选
    candidates = generate_candidates(model, tokenizer, prompt, n_candidates)
    
    # Step 2: 硬约束过滤
    if keywords:
        valid_candidates = []
        for c in candidates:
            c_lower = c.lower()
            if all(kw.lower() in c_lower for kw in keywords):
                valid_candidates.append(c)
    else:
        valid_candidates = candidates.copy()
    
    constraint_pass_rate = len(valid_candidates) / len(candidates)
    
    # Step 3: 如果没有候选通过硬约束，使用 fallback
    if not valid_candidates:
        # Fallback: 选包含最多关键词的候选
        if keywords:
            best = max(candidates, 
                      key=lambda c: sum(kw.lower() in c.lower() for kw in keywords))
        else:
            best = candidates[0]
        return {
            "best_response": best,
            "all_candidates": candidates,
            "constraint_pass_rate": 0.0,
            "scores": [],
        }
    
    # Step 4: 软约束排序（幽默评分）
    if humor_scorer:
        scored = [(c, humor_scorer(prompt, c)) for c in valid_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        scores = [s for _, s in scored]
    else:
        # 没有 humor scorer 则随机选一个合规候选
        best = valid_candidates[0]
        scores = []
    
    return {
        "best_response": best,
        "all_candidates": candidates,
        "constraint_pass_rate": constraint_pass_rate,
        "scores": scores,
    }


# === 使用示例 ===
if __name__ == "__main__":
    model, tokenizer = load_model(
        base_model_path="Qwen/Qwen3-8B",
        adapter_path="checkpoints/grpo/final"
    )
    
    prompt = (
        "You are a witty comedian. Given the following news headline, "
        "write a short, funny one-liner joke inspired by it.\n\n"
        "Headline: \"Tech Giants Face New Regulations on AI Safety\"\n"
        "Required words: penguin, bankruptcy\n\n"
        "Write a humorous one-liner that includes both required words."
    )
    
    result = rejection_sample(
        model, tokenizer, prompt,
        keywords=["penguin", "bankruptcy"],
        n_candidates=16,
    )
    
    print(f"Best response: {result['best_response']}")
    print(f"Constraint pass rate: {result['constraint_pass_rate']:.2%}")
    print(f"Total candidates: {len(result['all_candidates'])}")
```

### 7.2 使用 vLLM 加速推理（可选）

当需要对大量 prompt 进行 rejection sampling 时，使用 vLLM 可以显著提速：

```python
"""
使用 vLLM 的批量推理（比 HuggingFace generate 快 5-10x）
注意: vLLM 对 LoRA adapter 的支持需要 vLLM >= 0.4.0
"""
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# 加载模型（vLLM 方式）
llm = LLM(
    model="Qwen/Qwen3-8B",
    enable_lora=True,
    max_lora_rank=64,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    max_tokens=256,
    n=16,  # 每个 prompt 生成 16 个候选
)

# 准备 LoRA adapter
lora_request = LoRARequest("grpo_adapter", 1, "checkpoints/grpo/final")

# 批量推理
prompts = [...]  # prompt 列表
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```

---

## 8. 评估方案

### 8.1 自动化指标（Tier 1 — 无需模型）

| 指标 | 适用子任务 | 计算方式 |
|------|-----------|---------|
| Constraint Satisfaction Rate | 关键词子任务 | 两个关键词是否都出现（精确匹配/模糊匹配） |
| Format Compliance | 全部 | 单句/长度/非空检查 |
| Degeneracy Rate | 全部 | 重复 n-gram 占比 |
| Distinct-1 / Distinct-2 | 全部 | 生成文本的 unigram/bigram 多样性 |

### 8.2 LLM-as-Judge（Tier 2）

```python
"""
LLM-as-Judge Pairwise 评估

对每个 prompt，比较 baseline（零样本基座）和 proposed（训练后模型）的输出
"""

JUDGE_PROMPT_TEMPLATE = """You are an expert judge of humor quality.

Given the following news headline and two joke responses (A and B), determine which joke is funnier.

Headline: "{headline}"
{keyword_section}

Response A: "{response_a}"
Response B: "{response_b}"

Consider:
1. Is the joke genuinely funny (not just random or nonsensical)?
2. Does it relate to the headline?
3. Is it a single, well-formed sentence?
{keyword_criteria}

Which response is funnier? Reply with ONLY "A" or "B" or "TIE".
"""
```

### 8.3 人工评估（Tier 3）

- 从测试集中随机抽取 20-40 个 prompt
- 2-3 位评估者独立做 A/B 盲评
- 报告与 LLM judge 的 agreement（Cohen's kappa）

---

## 9. 项目目录结构（建议）

```
proj_2026_1/
├── configs/                        # 训练配置文件
│   ├── sft_config.yaml
│   └── grpo_config.yaml
│
├── data/
│   ├── semeval_task/               # SemEval 原始数据（已有）
│   │   ├── task-a-en.tsv
│   │   ├── task-a-es.tsv
│   │   └── task-a-zh.tsv
│   ├── sft/
│   │   ├── raw/                    # 原始数据集（已有）
│   │   │   ├── rjoke/
│   │   │   ├── cfun/
│   │   │   ├── haha/
│   │   │   └── news_headlines.../
│   │   └── preprocessed/           # 处理后的 SFT 数据
│   │       ├── sft_train.jsonl
│   │       └── sft_val.jsonl
│   └── grpo/
│       └── grpo_prompts.jsonl      # GRPO 训练用的 prompt 集合
│
├── data_preprocessing/
│   ├── visulization.ipynb          # 数据可视化（已有）
│   ├── prepare_sft_data.py         # SFT 数据预处理脚本
│   ├── prepare_grpo_prompts.py     # GRPO prompt 准备脚本
│   └── synthesize_task_data.py     # 任务格式化数据合成脚本
│
├── scripts/
│   ├── train_sft.py                # SFT 训练入口
│   ├── train_grpo.py               # GRPO 训练入口
│   └── inference.py                # 推理 + Rejection Sampling
│
├── src/
│   ├── __init__.py
│   ├── rewards.py                  # Reward function 定义
│   ├── data_utils.py               # 数据加载/处理工具
│   └── eval_utils.py               # 评估工具
│
├── evaluation/
│   ├── run_auto_eval.py            # 自动化指标评估
│   ├── run_llm_judge.py            # LLM-as-Judge 评估
│   └── analyze_results.py          # 结果分析与可视化
│
├── checkpoints/                    # 模型检查点（已有）
│   ├── sft/
│   └── grpo/
│
├── utils/                          # 通用工具（已有）
│
├── Dockerfile
├── docker_build.sh
├── docker_run.sh
├── requirements.txt
├── TECHNICAL_ROADMAP.md            # 本文档
└── .gitignore
```

---

## 10. 时间线与里程碑

### Week 1: 数据准备 + 环境搭建

| 任务 | 预计耗时 | 交付物 |
|------|---------|--------|
| 环境搭建（Docker + 依赖） | 0.5 天 | 可运行的容器环境 |
| 数据探索与可视化 | 1 天 | EDA notebook |
| rJokes / CFun / HAHA 数据清洗 | 1 天 | 清洗后的通用幽默数据 |
| 任务格式化数据合成 | 1.5 天 | 合成 SFT 数据 |
| SFT 数据集构建 + 验证 | 1 天 | `sft_train.jsonl`, `sft_val.jsonl` |

### Week 2: SFT + GRPO 训练

| 任务 | 预计耗时 | 交付物 |
|------|---------|--------|
| SFT 训练脚本编写 | 0.5 天 | `train_sft.py` |
| SFT 训练 + 调参 | 1 天 | SFT checkpoint |
| Reward function 实现 | 1 天 | `rewards.py` |
| GRPO 训练脚本编写 | 0.5 天 | `train_grpo.py` |
| GRPO Phase 1（纯规则 reward）| 1 天 | GRPO Phase 1 checkpoint |
| GRPO Phase 2（加入幽默评分）| 1 天 | GRPO Phase 2 checkpoint |

### Week 3: 评估 + 调优

| 任务 | 预计耗时 | 交付物 |
|------|---------|--------|
| 推理 + Rejection Sampling 实现 | 0.5 天 | `inference.py` |
| 自动化指标评估 | 0.5 天 | 指标报告 |
| LLM-as-Judge 评估 | 1 天 | pairwise 胜率报告 |
| 人工评估 | 1 天 | 人工评估报告 |
| 模型调优迭代 | 2 天 | 最终模型 |

### Week 4: 报告 + 整理

| 任务 | 预计耗时 | 交付物 |
|------|---------|--------|
| 代码整理 + 文档 | 1 天 | 整洁的代码仓库 |
| 最终报告撰写 | 3-4 天 | 课程论文 |

---

## 附录 A: GRPO 数学原理

### A.1 目标函数

GRPO 的策略梯度目标函数：

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} \hat{A}_i,\; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) \right]
$$

其中 $\hat{A}_i$ 是组内相对优势：

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_G)^2}
$$

### A.2 KL 散度正则化

为防止策略偏离参考模型（SFT 模型）过远，加入 KL 惩罚：

$$
\mathcal{L}_{\text{total}} = -\mathcal{J}_{\text{GRPO}}(\theta) + \beta \cdot \mathbb{E}_{x}\left[D_{\text{KL}}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right)\right]
$$

$\beta$ 控制 KL 惩罚的强度：
- $\beta$ 太大 → 策略几乎不更新，等于没训
- $\beta$ 太小 → 策略更新过猛，可能出现 reward hacking

### A.3 与 PPO 的对比

| 特性 | PPO | GRPO |
|------|-----|------|
| Value Model | 需要（额外训练 Critic） | **不需要** |
| Advantage 估计 | GAE（需要 value function） | **组内归一化**（无需 value function） |
| 显存占用 | 高（policy + value + ref 三个模型） | 低（policy + ref 两个模型） |
| 训练稳定性 | 需要仔细平衡 actor/critic | 相对更稳定 |
| 适用场景 | 通用 RL | 特别适合有规则化 reward 的场景 |

---

## 附录 B: 常见问题与调参建议

### B.1 GRPO 训练 reward 不增长

**可能原因与解决方案:**

| 症状 | 原因 | 解决 |
|------|------|------|
| reward 完全不动 | 学习率太小 | 增大 lr (如 5e-6 → 1e-5) |
| reward 不动 | beta 太大 | 减小 beta (如 0.1 → 0.04) |
| reward 不动 | reward function 区分度不够 | 检查生成的回复是否全部得到相似分数 |
| reward 不动 | SFT 模型太差 | 检查 SFT 模型生成质量，可能需要更多/更好的 SFT 数据 |

### B.2 Reward Hacking

**症状**: reward 快速上升但生成质量下降（模型找到了 reward function 的漏洞）

**常见案例与对策:**
- **模型重复关键词**: 如 "penguin penguin penguin bankruptcy" → 在 `reward_format` 中加入更强的重复惩罚
- **模型生成超长文本**: → 在 reward 中加入长度惩罚
- **模型输出不像笑话**: → 增大 beta 限制偏离 SFT 策略，或引入 humor reward

### B.3 显存不足 (OOM)

**GRPO 阶段比 SFT 更容易 OOM**，因为需要同时处理一组回复。调整顺序：

1. 减小 `num_generations` (如 8 → 4)
2. 减小 `per_device_train_batch_size` (如 2 → 1)
3. 增大 `gradient_accumulation_steps` 补偿
4. 减小 `max_completion_length`
5. 启用 gradient checkpointing (`gradient_checkpointing=True`)
6. 使用 QLoRA (4-bit 量化) 替代 LoRA

### B.4 Qwen3 Thinking Mode 处理

Qwen3 的 thinking mode 会在回复中生成 `<think>...</think>` 标签。对于幽默生成，**必须关闭**：

```python
# 方法 1: 在 chat template 中关闭
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,   # 关闭 thinking mode
)

# 方法 2: 在 system prompt 中指定
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "your prompt here"},
]
```

在 SFT 数据和 GRPO prompt 构造中，**统一使用 `/no_think` system prompt** 以保持一致性。
