# 数据处理模块设计文档

## 目录

- [1. 概述](#1-概述)
- [2. 数据源盘点](#2-数据源盘点)
  - [2.1 各数据源格式详情](#21-各数据源格式详情)
  - [2.2 SemEval Task A 子任务结构](#22-semeval-task-a-子任务结构)
- [3. 处理架构](#3-处理架构)
  - [3.1 三层架构总览](#31-三层架构总览)
  - [3.2 Layer 1: Source Parsers](#32-layer-1-source-parsers)
  - [3.3 Layer 2: 统一中间格式](#33-layer-2-统一中间格式)
  - [3.4 Layer 3: Formatters](#34-layer-3-formatters)
- [4. SFT 数据设计](#4-sft-数据设计)
  - [4.1 Type A: 通用幽默数据](#41-type-a-通用幽默数据)
  - [4.2 Type B: 任务格式化数据（合成）](#42-type-b-任务格式化数据合成)
  - [4.3 Type A 与 Type B 的混合策略](#43-type-a-与-type-b-的混合策略)
  - [4.4 Qwen3 Thinking Mode 处理](#44-qwen3-thinking-mode-处理)
- [5. GRPO 数据设计](#5-grpo-数据设计)
  - [5.1 子任务 1: Headline-Only](#51-子任务-1-headline-only)
  - [5.2 子任务 2: Keywords-Only](#52-子任务-2-keywords-only)
- [6. Reward Model 偏好对数据设计](#6-reward-model-偏好对数据设计)
  - [6.1 用途与训练目标](#61-用途与训练目标)
  - [6.2 偏好对构造策略](#62-偏好对构造策略)
  - [6.3 数据格式](#63-数据格式)
  - [6.4 构造流程与采样细节](#64-构造流程与采样细节)
  - [6.5 注意事项](#65-注意事项)
- [7. Prompt 模板设计](#7-prompt-模板设计)
  - [7.1 Type A 通用幽默 Prompt 池](#71-type-a-通用幽默-prompt-池)
  - [7.2 Type B 任务格式化 Prompt 模板](#72-type-b-任务格式化-prompt-模板)
  - [7.3 GRPO Prompt 模板](#73-grpo-prompt-模板)
- [8. 质量筛选策略](#8-质量筛选策略)
- [9. 文件组织与输出](#9-文件组织与输出)
- [10. 处理流水线调用方式](#10-处理流水线调用方式)

---

## 1. 概述

本模块负责将多个原始数据源处理为 SFT 和 GRPO 两个训练阶段所需的标准格式数据。
设计原则：

1. **以 HuggingFace `datasets` 库为核心** — 使用 `load_dataset`、`map`、`filter`、`concatenate_datasets` 等现成 API，尽量不造轮子
2. **管道式处理** — 原始数据 → 统一中间格式 → 训练格式，每一步独立可调
3. **SFT 和 GRPO 使用不同的 Formatter** — 二者的数据结构不同

---

## 2. 数据源盘点

### 2.1 各数据源格式详情

| 数据集 | 语言 | 用途 | 原始格式 | 关键字段 | 数据量 | 路径 |
|--------|------|------|---------|---------|--------|------|
| **rJokes** | EN | SFT (Type A) | TSV.gz | `score` (int), `joke` (str) | ~43K (dev) + train + test | `data/rjoke/` |
| **CFun** | ZH | SFT (Type A) | HF Arrow | `instruction`, `input`, `output` | 164K | `data/cfun/` |
| **HAHA 2019** | ES | SFT (Type A) | CSV | `text`, `is_humor` (0/1), `funniness_average` (float) | ~36K | `data/haha/` |
| **Chinese Humor Multi-Labeled** | ZH | SFT (Type A) + 偏好对 | TSV (tab-separated) | `Content` (str), `HumorLevel` (1-5) | ~3.3K | `data/Chinese_Humor_Multi-Labeled/` |
| **SemEval Task A** | EN/ZH/ES | GRPO prompts | TSV | `headline`, `word1`, `word2` | 各 ~300 (275 headline + 25 keyword) | `data/semeval_task/` |
| **合成 Type B 数据** | EN/ZH/ES | SFT (Type B) | JSONL（合成后存储） | `messages` | 按需 | `data/synthesized/` |

#### rJokes 字段说明

```
score (int)  |  joke (str)
─────────────┼──────────────────────────────────
1            |  "I'll have a cheeseburger..."
0            |  "Who is Michael J. Fox's..."
3            |  "A guy calls in sick to work..."
```

- score 为 Reddit 社区投票得分，数值越高越受欢迎
- 分布集中在 0-3，长尾延伸到更高分数

#### CFun 字段说明

```
instruction (str)  |  input (str)  |  output (str)
───────────────────┼───────────────┼──────────────────────
"请讲一个笑话"     |  ""           |  "有一天小明..."
```

- 已是指令微调格式，但其原生 instruction 与本任务不匹配
- **仅使用 `output` 字段**作为笑话文本，重新配 prompt

#### HAHA 2019 字段说明

```
id  |  text (str)              |  is_humor (0/1)  |  funniness_average (float)
────┼──────────────────────────┼──────────────────┼───────────────────────────
... |  "Niveles de retraso..." |  1               |  1.5
```

- `is_humor=1` 表示被标注为幽默文本
- `funniness_average` 为 1-5 分的平均评分（仅 is_humor=1 时有意义）
- SFT 阶段仅使用 `is_humor=1` 的数据
- 保留 `is_humor=0` 的数据，以备后续构造偏好对

#### Chinese Humor Multi-Labeled 字段说明

```
ID (str)  |  Title (str)  |  Content (str)         |  HumorLevel (1-5)
──────────┼───────────────┼────────────────────────┼──────────────────
L0001     |  要求加薪      |  員工：老闆，你必須...  |  4
L0004     |  職業習慣      |  一天，一位法官的...    |  2
```

- `HumorLevel` 1-5 分评级
- 文本为**繁体中文**
- SFT 阶段使用 `HumorLevel >= 4` 的高质量笑话
- 全量数据（包含低分）保留用于偏好对构造

### 2.2 SemEval Task A 子任务结构

SemEval Task A 包含**两个互斥的子任务**：

| 子任务 | 输入 | 约束 | 数据范围（EN 为例） |
|--------|------|------|-------------------|
| **Headline-based** | 新闻标题 | 生成与标题相关的笑话 | en_2001 ~ en_2275 (275条) |
| **Keyword-based** | 两个关键词 | 笑话中必须包含这两个词 | en_2276 ~ en_2300 (25条) |

数据特征：
- Headline-based 条目：`word1 = "-"`, `word2 = "-"`, `headline` 有值
- Keyword-based 条目：`headline = "-"`, `word1` 和 `word2` 有值
- **两种子任务互斥**，不存在同时有 headline 和 keywords 的条目

---

## 3. 处理架构

### 3.1 三层架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                           数据处理架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Layer 1: Source Parsers                                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │  rJokes   │ │   CFun   │ │   HAHA   │ │ Chinese  │ │ SemEval  │ │
│  │  Parser   │ │  Parser  │ │  Parser  │ │ Humor    │ │  Parser  │ │
│  │          │ │          │ │          │ │  Parser  │ │          │ │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ │
│        │            │            │             │            │       │
│        ▼            ▼            ▼             ▼            │       │
│  Layer 2: 统一中间格式                                       │       │
│  ┌──────────────────────────────────────────────┐          │       │
│  │ { "text", "lang", "score", "source" }        │          │       │
│  └──────────────────┬───────────────────────────┘          │       │
│                     │                                       │       │
│  Layer 3: Formatters│                                       │       │
│        ┌────────────┼────────────┐          ┌───────────────┘       │
│        ▼            ▼            ▼          ▼                       │
│  ┌──────────┐ ┌──────────┐ ┌─────────────────┐                     │
│  │ SFT      │ │ SFT      │ │ GRPO Prompt     │                     │
│  │ Type A   │ │ Type B   │ │ Formatter       │                     │
│  │ Formatter│ │ Formatter│ │ (直接从 SemEval) │                     │
│  └────┬─────┘ └────┬─────┘ └───────┬─────────┘                     │
│       │            │               │                                │
│       ▼            ▼               ▼                                │
│  ┌──────────────────────────────────────────────┐                  │
│  │ 输出: HuggingFace Dataset (JSONL)             │                  │
│  │ • data/sft/sft_train.jsonl                    │                  │
│  │ • data/sft/sft_val.jsonl                      │                  │
│  │ • data/grpo/grpo_prompts.jsonl                │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

> 注意：SemEval 数据**不经过**统一中间格式，而是由 SemEval Parser 直接输出给 GRPO Formatter，因为它的结构和用途与 SFT 数据完全不同。

### 3.2 Layer 1: Source Parsers

每个 parser 是一个独立函数，输入为原始文件路径，输出为 `datasets.Dataset` 对象。

| Parser 函数 | 输入文件 | 处理逻辑 |
|------------|---------|---------|
| `parse_rjokes(path)` | `data/rjoke/*.tsv.gz` | 读取 score + joke；过滤空值；score 归一化 |
| `parse_cfun(cache_dir)` | `data/cfun/` | 从 HF cache 加载；仅取 `output` 字段 |
| `parse_haha(path)` | `data/haha/*.csv` | 读取全部列；过滤 `is_humor=1`；归一化 funniness_average |
| `parse_chinese_humor(path)` | `data/Chinese_Humor_Multi-Labeled/mlabel_corpora/JokeHumorLevel.txt` | 读取 Content + HumorLevel；归一化 |
| `parse_semeval(path)` | `data/semeval_task/task-a-*.tsv` | 区分 headline-only 和 keyword-only 两种子任务 |

### 3.3 Layer 2: 统一中间格式

所有幽默数据（rJokes, CFun, HAHA, Chinese Humor）统一为以下 schema：

```python
{
    "text": str,              # 笑话/幽默文本正文
    "lang": "en"|"zh"|"es",   # 语言标识
    "score": float | None,    # 归一化到 [0, 1] 的质量评分，无评分为 None
    "source": str,            # 数据源标识 ("rjokes" / "cfun" / "haha" / "chinese_humor")
}
```

**Score 归一化方案：**

| 数据源 | 原始评分 | 归一化方法 | 说明 |
|--------|---------|-----------|------|
| rJokes | int (0 ~ 数百) | `min(score, 20) / 20.0` | 封顶到 20，避免极端高分 |
| CFun | 无 | `None` | 无评分信息 |
| HAHA 2019 | float (1.0 ~ 5.0) | `funniness_average / 5.0` | 直接线性映射 |
| Chinese Humor | int (1 ~ 5) | `HumorLevel / 5.0` | 直接线性映射 |

### 3.4 Layer 3: Formatters

#### SFT Type A Formatter

将统一中间格式的笑话数据转为 SFT 训练所需的 chat 格式：

```python
# 输入: 统一中间格式
{"text": "...", "lang": "en", "score": 0.75, "source": "rjokes"}

# 输出: SFT chat 格式
{
    "messages": [
        {"role": "user", "content": "<从对应语言的 prompt 池中随机选取>"},
        {"role": "assistant", "content": "<text 字段>"}
    ]
}
```

#### SFT Type B Formatter

将合成的任务格式化数据（已存储为 JSONL）直接加载，无需额外转换。

#### GRPO Formatter

将 SemEval 数据转为 GRPO 训练所需的 prompt 格式：

```python
# 输入: SemEval 解析结果
{"id": "en_2001", "headline": "...", "word1": "-", "word2": "-"}

# 输出 (headline-only):
{
    "prompt": [
        {"role": "user", "content": "<从 headline prompt 模板生成>"}
    ],
    "headline": "原始标题文本",
    "keywords": []
}

# 输出 (keyword-only):
{
    "prompt": [
        {"role": "user", "content": "<从 keyword prompt 模板生成>"}
    ],
    "headline": "",
    "keywords": ["word1", "word2"]
}
```

---

## 4. SFT 数据设计

### 4.1 Type A: 通用幽默数据

**目标**：教会模型幽默语言风格。

**数据来源与质量筛选**：

| 数据源 | 语言 | 筛选条件 | 预估筛选后数量 |
|--------|------|---------|--------------|
| rJokes | EN | `score >= 5` | ~5K-8K（待确认） |
| CFun | ZH | 无（全部使用，随机采样控制量） | 采样 ~5K |
| HAHA 2019 | ES | `is_humor == 1` | ~10K |
| Chinese Humor | ZH | `HumorLevel >= 4` | ~1K |

> CFun 有 164K 条数据，直接全部使用会导致中文数据量远大于英/西，需要**下采样**以保持语言平衡。

**最终格式**：

```json
{
    "messages": [
        {"role": "user", "content": "Tell me a short joke."},
        {"role": "assistant", "content": "I told my wife she was drawing her eyebrows too high. She looked surprised."}
    ]
}
```

### 4.2 Type B: 任务格式化数据（合成）

**目标**：教会模型理解"标题 → 笑话"和"关键词 → 笑话"的输入-输出映射。

**合成流程**（由独立脚本 `synthesize_task_data.py` 完成）：

1. 从新闻标题数据集（推荐 Babel Briefings）中抽取标题
2. 从词表中随机配对两个低频词作为关键词
3. 调用强模型 API（如 Gemini）生成符合约束的幽默回复
4. 质量过滤（检查关键词包含、长度合理性等）
5. 存储为 JSONL

**存储位置**：`data/synthesized/type_b_en.jsonl`, `type_b_zh.jsonl`, `type_b_es.jsonl`

**最终格式（headline 子任务）**：

```json
{
    "messages": [
        {"role": "user", "content": "You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.\n\nHeadline: \"Tech Giants Face New Regulations on AI Safety\"\n\nWrite a humorous one-liner inspired by the headline."},
        {"role": "assistant", "content": "The new AI safety regulations are so strict, even Siri is hiring a lawyer."}
    ]
}
```

**最终格式（keyword 子任务）**：

```json
{
    "messages": [
        {"role": "user", "content": "You are a witty comedian. Write a short, funny one-liner joke that naturally includes both of the following words: 'hammer' and 'flower'.\n\nWrite a humorous one-liner that contains both required words."},
        {"role": "assistant", "content": "I tried to fix my garden with a hammer, but all I got was a flat flower and a noise complaint."}
    ]
}
```

### 4.3 Type A 与 Type B 的混合策略

| 数据类型 | 占比 | 说明 |
|---------|------|------|
| Type A（通用幽默） | ~60-70% | 建立幽默语言风格基础 |
| Type B（任务格式化） | ~30-40% | 教会模型理解任务输入-输出映射 |

混合后 shuffle，然后按 90/10 比例划分 train/val。

### 4.4 Qwen3 Thinking Mode 处理

Qwen3 系列模型（包括 Qwen3-8B）**默认启用 thinking mode**，会在回复前生成 `<think>...</think>` 标签的内部推理。

对于幽默生成任务，thinking mode 不必要且有害（浪费 token、干扰 reward 计算），需关闭。

**处理方式**：**不在数据中添加任何特殊标记**。thinking mode 在训练脚本中通过 `tokenizer.apply_chat_template(..., enable_thinking=False)` 参数统一关闭。数据层面保持干净。

---

## 5. GRPO 数据设计

GRPO 阶段的"训练数据"是 **prompt 集合**（不含 response），模型自行生成多条回复后由 reward function 打分。

### 5.1 子任务 1: Headline-Only

```json
{
    "prompt": [
        {"role": "user", "content": "You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.\n\nHeadline: \"Panamanian lawmakers' Taiwan trip sparks diplomatic row with China\"\n\nWrite a humorous one-liner inspired by the headline."}
    ],
    "headline": "Panamanian lawmakers' Taiwan trip sparks diplomatic row with China",
    "keywords": []
}
```

- `headline` 和 `keywords` 字段**不传入模型**，仅供 reward function 在计算奖励时使用
- `keywords` 为空列表表示无关键词约束

### 5.2 子任务 2: Keywords-Only

```json
{
    "prompt": [
        {"role": "user", "content": "You are a witty comedian. Write a short, funny one-liner joke that naturally includes both of the following words: 'hammer' and 'flower'.\n\nWrite a humorous one-liner that contains both required words."}
    ],
    "headline": "",
    "keywords": ["hammer", "flower"]
}
```

- `headline` 为空字符串表示无标题约束
- `keywords` 包含两个必须出现的关键词，reward function 将检查生成文本是否包含这些词

---

## 6. Reward Model 偏好对数据设计

### 6.1 用途与训练目标

在 GRPO 训练中，我们需要一个 reward function 来对模型生成的每条回复打分。其中"幽默程度"的评分有两种实现方式：

| 方式 | 优点 | 缺点 |
|------|------|------|
| **外部 LLM-as-Judge**（调用 API） | 无需额外训练 | API 调用成本高、速度慢 |
| **训练一个小 Reward Model** | 推理快、无 API 成本 | 需要偏好对数据 + 额外训练步骤 |

如果选择训练 reward model，我们需要构造偏好对数据。Reward model 的训练目标是：给定一条文本，输出一个标量分数，使得"更幽默"的文本得分高于"不幽默"的文本。

### 6.2 偏好对构造策略

我们利用**已有的评分数据**来构造偏好对。核心思路：对同一个 prompt，从高评分样本中选 chosen，从低评分样本中选 rejected。

**可用数据源：**

| 数据源 | 语言 | 评分字段 | chosen 条件 | rejected 条件 | 丢弃中间段 |
|--------|------|---------|------------|--------------|-----------|
| rJokes | EN | score (int) | 归一化 score 前 30% | 归一化 score 后 30% | 中间 40% |
| HAHA 2019 | ES | funniness_average + is_humor | `is_humor=1` 且 funniness >= 3.5 | `is_humor=0`，或 `is_humor=1` 且 funniness <= 2.0 | 中间段 |
| Chinese Humor | ZH | HumorLevel (1-5) | HumorLevel >= 4 | HumorLevel <= 2 | HumorLevel = 3 |

> **关于西班牙语**：HAHA 2019 中 `is_humor=0` 的样本可以直接作为 rejected 来源（这些是标注者认为"不幽默"的文本），比仅依靠分数筛选更可靠。

### 6.3 数据格式

偏好对数据采用 TRL `RewardTrainer` 兼容的格式：

```json
{
    "prompt": [
        {"role": "user", "content": "Tell me a joke."}
    ],
    "chosen": [
        {"role": "assistant", "content": "高评分笑话文本"}
    ],
    "rejected": [
        {"role": "assistant", "content": "低评分笑话文本"}
    ]
}
```

### 6.4 构造流程与采样细节

```
原始带评分数据
       │
       ▼
┌──────────────────────┐
│ 按评分分为三组:        │
│  • high (前 30%)      │
│  • mid  (中间 40%)    │  ← 丢弃，不参与配对
│  • low  (后 30%)      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 配对策略:                         │
│  对同一语言内的 high 和 low 样本,  │
│  随机配对构成 (chosen, rejected)   │
│                                  │
│  prompt 从对应语言的 Type A       │
│  prompt 池中随机选取              │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ 质量控制:                         │
│  • 每个 chosen 样本最多配对 3 次   │
│    (避免一个好笑话被重复使用太多)   │
│  • chosen 和 rejected 文本不能     │
│    过于相似 (可选: 余弦相似度过滤)  │
└──────────────────────────────────┘
```

**预估偏好对数量（粗略估计）：**

| 语言 | 数据源 | high 估计量 | low 估计量 | 可构造偏好对数 |
|------|--------|-----------|-----------|-------------|
| EN | rJokes (~43K dev) | ~6K-8K | ~6K-8K | ~6K-8K 对 |
| ZH | Chinese Humor (~3.3K) | ~1K | ~800 | ~800 对 |
| ES | HAHA 2019 (~36K, 含 is_humor=0) | ~3K-5K | ~15K+ | ~3K-5K 对 |

> 中文偏好对数量偏少（~800 对）。如果效果不够，可以考虑：
> - 使用 CFun 数据 + LLM-as-Judge 进行 RLAIF 合成偏好对
> - 或从 Chinese Humor 中放宽选择范围（如 chosen >= 3, rejected <= 1）

### 6.5 注意事项

**1. Prompt 一致性问题**

标准 RLHF 偏好对要求 chosen 和 rejected 是**对同一个具体 prompt 的不同回复**。但我们的数据是独立采集的笑话，而非同一 prompt 下的不同回复。

这在实践中可行——reward model 本质上学习的是"什么样的文本更幽默"这个分类/排序任务。但需注意：
- 配对时使用相同的 prompt（从 prompt 池中选取同一个）
- 这意味着 reward model 学到的更多是**文本本身的幽默程度**，而非**对特定 prompt 的回复质量**

**2. 是否必须训练 Reward Model**

对于本课程项目，建议**分阶段实施**：
- **Phase 1**: GRPO 仅使用规则 reward（format + keyword）。跑通 GRPO 流程，确认稳定性。
- **Phase 2**: 引入 reward model 或 LLM-as-Judge 增加幽默评分维度。

偏好对数据的处理可以在 Phase 1 期间并行准备，但不阻塞 GRPO 训练的启动。

**3. Reward Model 架构选择**

推荐方案：在 SFT 后的 Qwen3-8B 基础上加一个 scalar value head（TRL 的 `AutoModelForSequenceClassification` 支持）。也可以使用更小的模型（如 Qwen3-1.7B）降低推理成本。

---

## 7. Prompt 模板设计

### 7.1 Type A 通用幽默 Prompt 池

用于 SFT Type A 数据的 user 侧 prompt。每条训练样本构造时，从对应语言的池中**随机抽取一个**。

**英文 Prompt 池：**

```
1.  Tell me a joke.
2.  Tell me a short joke.
3.  Make me laugh with a quick joke.
4.  Can you share something funny?
5.  I need a good laugh. Hit me with a joke.
6.  Share a humorous one-liner.
7.  Give me your best short joke.
8.  I could use some humor right now. Got a joke?
9.  Tell me something that will make me smile.
10. What's a good joke you know?
11. Surprise me with a funny one-liner.
12. Got any good jokes?
13. Make me laugh.
14. Hit me with your funniest joke.
15. Tell me a witty joke.
```

**中文 Prompt 池：**

```
1.  给我讲个笑话吧。
2.  说个段子听听。
3.  来点幽默的。
4.  讲个好笑的故事。
5.  我想听个笑话。
6.  能给我讲个段子吗？
7.  来个短笑话。
8.  说点有趣的东西。
9.  给我说个让人发笑的笑话。
10. 讲个让我开心一下的笑话。
11. 你知道什么好笑的笑话吗？
12. 逗我笑一个。
13. 有没有什么有趣的段子？
14. 给我讲个冷笑话。
15. 来个幽默的短段子吧。
```

**西班牙语 Prompt 池：**

```
1.  Cuéntame un chiste.
2.  Hazme reír con un chiste corto.
3.  ¿Puedes compartir algo gracioso?
4.  Necesito reírme. Dime un chiste.
5.  Comparte un chiste ingenioso.
6.  Dame tu mejor chiste corto.
7.  Me vendría bien algo de humor. ¿Tienes un chiste?
8.  Dime algo que me haga sonreír.
9.  ¿Cuál es un buen chiste que conozcas?
10. Sorpréndeme con un chiste divertido.
11. ¿Tienes algún chiste bueno?
12. Hazme reír.
13. Cuéntame tu chiste más gracioso.
14. Dime un chiste ingenioso.
15. ¿Sabes algún chiste corto?
```

### 7.2 Type B 任务格式化 Prompt 模板

用于 SFT Type B 数据和合成脚本。按子任务和语言区分。

#### Headline 子任务模板

**英文：**

```
You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.

Headline: "{headline}"

Write a humorous one-liner inspired by the headline.
```

**中文：**

```
你是一位机智的喜剧演员。根据以下新闻标题，写一个简短有趣的笑话。

新闻标题：「{headline}」

写一句幽默的段子。
```

**西班牙语：**

```
Eres un comediante ingenioso. Dado el siguiente titular de noticias, escribe un chiste corto y divertido inspirado en él.

Titular: "{headline}"

Escribe un chiste divertido de una línea inspirado en el titular.
```

#### Keyword 子任务模板

**英文：**

```
You are a witty comedian. Write a short, funny one-liner joke that naturally includes both of the following words: '{word1}' and '{word2}'.

Write a humorous one-liner that contains both required words.
```

**中文：**

```
你是一位机智的喜剧演员。写一个简短有趣的笑话，其中必须自然地包含以下两个词：「{word1}」和「{word2}」。

写一句包含以上两个词语的幽默段子。
```

**西班牙语：**

```
Eres un comediante ingenioso. Escribe un chiste corto y divertido que incluya naturalmente las siguientes dos palabras: '{word1}' y '{word2}'.

Escribe un chiste divertido de una línea que contenga ambas palabras.
```

### 7.3 GRPO Prompt 模板

GRPO 阶段使用与 Type B 相同的模板结构，但数据来源为 SemEval 官方数据。模板复用 6.2 节定义的模板。

---

## 8. 质量筛选策略

### SFT 阶段筛选

| 数据源 | 筛选条件 | 说明 |
|--------|---------|------|
| rJokes | `score >= 5` | 取高评分笑话（约 top 20-30%），后续可根据训练效果调整阈值 |
| CFun | 无筛选 + 下采样 | 164K 全部可用，但需下采样至 ~5K 以平衡语言比例 |
| HAHA 2019 | `is_humor == 1` | 排除非幽默文本 |
| Chinese Humor | `HumorLevel >= 4` | 取较高评分笑话 |

### 通用文本质量过滤

对所有数据源统一应用：

1. **非空检查** — 过滤空文本或仅含空白符的样本
2. **最小长度** — 文本长度 >= 10 字符
3. **最大长度** — 文本长度 <= 2000 字符（过长的文本可能是数据噪声）
4. **去重** — 基于文本精确匹配去重

### 偏好对构造预留

以下数据保留完整（含低分/非幽默样本），不在 SFT 阶段使用，但可用于后续偏好对构造：

- rJokes: 全量数据（含低分）
- HAHA 2019: `is_humor=0` 的数据
- Chinese Humor: `HumorLevel <= 2` 的数据

---

## 9. 文件组织与输出

### 代码文件组织

```
proj_2026_1/
├── data_preprocessing/
│   ├── DATA_PIPELINE_DESIGN.md     # 本设计文档
│   ├── parsers.py                  # Layer 1: 各数据源的 parser 函数
│   ├── prompt_templates.py         # 多语言 prompt 池 + 任务模板
│   ├── formatters.py               # Layer 3: SFT / GRPO / 偏好对 格式转换器
│   ├── pipeline.py                 # 端到端流水线 (解析 → 筛选 → 格式化 → 保存)
│   ├── synthesize_task_data.py     # Type B 数据合成脚本（独立，需 API）
│   └── visulization.ipynb          # 数据可视化（已有）
```

### 数据输出结构

```
proj_2026_1/
├── data/
│   ├── raw/                        # 原始数据 (用户已整理)
│   │   ├── rjoke/
│   │   ├── cfun/
│   │   ├── haha/
│   │   ├── Chinese_Humor_Multi-Labeled/
│   │   └── semeval_task/
│   │
│   ├── preprocessed/               # 统一中间格式 (含全量评分，供多阶段复用)
│   │   ├── unified_en.jsonl
│   │   ├── unified_zh.jsonl
│   │   └── unified_es.jsonl
│   │
│   ├── synthesized/                # 合成的 Type B 数据
│   │   ├── type_b_en.jsonl
│   │   ├── type_b_zh.jsonl
│   │   └── type_b_es.jsonl
│   │
│   ├── sft/                        # 最终 SFT 训练数据
│   │   ├── sft_train.jsonl
│   │   └── sft_val.jsonl
│   │
│   ├── reward/                     # Reward Model 偏好对训练数据
│   │   ├── preference_train.jsonl
│   │   └── preference_val.jsonl
│   │
│   └── grpo/                       # 最终 GRPO 训练数据
│       └── grpo_prompts.jsonl
```

### 输出文件格式说明

**`preprocessed/unified_*.jsonl`** — 统一中间格式，每行一个 JSON：

```json
{"text": "...", "lang": "en", "score": 0.75, "source": "rjokes"}
```

**`sft/sft_train.jsonl`** — SFT 训练数据（Type A + Type B 混合 & shuffle 后），每行一个 JSON：

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**`reward/preference_train.jsonl`** — Reward Model 偏好对数据，每行一个 JSON：

```json
{"prompt": [{"role": "user", "content": "..."}], "chosen": [{"role": "assistant", "content": "高分笑话"}], "rejected": [{"role": "assistant", "content": "低分笑话"}]}
```

**`grpo/grpo_prompts.jsonl`** — GRPO prompt 数据，每行一个 JSON：

```json
{"prompt": [{"role": "user", "content": "..."}], "headline": "...", "keywords": [...]}
```

---

## 10. 处理流水线调用方式

```bash
# Step 1: 解析所有原始数据 → 统一中间格式 (JSONL)
#   输入: data/raw/* 各原始数据集
#   输出: data/preprocessed/unified_{en,zh,es}.jsonl
python data_preprocessing/pipeline.py --stage parse

# Step 2: 统一中间格式 → SFT 训练数据 (质量筛选 + prompt 配对 + Type A/B 混合)
#   输入: data/preprocessed/unified_*.jsonl + data/synthesized/type_b_*.jsonl
#   输出: data/sft/sft_{train,val}.jsonl
#   注意: Type B 数据需要提前通过 synthesize_task_data.py 生成
python data_preprocessing/pipeline.py --stage format_sft

# Step 3: SemEval 数据 → GRPO prompt 数据
#   输入: data/raw/semeval_task/task-a-*.tsv
#   输出: data/grpo/grpo_prompts.jsonl
python data_preprocessing/pipeline.py --stage format_grpo

# Step 4: 统一中间格式 → Reward Model 偏好对数据
#   输入: data/preprocessed/unified_*.jsonl (利用其中的 score 字段)
#   输出: data/reward/preference_{train,val}.jsonl
python data_preprocessing/pipeline.py --stage format_reward

# 独立步骤: 合成 Type B 数据 (需要网络和 API key，不阻塞其他 stage)
python data_preprocessing/synthesize_task_data.py --lang en --n_samples 500
python data_preprocessing/synthesize_task_data.py --lang zh --n_samples 500
python data_preprocessing/synthesize_task_data.py --lang es --n_samples 500
```

各 stage 可独立运行，也可通过 `--stage all` 一键串联（但需确保 Type B 合成数据已准备好）。

**推荐执行顺序**：
1. `parse` → 所有后续 stage 的基础
2. `format_sft` → 启动 SFT 训练
3. `format_grpo` + `format_reward` → 可并行准备，SFT 训练期间完成
4. SFT 完成后 → 启动 reward model 训练（如需要）→ 启动 GRPO 训练
