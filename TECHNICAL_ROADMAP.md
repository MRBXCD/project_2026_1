# SemEval 2026 Task A: Humor Generation — Technical Roadmap and Implementation Plan

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Technical Architecture Overview](#2-technical-architecture-overview)
- [3. Environment and Dependencies](#3-environment-and-dependencies)
- [4. Data Pipeline](#4-data-pipeline)
  - [4.1 SFT Data Preparation](#41-sft-data-preparation)
  - [4.2 GRPO Training Data and Reward Design](#42-grpo-training-data-and-reward-design)
- [5. Stage 1: Supervised Fine-Tuning (SFT)](#5-stage-1-supervised-fine-tuning-sft)
  - [5.1 SFT Objectives](#51-sft-objectives)
  - [5.2 LoRA Configuration](#52-lora-configuration)
  - [5.3 Training Configuration and Code](#53-training-configuration-and-code)
- [6. Stage 2: Group Relative Policy Optimization (GRPO)](#6-stage-2-group-relative-policy-optimization-grpo)
  - [6.1 GRPO Principle Overview](#61-grpo-principle-overview)
  - [6.2 Reward Function Design (Core)](#62-reward-function-design-core)
  - [6.3 Training Configuration and Code](#63-training-configuration-and-code)
- [7. Inference and Constraint Satisfaction](#7-inference-and-constraint-satisfaction)
  - [7.1 Rejection Sampling](#71-rejection-sampling)
  - [7.2 Inference Pipeline](#72-inference-pipeline)
- [8. Evaluation Scheme](#8-evaluation-scheme)
- [9. Project Directory Structure (Proposed)](#9-project-directory-structure-proposed)
- [10. Timeline and Milestones](#10-timeline-and-milestones)
- [Appendix A: GRPO Mathematical Principles](#appendix-a-grpo-mathematical-principles)
- [Appendix B: FAQ and Hyperparameter Tuning Suggestions](#appendix-b-faq-and-hyperparameter-tuning-suggestions)

---

## 1. Project Overview

| Item | Content |
|---|---|
| **Task** | SemEval 2026 Task A — Generate short humorous text given a news headline (+ optional keyword constraints) |
| **Languages** | English, Chinese, Spanish |
| **Base Model** | Qwen3-8B (Apache 2.0 open source, supports 119 languages) |
| **Training Framework** | HuggingFace TRL + PEFT (LoRA) + Accelerate |
| **Training Process** | SFT → GRPO |
| **Hardware** | Single 80GB GPU (A100/H100) |
| **Training Precision** | bf16 mixed precision |

---

## 2. Technical Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Overall Pipeline Architecture               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐      ┌──────────────┐      ┌─────────────────┐   │
│  │  Raw     │─────▶│  Data Prep   │─────▶│  SFT Dataset    │   │
│  │  Data    │      │(Clean/Format/│      │({prompt,        │   │
│  │(rJokes,  │      │ Synthesize)  │      │  completion})   │   │
│  │ CFun,    │      └──────────────┘      └────────┬────────┘   │
│  │ HAHA,...)│                                       │            │
│  └──────────┘                                       │            │
│                                                     ▼            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Stage 1: SFT (LoRA Fine-Tuning)             │   │
│  │  Qwen3-8B + LoRA → Learn Humor Style + Task Mapping      │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Stage 2: GRPO (Online RL)                   │   │
│  │                                                          │   │
│  │  ┌─────────┐    ┌──────────────┐    ┌────────────────┐  │   │
│  │  │ Policy  │───▶│  Group       │───▶│  Reward        │  │   │
│  │  │ Model   │    │  Sampling    │    │  Function      │  │   │
│  │  │(Post-SFT)│    │  (G=8/group) │    │  (Rule+LLM)    │  │   │
│  │  └────┬────┘    └──────────────┘    └────────┬───────┘  │   │
│  │       │                                       │          │   │
│  │       │         ┌──────────────┐              │          │   │
│  │       └─────────│  GRPO Update │◀─────────────┘          │   │
│  │                 │  (Group-Rel  │                          │   │
│  │                 │   Advantage) │                          │   │
│  │                 └──────────────┘                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           Inference: Rejection Sampling                   │   │
│  │  Generate N candidates → Hard Constraint Filter →        │   │
│  │  Humor Score → Select Best                               │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Environment and Dependencies

### 3.1 Core Dependencies

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

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Evaluation
rouge-score
nltk
scikit-learn

# Experiment Management
wandb
tensorboard
tqdm

# Inference Acceleration (Optional)
vllm>=0.7.0
```

> **Note**: `trl>=0.18.0` is critical because GRPOTrainer matured in newer versions. It is recommended not to specify an upper bound version during installation, just use the latest.

### 3.2 Dockerfile Update Suggestions

The existing Dockerfile is basically usable, but it is recommended to update `trl` and `transformers` versions to ensure GRPO support. Use `pip install --upgrade` during installation to get the latest versions.

### 3.3 Model Download

```bash
# Use huggingface-cli to download model (execute inside container)
huggingface-cli download Qwen/Qwen3-8B --local-dir /workspace/models/Qwen3-8B
```

---

## 4. Data Pipeline

### 4.1 SFT Data Preparation

The core goal of SFT data is to teach the model **two things**:
1. **Humorous Language Style** — From real humor corpus
2. **Task Input-Output Mapping** — From synthesized task-formatted data

#### 4.1.1 Data Sources and Usage

| Dataset | Language | Usage | Format |
|---|---|---|---|
| rJokes (Reddit) | EN | General Humor + Preference Ranking (with scores) | TSV with scores |
| News Headlines Sarcasm | EN | **Deprecated** (Sarcasm ≠ Humor, format mismatch) | - |
| CFun | ZH | General Humor | Arrow (HuggingFace) |
| HAHA 2019 | ES | General Humor | CSV with scores |
| **Synthesized Data** | EN/ZH/ES | **Task Formatted Training** | Generated by strong models |

#### 4.1.2 SFT Data Format

All data is unified into **chat template** format to match Qwen3's chat format:

**Type A: General Humor Data (Teaching Style)**

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

**Type B: Task Formatted Data (Teaching Mapping) — This part is crucial**

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

> **Key**: Type B data needs to be generated through synthesis (see script below). It is recommended that the ratio of Type A to Type B in SFT data be approximately **6:4** or **7:3**.

#### 4.1.3 Data Synthesis Script Idea

Synthesis flow for task-formatted data (Type B):

```python
"""
Example idea for synthesizing task-formatted SFT data (Pseudocode)

Core Idea:
1. Extract headlines from news datasets (e.g., SemEval provided headlines or public news datasets)
2. Randomly pair two low-frequency words from vocabulary as keyword constraints
3. Call strong model (Gemini / GPT-4) to generate humorous responses satisfying constraints
4. Perform quality filtering and store
"""
import random
import json


def build_prompt_for_synthesis(headline: str, word1: str, word2: str, lang: str) -> str:
    """Construct prompt to let strong model generate humorous response"""
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
    """Wrap synthesis result into SFT training format"""
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


# === Synthesis Flow (Conceptual code, needs API call logic) ===
def synthesize_task_data(headlines: list, word_pairs: list, lang: str, 
                          api_call_fn, n_samples: int = 500) -> list:
    """
    headlines: List of news headlines
    word_pairs: [(word1, word2), ...] List of keyword pairs
    lang: Language code "en" / "zh" / "es"
    api_call_fn: Function to call strong model (prompt) -> response
    """
    results = []
    for i in range(n_samples):
        headline = random.choice(headlines)
        w1, w2 = random.choice(word_pairs)
        
        synthesis_prompt = build_prompt_for_synthesis(headline, w1, w2, lang)
        response = api_call_fn(synthesis_prompt)
        
        # Quality Filter: Check if keywords actually appear in response
        if w1.lower() in response.lower() and w2.lower() in response.lower():
            example = build_sft_example(headline, w1, w2, response, lang)
            results.append(example)
    
    return results
```

#### 4.1.4 rJokes Data Processing Points

rJokes dataset comes with scores, which can be used for two things:
1. **SFT**: Filter high-score (e.g., score > 10) jokes as response corpus
2. **Subsequent Reward Model Training** (if needed): Construct preference pairs using high-score vs low-score

```python
"""rJokes Data Preprocessing Idea"""
import pandas as pd

# rJokes TSV format usually is: id, body (setup), score, ...
# Adjust based on actual column names
df = pd.read_csv("data/sft/raw/rjoke/train.tsv.gz", sep="\t", compression="gzip")

# Filter high quality jokes for SFT
high_quality = df[df["score"] > 10].copy()

# Convert to SFT format
sft_data = []
generic_prompts_en = [
    "Tell me a joke.",
    "Make me laugh with a short joke.",
    "Can you tell me something funny?",
    "I need a good laugh. Give me a joke.",
    "Share a humorous one-liner.",
]

for _, row in high_quality.iterrows():
    joke_text = row["body"]  # Adjust based on actual column name
    prompt = random.choice(generic_prompts_en)
    sft_data.append({
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": joke_text}
        ]
    })
```

### 4.2 GRPO Training Data and Reward Design

"Training data" in the GRPO stage is not traditional labeled data, but a **collection of prompts**. The model generates multiple responses (rollouts) for each prompt, which are then scored by a reward function.

#### 4.2.1 GRPO Prompt Sources

| Source | Description |
|---|---|
| SemEval Training Prompts | Headlines and keywords provided in `data/semeval_task/task-a-{en,es,zh}.tsv` |
| Synthesized Prompts | Extra headlines extracted from news datasets + random keyword pairs |

> **Note**: Based on your data, the word1/word2 columns in SemEval TSV files are currently `-` (i.e., keyword constraints are not yet provided). This means for the current stage you can focus on the **headline subtask**, and add keyword constraints later when SemEval releases full data.

#### 4.2.2 Reward Function Design (Core of the Core)

The effectiveness of GRPO training **highly depends on the quality of the reward function**. We design a **composite reward function**, including rule-based terms and model scoring terms:

```python
"""
GRPO Reward Function Design

Reward = R_format + R_keyword + R_relevance + R_humor

Where:
- R_format:  Format compliance (Hard constraint, rule check)
- R_keyword: Keyword inclusion (Hard constraint, rule check)
- R_relevance: Relevance to news headline (Soft constraint, optional)
- R_humor:   Humor level (Soft constraint, LLM-as-Judge or Reward Model)
"""
import re


def reward_format(response: str) -> float:
    """
    Format compliance check
    
    Rules:
    - Must be non-empty text
    - Length within reasonable range (e.g., 10-280 characters)
    - No significant repetition (degeneracy check)
    """
    if not response or not response.strip():
        return -2.0
    
    text = response.strip()
    
    # Length check
    if len(text) < 10:
        return -1.0
    if len(text) > 280:
        return -0.5
    
    # Repetition check (simple n-gram degeneracy check)
    words = text.split()
    if len(words) >= 4:
        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
        unique_ratio = len(set(trigrams)) / len(trigrams)
        if unique_ratio < 0.5:  # More than half of trigrams are repeated
            return -1.5
    
    return 0.5  # Base reward for format compliance


def reward_keyword(response: str, keywords: list[str]) -> float:
    """
    Keyword inclusion check
    
    +1.0 for each included keyword, extra bonus +0.5 if all included
    -1.0 if no keyword is included
    """
    if not keywords:  # Prompt with no keyword constraints
        return 0.0
    
    text = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    
    if hits == 0:
        return -1.0
    elif hits == len(keywords):
        return hits * 1.0 + 0.5  # All hit bonus
    else:
        return hits * 1.0 - 0.5  # Partial hit


def reward_humor_llm_judge(prompt: str, response: str, 
                            judge_fn) -> float:
    """
    Evaluate humor level using external LLM
    
    judge_fn: Function calling external LLM, returning 1-5 score
    
    Note: This call is slow and costly, suggestions:
    - Reduce call frequency in early training (e.g., use LLM judge every N steps)
    - Or substitute with a trained small reward model
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
        # Map 1-5 to [-1, 1] range
        return (score - 3) / 2.0
    except:
        return 0.0  # Neutral score on failure


def compute_reward(prompt: str, response: str, 
                   keywords: list[str] = None,
                   judge_fn=None) -> float:
    """
    Composite Reward Function
    
    Weights can be adjusted experimentally
    """
    r_format = reward_format(response)
    r_keyword = reward_keyword(response, keywords or [])
    
    # If format is severely non-compliant, return low score directly (short-circuit)
    if r_format <= -1.0:
        return r_format
    
    r_humor = 0.0
    if judge_fn is not None:
        r_humor = reward_humor_llm_judge(prompt, response, judge_fn)
    
    # Weighted sum (Weights are hyperparameters, need experimental tuning)
    total = (
        1.0 * r_format +    # Format compliance
        2.0 * r_keyword +    # Keyword inclusion (Higher weight as it's hard constraint)
        1.5 * r_humor        # Humor level
    )
    
    return total
```

> **About LLM-as-Judge Cost**: In GRPO training, calling external API for every rollout is costly and slow. Several strategies in practice:
> 
> 1. **Train a small Reward Model first** (Recommended): Train a lightweight reward model (can be Qwen3-1.7B + classification head) using rJokes score data, then use it for scoring in GRPO.
> 2. **Phased Training**: Use rule-based reward (format + keyword) initially, add humor reward after training stabilizes.
> 3. **Batch Async Calls**: Accumulate a batch of rollouts then call LLM judge asynchronously to reduce API overhead.
> 
> **For course projects, Scheme 2 is recommended** — First run through GRPO flow with pure rule reward, then gradually introduce humor scoring after confirming stability.

---

## 5. Stage 1: Supervised Fine-Tuning (SFT)

### 5.1 SFT Objectives

| Objective | Description |
|---|---|
| Learn Humor Language Style | Acquire humor expression patterns from real joke corpora |
| Learn Task Input-Output Mapping | Understand generation format "Headline + Keywords → Joke" |
| Establish Initial Policy for GRPO | Ensure GRPO starts from a reasonable point, avoiding random policy start |

### 5.2 LoRA Configuration

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                          # LoRA rank, 32-64 recommended for 8B model
    lora_alpha=128,                # Usually set to 2*r
    lora_dropout=0.05,
    target_modules=[               # Attention layers of Qwen3
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    # Note: Do not add "lm_head", fine-tuning attention + FFN is enough for SFT
)
```

> **LoRA rank selection basis**: For 8B model, r=64 produces ~160M trainable parameters (~2% of total), no VRAM pressure on 80GB GPU. If overfitting occurs, reduce to r=32.

### 5.3 Training Configuration and Code

```python
"""
SFT Training Script Skeleton

File: sft/train_sft.py
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTConfig, SFTTrainer


def main():
    # ============================================================
    # 1. Load Model and Tokenizer
    # ============================================================
    model_name = "Qwen/Qwen3-8B"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",        # Padding on right for SFT training
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",           # Single card directly auto
        trust_remote_code=True,
        attn_implementation="flash_attention_2",  # Use FlashAttention 2
    )

    # ============================================================
    # 2. LoRA Configuration
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
    # 3. Load Dataset
    # ============================================================
    # Dataset should be JSON/JSONL format, each containing "messages" field
    # Example: {"messages": [{"role": "user", "content": "..."}, 
    #                        {"role": "assistant", "content": "..."}]}
    dataset = load_dataset(
        "json",
        data_files={
            "train": "data/sft/preprocessed/sft_train.jsonl",
            "validation": "data/sft/preprocessed/sft_val.jsonl",
        }
    )

    # ============================================================
    # 4. Training Configuration
    # ============================================================
    training_args = SFTConfig(
        output_dir="checkpoints/sft",
        
        # --- Training Hyperparams ---
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,    # Effective batch_size = 4 * 4 = 16
        learning_rate=2e-4,               # LoRA LR usually higher than full fine-tuning
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        # --- Precision ---
        bf16=True,
        
        # --- Sequence Length ---
        max_seq_length=512,               # Jokes usually short, 512 is enough
        
        # --- Logging and Saving ---
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        
        # --- Others ---
        report_to="wandb",               # or "tensorboard"
        seed=42,
        
        # --- PEFT ---
        peft_config=lora_config,          # TRL >= 0.18 pass LoRA config directly
    )

    # ============================================================
    # 5. Create Trainer and Train
    # ============================================================
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    trainer.train()
    
    # Save final model (Only saves LoRA adapter weights)
    trainer.save_model("checkpoints/sft/final")
    tokenizer.save_pretrained("checkpoints/sft/final")


if __name__ == "__main__":
    main()
```

### 5.4 Key Notes for SFT Stage

1. **Qwen3 Thinking Mode**: Qwen3 defaults to "thinking" mode (generates thinking process in `<think>...</think>` tags). For humor generation tasks, it is recommended to **disable thinking mode** during inference (specify `/no_think` in system prompt, or configure in generation parameters). Assistant responses in SFT data should not contain thinking tags.

2. **Multilingual Mixed Training**: Train English/Chinese/Spanish SFT data mixed together (not separately). Qwen3 itself has strong multilingual capabilities, mixed training can mutually benefit.

3. **Data Volume Suggestion**: Approx 3000-8000 SFT samples in total (too many might overfit to specific joke patterns).

---

## 6. Stage 2: Group Relative Policy Optimization (GRPO)

### 6.1 GRPO Principle Overview

Core idea of GRPO (from DeepSeek-R1 paper):

1. **Group Sampling**: For each prompt $x$, generate a group of $G$ responses $\{y_1, y_2, \ldots, y_G\}$ using current policy $\pi_\theta$
2. **Reward Calculation**: Calculate score $r_i = R(x, y_i)$ for each response using reward function
3. **Group-Relative Advantage**: Normalize using group mean and std dev to get advantage estimate:
   $$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\}) + \epsilon}$$
4. **Policy Update**: Update policy using PPO-clip style objective, but without value model:
   $$\mathcal{L} = -\mathbb{E}\left[\min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1\pm\epsilon)\hat{A}_i\right)\right] + \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$$
   Where $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}$ is importance ratio.

**Advantages over PPO**:
- **No Value Model**: Saves cost and instability of training a critic
- **No separate reward model forward pass**: Reward can be calculated directly by rule functions
- **More Stable**: Group normalization naturally provides baseline, reducing variance

### 6.2 Reward Function Design

See detailed design in [Section 4.2.2](#422-reward-function-design-core).

**Phased Training Strategy (Recommended)**:

| Phase | Reward Composition | Training Steps | Goal |
|---|---|---|---|
| Phase 1 | `R_format + R_keyword` (Pure Rules) | ~200-500 steps | Learn to satisfy hard constraints |
| Phase 2 | `R_format + R_keyword + R_humor` (Rule+LLM) | ~300-800 steps | Improve humor quality while satisfying constraints |

> The benefit of this phased strategy is: Run through the flow and tune hyperparameters with cheap rule rewards first, then introduce expensive LLM judge. Avoid burning money on tuning at the start.

### 6.3 Training Configuration and Code

```python
"""
GRPO Training Script Skeleton

File: rl/train_grpo.py

TRL's GRPOTrainer encapsulates GRPO core logic:
- Automatically handles group sampling (generate G responses per prompt)
- Automatically computes group-relative advantage
- Automatically handles KL divergence constraints
- Supports custom reward functions
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer


# ============================================================
# 1. Reward Function (Core component passed to GRPOTrainer)
# ============================================================
def reward_fn(completions: list[str], prompts: list[str] = None, 
              **kwargs) -> list[float]:
    """
    Reward function signature required by GRPOTrainer:
    - completions: List of generated responses
    - prompts: List of corresponding prompts (Supported in TRL >= 0.18)
    
    Returns: List of floats same length as completions
    
    Note: GRPOTrainer combines prompt and completion internally before passing,
    check documentation for specific signature of your installed TRL version.
    """
    rewards = []
    for i, completion in enumerate(completions):
        prompt = prompts[i] if prompts else ""
        
        # Extract keywords from prompt (if any)
        keywords = extract_keywords_from_prompt(prompt)
        
        # Compute composite reward
        r = compute_reward(
            prompt=prompt,
            response=completion,
            keywords=keywords,
            judge_fn=None,  # Phase 1: No LLM judge
        )
        rewards.append(r)
    
    return rewards


def extract_keywords_from_prompt(prompt: str) -> list[str]:
    """Extract keyword constraints from prompt text"""
    import re
    # Match "Required words: word1, word2" format
    match = re.search(r"Required words?:\s*(.+?)(?:\n|$)", prompt, re.IGNORECASE)
    if not match:
        # Try matching Chinese format
        match = re.search(r"必须包含的词语[：:]\s*(.+?)(?:\n|$)", prompt)
    if not match:
        return []
    
    words_str = match.group(1).strip()
    # Split by comma or enumeration comma
    keywords = re.split(r"[,，、]", words_str)
    return [kw.strip() for kw in keywords if kw.strip()]


# ============================================================
# 2. Main Training Flow
# ============================================================
def main():
    # --- Load Model Trained in SFT Stage ---
    # Method: Load base model + merge SFT LoRA adapter
    base_model_name = "Qwen/Qwen3-8B"
    sft_adapter_path = "checkpoints/sft/final"

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        padding_side="left",          # Left padding needed for GRPO generation
    )
    # Ensure pad_token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    # Load and merge SFT adapter -> As initial policy and reference model for GRPO
    # Note: TRL GRPOTrainer automatically handles reference model
    # You can choose:
    #   A) Merge SFT adapter to base, then wrap a new LoRA for GRPO
    #   B) Pass SFT adapter path directly, let GRPOTrainer handle
    # Showing Scheme A here (Clearer):
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, sft_adapter_path)
    model = model.merge_and_unload()  # Merge LoRA to base weights

    # New LoRA config for GRPO stage (Can use smaller rank)
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

    # --- Load GRPO Training Prompts ---
    # Dataset only needs "prompt" field
    # Format: {"prompt": [{"role": "user", "content": "..."}]}
    dataset = load_dataset(
        "json",
        data_files="data/grpo/grpo_prompts.jsonl",
        split="train"
    )

    # --- GRPO Training Configuration ---
    grpo_config = GRPOConfig(
        output_dir="checkpoints/grpo",
        
        # --- GRPO Core Hyperparams ---
        num_generations=8,            # G: Number of responses per prompt
        max_completion_length=256,    # Max generation length
        
        # --- Training Hyperparams ---
        num_train_epochs=1,           # GRPO usually runs 1-2 epochs
        per_device_train_batch_size=1,# GRPO batch_size usually small
                                       # (because each sample generates G items)
        gradient_accumulation_steps=8,# Effective batch = 1 * 8 = 8 prompts
        learning_rate=5e-6,           # RL stage LR much smaller than SFT
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        
        # --- KL Divergence Constraint ---
        beta=0.04,                    # KL penalty coefficient, prevents deviating too far from SFT policy
                                       # Too large → Policy doesn't update; Too small → Reward hacking
        
        # --- Precision ---
        bf16=True,
        
        # --- Logging and Saving ---
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        report_to="wandb",
        seed=42,
        
        # --- PEFT ---
        peft_config=grpo_lora_config,
    )

    # --- Create Trainer and Train ---
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_fn,       # Custom reward function
    )

    trainer.train()
    
    # Save
    trainer.save_model("checkpoints/grpo/final")
    tokenizer.save_pretrained("checkpoints/grpo/final")


if __name__ == "__main__":
    main()
```

### 6.4 GRPO Training Key Hyperparameters and Tuning Guide

| Hyperparameter | Suggested Range | Description |
|---|---|---|
| `num_generations` (G) | 4 - 16 | Group size. Larger reduces variance but linearly increases VRAM/time. **Start with 8** |
| `beta` (KL coeff) | 0.01 - 0.1 | Core hyperparam. Too large prevents policy update; too small causes reward hacking. **Start with 0.04** |
| `learning_rate` | 1e-6 - 1e-5 | RL stage 1-2 orders of magnitude smaller than SFT. **Suggest 5e-6** |
| `per_device_train_batch_size` | 1 - 2 | Actual throughput = batch × G. Set small to prevent OOM |
| `gradient_accumulation_steps` | 4 - 16 | Increase effective batch via accumulation for stability |
| `max_completion_length` | 128 - 512 | Jokes are short, 256 usually enough |
| `temperature` (Generation) | 0.7 - 1.0 | GRPO needs diversity, don't set too low. **Suggest 0.9** |

### 6.5 GRPO Training Monitoring Metrics

Monitor these metrics (via wandb/tensorboard):

| Metric | Normal Trend | Abnormal Signal |
|---|---|---|
| `reward/mean` | Slowly increasing | Rapid spike → reward hacking |
| `reward/std` | Gradually decreasing | Consistently high → unstable training |
| `kl_divergence` | Slowly increasing, moderate | Explosion → reduce lr or increase beta |
| `policy_loss` | Fluctuating but generally decreasing | Flatline → lr too small or reward signal too weak |
| `completion_length` | Relatively stable | Trending to max_length → model padding or rambling |

---

## 7. Inference and Constraint Satisfaction

### 7.1 Rejection Sampling

After training, use rejection sampling during inference to guarantee constraint satisfaction:

```python
"""
Rejection Sampling during Inference

File: rl/inference.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_model(base_model_path: str, adapter_path: str):
    """Load GRPO trained model"""
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Load GRPO adapter
    # Note: If GRPO was trained on SFT-merged model,
    # Need to merge SFT adapter first then load GRPO adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_candidates(model, tokenizer, prompt: str, 
                         n_candidates: int = 16,
                         max_new_tokens: int = 256,
                         temperature: float = 0.9,
                         top_p: float = 0.95) -> list[str]:
    """Generate N candidate responses for a single prompt"""
    messages = [{"role": "user", "content": prompt}]
    
    # Use Qwen3 chat template
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=False,  # Disable thinking mode
    )
    
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        num_return_sequences=n_candidates,  # Generate N at once
    )
    
    # Decode (Remove prompt part)
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
    Rejection Sampling Inference Flow
    
    Returns: {
        "best_response": str,
        "all_candidates": list[str],
        "constraint_pass_rate": float,
        "scores": list[float]
    }
    """
    # Step 1: Generate candidates
    candidates = generate_candidates(model, tokenizer, prompt, n_candidates)
    
    # Step 2: Hard constraint filter
    if keywords:
        valid_candidates = []
        for c in candidates:
            c_lower = c.lower()
            if all(kw.lower() in c_lower for kw in keywords):
                valid_candidates.append(c)
    else:
        valid_candidates = candidates.copy()
    
    constraint_pass_rate = len(valid_candidates) / len(candidates)
    
    # Step 3: Fallback if no candidate passes hard constraints
    if not valid_candidates:
        # Fallback: Pick candidate with most keywords
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
    
    # Step 4: Soft constraint ranking (Humor score)
    if humor_scorer:
        scored = [(c, humor_scorer(prompt, c)) for c in valid_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        best = scored[0][0]
        scores = [s for _, s in scored]
    else:
        # Pick random valid candidate if no scorer
        best = valid_candidates[0]
        scores = []
    
    return {
        "best_response": best,
        "all_candidates": candidates,
        "constraint_pass_rate": constraint_pass_rate,
        "scores": scores,
    }


# === Usage Example ===
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

### 7.2 Accelerate Inference with vLLM (Optional)

Using vLLM significantly speeds up inference when rejection sampling is needed for many prompts:

```python
"""
Batch Inference with vLLM (5-10x faster than HuggingFace generate)
Note: vLLM LoRA adapter support requires vLLM >= 0.4.0
"""
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Load Model (vLLM way)
llm = LLM(
    model="Qwen/Qwen3-8B",
    enable_lora=True,
    max_lora_rank=64,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

# Configure Sampling Params
sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.95,
    max_tokens=256,
    n=16,  # Generate 16 candidates per prompt
)

# Prepare LoRA adapter
lora_request = LoRARequest("grpo_adapter", 1, "checkpoints/grpo/final")

# Batch Inference
prompts = [...]  # List of prompts
outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
```

---

## 8. Evaluation Scheme

### 8.1 Automated Metrics (Tier 1 — No Model Required)

| Metric | Applicable Subtask | Calculation Method |
|---|---|---|
| Constraint Satisfaction Rate | Keyword Subtask | Whether both keywords appear (Exact/Fuzzy match) |
| Format Compliance | All | Single sentence/Length/Non-empty check |
| Degeneracy Rate | All | Percentage of repeated n-grams |
| Distinct-1 / Distinct-2 | All | Unigram/Bigram diversity of generated text |

### 8.2 LLM-as-Judge (Tier 2)

```python
"""
LLM-as-Judge Pairwise Evaluation

Compare baseline (Zero-shot base) vs proposed (Trained model) outputs for each prompt
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

### 8.3 Human Evaluation (Tier 3)

- Randomly sample 20-40 prompts from test set
- 2-3 evaluators perform blind A/B testing independently
- Report agreement with LLM judge (Cohen's kappa)

---

## 9. Project Directory Structure (Proposed)

```
proj_2026_1/
├── configs/                        # Training Configs
│   ├── sft_config.yaml
│   └── grpo_config.yaml
│
├── data/
│   ├── semeval_task/               # SemEval Raw Data (Existing)
│   │   ├── task-a-en.tsv
│   │   ├── task-a-es.tsv
│   │   └── task-a-zh.tsv
│   ├── sft/
│   │   ├── raw/                    # Raw Datasets (Existing)
│   │   │   ├── rjoke/
│   │   │   ├── cfun/
│   │   │   ├── haha/
│   │   │   └── news_headlines.../
│   │   └── preprocessed/           # Processed SFT Data
│   │       ├── sft_train.jsonl
│   │       └── sft_val.jsonl
│   └── grpo/
│       └── grpo_prompts.jsonl      # GRPO Training Prompts
│
├── data_preprocessing/
│   ├── visulization.ipynb          # Data Visualization (Existing)
│   ├── prepare_sft_data.py         # SFT Data Prep Script
│   ├── prepare_grpo_prompts.py     # GRPO Prompt Prep Script
│   └── synthesize_task_data.py     # Task Formatted Data Synthesis Script
│
├── sft/
│   ├── train_sft.py                # SFT Training Entry
│   └── eval_sft.py                 # SFT Evaluation Entry
│
├── rl/
│   ├── rewards.py                  # Reward Function Definition
│   ├── train_grpo.py               # GRPO Training Entry
│   └── inference.py                # Inference + Rejection Sampling
│
├── src/
│   ├── __init__.py
│   ├── data_utils.py               # Data Loading/Processing Utils
│   └── eval_utils.py               # Evaluation Utils
│
├── evaluation/
│   ├── run_auto_eval.py            # Automated Metrics Eval
│   ├── run_llm_judge.py            # LLM-as-Judge Eval
│   └── analyze_results.py          # Results Analysis & Viz
│
├── checkpoints/                    # Model Checkpoints (Existing)
│   ├── sft/
│   └── grpo/
│
├── utils/                          # Common Utils (Existing)
│
├── Dockerfile
├── docker_build.sh
├── docker_run.sh
├── requirements.txt
├── TECHNICAL_ROADMAP.md            # This Document
└── .gitignore
```

---

## 10. Timeline and Milestones

### Week 1: Data Prep + Environment Setup

| Task | Est. Time | Deliverable |
|---|---|---|
| Environment Setup (Docker + Deps) | 0.5 Day | Runnable Container Environment |
| Data Exploration & Visualization | 1 Day | EDA Notebook |
| rJokes / CFun / HAHA Cleaning | 1 Day | Cleaned General Humor Data |
| Task Formatted Data Synthesis | 1.5 Day | Synthesized SFT Data |
| SFT Dataset Construction + Verification | 1 Day | `sft_train.jsonl`, `sft_val.jsonl` |

### Week 2: SFT + GRPO Training

| Task | Est. Time | Deliverable |
|---|---|---|
| SFT Training Script Writing | 0.5 Day | `train_sft.py` |
| SFT Training + Tuning | 1 Day | SFT Checkpoint |
| Reward Function Implementation | 1 Day | `rl/rewards.py` |
| GRPO Training Script Writing | 0.5 Day | `train_grpo.py` |
| GRPO Phase 1 (Pure Rule Reward) | 1 Day | GRPO Phase 1 Checkpoint |
| GRPO Phase 2 (Add Humor Score) | 1 Day | GRPO Phase 2 Checkpoint |

### Week 3: Evaluation + Tuning

| Task | Est. Time | Deliverable |
|---|---|---|
| Inference + Rejection Sampling | 0.5 Day | `inference.py` |
| Automated Metrics Evaluation | 0.5 Day | Metrics Report |
| LLM-as-Judge Evaluation | 1 Day | Pairwise Win-Rate Report |
| Human Evaluation | 1 Day | Human Eval Report |
| Model Tuning Iteration | 2 Days | Final Model |

### Week 4: Report + Consolidation

| Task | Est. Time | Deliverable |
|---|---|---|
| Code Cleanup + Documentation | 1 Day | Clean Code Repository |
| Final Report Writing | 3-4 Days | Course Paper |

---

## Appendix A: GRPO Mathematical Principles

### A.1 Objective Function

GRPO Policy Gradient Objective Function:

$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, \{y_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|x)} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)} \hat{A}_i,\; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{\text{old}}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) \right]
$$

Where $\hat{A}_i$ is the group-relative advantage:

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G + \epsilon}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \mu_G)^2}
$$

### A.2 KL Divergence Regularization

To prevent policy from deviating too far from the reference model (SFT model), KL penalty is added:

$$
\mathcal{L}_{\text{total}} = -\mathcal{J}_{\text{GRPO}}(\theta) + \beta \cdot \mathbb{E}_{x}\left[D_{\text{KL}}\left(\pi_\theta(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right)\right]
$$

$\beta$ controls the strength of KL penalty:
- $\beta$ too large → Policy barely updates, equivalent to no training
- $\beta$ too small → Policy updates too aggressively, possibly leading to reward hacking

### A.3 Comparison with PPO

| Feature | PPO | GRPO |
|---|---|---|
| Value Model | Required (Extra Critic training) | **Not Required** |
| Advantage Estimation | GAE (Requires value function) | **Group Normalization** (No value function needed) |
| VRAM Usage | High (Policy + Value + Ref models) | Low (Policy + Ref models) |
| Training Stability | Requires balancing Actor/Critic | Relatively more stable |
| Usage Scenario | General RL | Especially suitable for scenarios with rule-based rewards |

---

## Appendix B: FAQ and Hyperparameter Tuning Suggestions

### B.1 GRPO Training Reward Not Increasing

**Possible Causes and Solutions:**

| Symptom | Cause | Solution |
|---|---|---|
| Reward completely flat | LR too small | Increase LR (e.g., 5e-6 → 1e-5) |
| Reward flat | Beta too large | Decrease beta (e.g., 0.1 → 0.04) |
| Reward flat | Reward function not distinctive | Check if generated responses all get similar scores |
| Reward flat | SFT model too poor | Check SFT generation quality, might need more/better SFT data |

### B.2 Reward Hacking

**Symptom**: Reward rises quickly but generation quality drops (Model finds loophole in reward function)

**Common Cases and Countermeasures:**
- **Model repeats keywords**: e.g., "penguin penguin penguin bankruptcy" → Add stronger repetition penalty in `reward_format`
- **Model generates extremely long text**: → Add length penalty in reward
- **Model output not like a joke**: → Increase beta to constrain deviation from SFT policy, or introduce humor reward

### B.3 Out of Memory (OOM)

**GRPO stage is more prone to OOM than SFT** because it processes a group of responses simultaneously. Tuning order:

1. Decrease `num_generations` (e.g., 8 → 4)
2. Decrease `per_device_train_batch_size` (e.g., 2 → 1)
3. Increase `gradient_accumulation_steps` to compensate
4. Decrease `max_completion_length`
5. Enable gradient checkpointing (`gradient_checkpointing=True`)
6. Use QLoRA (4-bit quantization) instead of LoRA

### B.4 Qwen3 Thinking Mode Handling

Qwen3 thinking mode generates `<think>...</think>` tags in response. For humor generation, **must disable**:

```python
# Method 1: Disable in chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,   # Disable thinking mode
)

# Method 2: Specify in system prompt
messages = [
    {"role": "system", "content": "/no_think"},
    {"role": "user", "content": "your prompt here"},
]
```

In SFT data and GRPO prompt construction, **uniformly use `/no_think` system prompt** to maintain consistency.
