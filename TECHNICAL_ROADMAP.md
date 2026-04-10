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
  - [5.3 Training Configuration](#53-training-configuration)
- [6. Stage 2: Group Relative Policy Optimization (GRPO)](#6-stage-2-group-relative-policy-optimization-grpo)
  - [6.1 GRPO Principle Overview](#61-grpo-principle-overview)
  - [6.2 Reward Function Design](#62-reward-function-design)
  - [6.3 Training Configuration](#63-training-configuration)
- [7. Inference and Constraint Satisfaction](#7-inference-and-constraint-satisfaction)
  - [7.1 Rejection Sampling](#71-rejection-sampling)
  - [7.2 Inference Pipeline](#72-inference-pipeline)
- [8. Evaluation Scheme](#8-evaluation-scheme)
  - [8.1 Pipeline Overview](#81-pipeline-overview)
  - [8.2 General Capability Benchmark](#82-general-capability-benchmark)
  - [8.3 Output Generation with Rejection Sampling](#83-output-generation-with-rejection-sampling)
  - [8.4 Automated Metrics (Tier 1)](#84-automated-metrics-tier-1)
  - [8.5 LLM-as-Judge Pairwise Comparison (Tier 2)](#85-llm-as-judge-pairwise-comparison-tier-2)
  - [8.6 Human Evaluation (Tier 3)](#86-human-evaluation-tier-3)
  - [8.7 Report Aggregation](#87-report-aggregation)
- [9. Project Directory Structure](#9-project-directory-structure)
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
│  │  │(Post-SFT)│    │ (G=16/group) │    │  (Rule+LLM)    │  │   │
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

Key packages: `torch>=2.4.0`, `transformers>=4.51.0`, `accelerate>=1.2.0`, `peft>=0.14.0`, `trl>=0.29.0`, `datasets>=3.0.0`, `flash-attn>=2.7.0`. Full dependency list is in `pyproject.toml`.

> **Note**: `trl>=0.29.0` is required for the GRPOTrainer API used in this project. It is recommended not to specify an upper bound version during installation, just use the latest.

### 3.2 Dockerfile Update Suggestions

The existing Dockerfile is basically usable, but it is recommended to update `trl` and `transformers` versions to ensure GRPO support. Use `pip install --upgrade` during installation to get the latest versions.

### 3.3 Model Download

Use `huggingface-cli download Qwen/Qwen3-8B` to download the base model inside the container.

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

> **Key**: Type B data needs to be generated through synthesis. It is recommended that the ratio of Type A to Type B in SFT data be approximately **6:4** or **7:3**.

#### 4.1.3 Data Synthesis Strategy

Type B task-formatted data synthesis flow (implemented in `data_preprocessing/synthesize_task_data.py`):
1. Extract headlines from news datasets (Babel Briefings)
2. Randomly pair keywords from language-specific vocabulary pools
3. Call Gemini API to generate humorous responses satisfying constraints
4. Quality filter: verify keyword inclusion, language match, and response quality

#### 4.1.4 rJokes Data Processing Points

rJokes dataset comes with scores, which can be used for two things:
1. **SFT**: Filter high-score jokes as response corpus (score normalization via log-cap in `data_preprocessing/parsers.py`)
2. **Reward Model Training**: Construct preference pairs using high-score vs low-score/synthesized-boring texts

### 4.2 GRPO Training Data and Reward Design

"Training data" in the GRPO stage is not traditional labeled data, but a **collection of prompts**. The model generates multiple responses (rollouts) for each prompt, which are then scored by a reward function.

#### 4.2.1 GRPO Prompt Sources

| Source | Description |
|---|---|
| SemEval Training Prompts | Headlines and keywords provided in `data/semeval_task/task-a-{en,es,zh}.tsv` |
| Synthesized Prompts | Extra headlines extracted from news datasets + random keyword pairs |

> **Note**: Based on your data, the word1/word2 columns in SemEval TSV files are currently `-` (i.e., keyword constraints are not yet provided). This means for the current stage you can focus on the **headline subtask**, and add keyword constraints later when SemEval releases full data.

#### 4.2.2 Reward Function Design (Core of the Core)

The effectiveness of GRPO training **highly depends on the quality of the reward function**. We design a **composite reward function**:

$$R_{total} = 1.0 \cdot R_{format} + 2.0 \cdot R_{keyword} + 0.5 \cdot R_{relevance} + 1.5 \cdot R_{humor}$$

| Component | Type | Description |
|---|---|---|
| `R_format` | Hard constraint (rule) | Length check, non-empty, repetition check. Accumulates penalties for compound failures. |
| `R_keyword` | Hard constraint (rule) | +1.0 per keyword hit, bonus for all-hit, -1.0 for zero hits. |
| `R_relevance` | Soft constraint (rule) | Triangular overlap curve peaking at ~30% headline-response token overlap. Weight intentionally low (0.5) as word overlap is a noisy proxy. |
| `R_humor` | Soft constraint (scorer) | Phase 1: returns 0.0 (no scorer). Phase 2: external scorer via `--use_humor_judge` or `--use_reward_model`. Output clamped to [-1, 1]. |

Format short-circuits: if `R_format <= -1.0`, returns early without computing other components.

See `rl/rewards.py` for full implementation. The `build_reward_fn()` factory creates a TRL-compatible closure.

> **About LLM-as-Judge Cost**: In GRPO training, calling external API for every rollout is costly and slow. Three strategies are implemented:
>
> 1. **Trained Reward Model** (Implemented, `rl/train_reward_model.py`): A lightweight reward model (Qwen3-1.7B + classification head) trained on humor preference pair data with Bradley-Terry loss. Fast, deterministic, and task-specific. Activated via `--use_reward_model`.
> 2. **Phased Training** (Default): Use rule-based reward (format + keyword + relevance) initially, add humor reward after training stabilizes. Phase 2 activated via `--use_humor_judge` or `--use_reward_model`.
> 3. **Batch API Calls** (Implemented, `rl/humor_judge.py`): Multiple (prompt, response) pairs packed into a single Gemini API call. Activated via `--use_humor_judge`.

---

## 5. Stage 1: Supervised Fine-Tuning (SFT)

### 5.1 SFT Objectives

| Objective | Description |
|---|---|
| Learn Humor Language Style | Acquire humor expression patterns from real joke corpora |
| Learn Task Input-Output Mapping | Understand generation format "Headline + Keywords → Joke" |
| Establish Initial Policy for GRPO | Ensure GRPO starts from a reasonable point, avoiding random policy start |

### 5.2 LoRA Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `r` | 64 | ~160M trainable parameters (~2% of 8B total), no VRAM pressure on 80GB |
| `lora_alpha` | 128 | 2×r convention |
| `lora_dropout` | 0.0 | No dropout needed at this scale |
| `target_modules` | q/k/v/o_proj, gate/up/down_proj | Full attention + FFN coverage |

> If overfitting occurs, reduce rank to r=32.

### 5.3 Training Configuration

See `sft/train_sft.py` for full implementation.

| Parameter | Value | Notes |
|---|---|---|
| `num_train_epochs` | 1 | Single pass to avoid overfitting on humor patterns |
| `per_device_train_batch_size` | 8 | |
| `gradient_accumulation_steps` | 1 | Effective batch = 8 |
| `learning_rate` | 1e-4 | LoRA LR, higher than full fine-tuning |
| `lr_scheduler_type` | cosine | |
| `warmup_ratio` | 0.05 | |
| `max_seq_length` | 512 | Jokes are short, 512 is enough |
| `packing` | True | Sequence packing for training efficiency |
| `bf16` | True | Mixed precision |
| `gradient_checkpointing` | False | Fits within A100 80GB |

TRL `SFTTrainer` with `peft_config` parameter handles LoRA integration directly (TRL >= 0.29).

### 5.4 Key Notes for SFT Stage

1. **Qwen3 Thinking Mode**: Qwen3 defaults to "thinking" mode (generates `<think>...</think>` tags). For humor generation, **disable thinking mode** by using `/no_think` in the system prompt and setting `enable_thinking=False` in generation parameters. SFT data should not contain thinking tags.

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

| Phase | Reward Composition | Activation | Goal |
|---|---|---|---|
| Phase 1 | `R_format + R_keyword + R_relevance` (Rule-based) | Default | Learn hard constraints + basic headline grounding |
| Phase 2 | `R_format + R_keyword + R_relevance + R_humor` (Rule+Scorer) | `--use_humor_judge` or `--use_reward_model` | Improve humor quality via external scorer |

> Phase 2 is fully implemented with two backends: Gemini LLM-as-Judge (`--use_humor_judge`) and trained reward model (`--use_reward_model`). The reward model approach is preferred for speed and determinism.

### 6.3 Training Configuration

See `rl/train_grpo.py` for full implementation.

**Model Loading**: Currently loads base Qwen3-8B directly (SFT merge is disabled on the `grpo_without_sft` branch). GRPOTrainer applies a new LoRA (r=32) and creates the reference model internally.

**Key Hyperparameters (A100 80GB)**:

| Parameter | Value | Rationale |
|---|---|---|
| `num_generations` (G) | 16 | Group size for advantage estimation |
| `per_device_train_batch_size` | 8 | 8 prompts × 16 generations = 128 completions/step |
| `gradient_accumulation_steps` | 2 | Effective batch = 16 prompts |
| `num_train_epochs` | 2 | Two passes over the prompt dataset |
| `learning_rate` | 5e-6 | ~40× smaller than SFT; RL needs cautious updates |
| `beta` (KL coeff) | 0.04 | Prevents policy from deviating too far from reference |
| `loss_type` | "grpo" | Classic GRPO (not DAPO default) |
| `temperature` | 0.9 | Diversity for meaningful group advantages |
| `max_completion_length` | 256 | Jokes are short |
| `gradient_checkpointing` | False | Fits within A100 80GB |
| GRPO LoRA rank | 32 | Smaller than SFT (64); fine-grained steering |
| `chat_template_kwargs` | `{"enable_thinking": False}` | Disable Qwen3 thinking mode |

**Monitoring** (`RewardStatsRecorder` + `LoggingGRPOTrainer`): Per-component reward statistics (format, keyword, relevance, humor) are logged at each training step via wandb/tensorboard.

### 6.4 GRPO Training Key Hyperparameters and Tuning Guide

| Hyperparameter | Suggested Range | Description |
|---|---|---|
| `num_generations` (G) | 4 - 16 | Group size. Larger reduces variance but linearly increases VRAM/time. **Default 16** |
| `beta` (KL coeff) | 0.01 - 0.1 | Core hyperparam. Too large prevents policy update; too small causes reward hacking. **Default 0.04** |
| `learning_rate` | 1e-6 - 1e-5 | RL stage 1-2 orders of magnitude smaller than SFT. **Default 5e-6** |
| `per_device_train_batch_size` | 1 - 8 | Actual throughput = batch × G. **Default 8** (fits A100 80GB) |
| `gradient_accumulation_steps` | 1 - 8 | Increase effective batch via accumulation for stability. **Default 2** |
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

After training, use rejection sampling during inference to guarantee constraint satisfaction. See `rl/inference.py` for full implementation.

**Strategy**: Bounded retry with up to 3 rounds of generation (16 candidates per round, accumulating up to 48 total). Each round checks hard constraints (keyword inclusion). If valid candidates are found, the best is selected by composite reward score. If no valid candidate after 3 rounds, a best-effort fallback is returned.

- **Hard constraint (filter)**: Keyword inclusion only
- **Soft constraint (ranking)**: Composite reward score (includes headline relevance)

### 7.2 Accelerate Inference with vLLM (Optional)

Using vLLM significantly speeds up inference when rejection sampling is needed for many prompts. vLLM supports LoRA adapter loading (>= 0.4.0) and batch generation with `n=16` candidates per prompt in a single call.

---

## 8. Evaluation Scheme

评估系统是一个六阶段的统一 pipeline（`evaluation/pipeline.py`），各阶段可独立运行也可按顺序串联执行。评估对象为三个模型变体：Base（零样本基线）、SFT、GRPO，在评估集（`data/grpo/grpo_prompts_eval.jsonl`）上分别生成输出后进行对比。

### 8.1 Pipeline Overview

```
评估提示词
    ↓
[Stage 1] benchmark — lm-eval 通用能力基准测试（MMLU 等）
    ↓
[Stage 2] generate  — 三个模型分别生成输出（rejection sampling）
    ↓
    ├→ [Stage 3] auto_metrics — 自动化指标计算
    ├→ [Stage 4] llm_judge   — LLM 裁判成对比较
    └→ [Stage 5] human_eval  — 导出盲评样本供人工评估
                    ↓
           [Stage 6] report — 汇总所有结果生成 Markdown 报告
```

### 8.2 General Capability Benchmark

使用 `lm-evaluation-harness` 在标准化基准（默认 MMLU）上测试三个模型变体，确保 SFT 和 GRPO 训练未显著损害通用语言能力。输出包含各子任务得分以及模型间差异（top-K 改善 / 退化子任务）。

### 8.3 Output Generation with Rejection Sampling

对每条评估提示词，每个模型生成 N=16 个候选（采样模式），经关键词硬约束过滤后，使用 `rl/rewards.py` 中的复合奖励函数对剩余候选打分并选取最优。记录每条提示词的最佳回复、全部候选、约束通过率和最佳得分，作为后续评估阶段的输入。

### 8.4 Automated Metrics (Tier 1)

规则化指标，无需模型推理，全量计算并按语言分组（en / zh / es）：

| Metric | Description |
|---|---|
| Format Compliance | 长度在 [10, 280] 字符且 trigram 唯一率 ≥ 0.5 的回复占比 |
| Degeneracy Rate | trigram 唯一率 < 0.5 的回复占比（重复退化检测） |
| Distinct-1 / Distinct-2 | 全评估集上的 unigram / bigram 多样性 |
| Keyword Satisfaction | 含关键词约束的提示词中，所有关键词均出现的回复占比 |
| Length Statistics | 平均 / 中位数 / 最小 / 最大字符数 |

### 8.5 LLM-as-Judge Pairwise Comparison (Tier 2)

使用 Gemini（`gemini-3-flash-preview`）作为裁判，对模型对（默认 base:grpo、sft:grpo、base:sft）进行成对幽默质量比较。

**评判标准**：幽默质量、与标题的相关性、句子完整性。裁判对每对回复给出 A / B / TIE 判定。

**位置偏差消除**：每对回复进行两轮判定（A/B 位置互换），仅两轮结论一致时计为有效胜出，不一致记为 TIE，并跟踪一致率（consistency rate）。

**输出指标**：每个模型对的胜率、平局率、一致率。

### 8.6 Human Evaluation (Tier 3)

从评估集中按语言分层采样（默认 36 条），随机分配两个模型的回复到 A/B 列（盲评），导出 CSV 供人工评估者填写判定（A / B / TIE）。答案密钥（JSON）单独保存用于解码。

**工作流**：导出盲评 CSV → 评估者独立填写 → 回收 CSV → 报告阶段自动解码计算胜率。

### 8.7 Report Aggregation

汇总以上所有已完成阶段的结果，生成一份 Markdown 评估报告（`evaluation/results/evaluation_report.md`）。支持增量生成：如人工评估尚未完成，报告中显示占位提示，后续可重新生成以纳入新结果。

---

## 9. Project Directory Structure

```
proj_2026_1/
├── data/
│   ├── raw/                        # Raw Datasets
│   ├── preprocessed/               # Unified Intermediate Format
│   │   ├── unified_all.jsonl       # All humor data
│   │   └── semeval.jsonl           # SemEval data
│   ├── synthesized/                # Synthesized data (Gemini API)
│   ├── sft/                        # SFT Training Data
│   │   ├── sft_train.jsonl
│   │   └── sft_val.jsonl
│   ├── grpo/                       # GRPO Prompt Data
│   │   ├── grpo_prompts_train.jsonl
│   │   └── grpo_prompts_eval.jsonl
│   └── reward/                     # Reward Preference Pairs
│       ├── preference_train.jsonl
│       └── preference_val.jsonl
│
├── data_preprocessing/
│   ├── config.py                   # Centralized configuration for all pipeline stages
│   ├── parsers.py                  # Raw data parsers
│   ├── formatters.py               # SFT / GRPO / Reward data formatters
│   ├── pipeline.py                 # CLI entry point for data pipeline
│   ├── prompt_templates.py         # Prompt template definitions
│   ├── synthesize_task_data.py     # Type B task data synthesis (Gemini)
│   └── synthesize_reward_data.py   # Reward hard-negative synthesis (Gemini)
│
├── sft/
│   └── train_sft.py                # SFT Training Entry
│
├── rl/
│   ├── rewards.py                  # Reward functions + RewardStatsRecorder
│   ├── train_grpo.py               # GRPO training + LoggingGRPOTrainer
│   ├── inference.py                # Inference + Rejection Sampling
│   ├── train_reward_model.py       # Reward model training (Bradley-Terry)
│   ├── reward_model.py             # Reward model inference scorers
│   └── humor_judge.py              # Gemini LLM-as-Judge scorer
│
├── evaluation/
│   ├── pipeline.py                 # Unified evaluation pipeline
│   ├── generate_outputs.py         # Generate model outputs for eval
│   ├── eval_auto_metrics.py        # Automated metrics
│   ├── eval_llm_judge.py           # LLM-as-Judge evaluation
│   ├── export_human_eval.py        # Human evaluation export
│   ├── benchmark_compare.py        # Cross-model comparison
│   └── generate_report.py          # Report generation
│
├── checkpoints/                    # Model Checkpoints
│   ├── sft/
│   ├── grpo/
│   └── reward_model/
│
├── utils/                          # Common Utils
│
├── Dockerfile
├── pyproject.toml
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

1. Decrease `num_generations` (e.g., 16 → 8)
2. Decrease `per_device_train_batch_size` (e.g., 8 → 4)
3. Increase `gradient_accumulation_steps` to compensate
4. Decrease `max_completion_length`
5. Enable gradient checkpointing (`gradient_checkpointing=True`)
6. Use QLoRA (4-bit quantization) instead of LoRA

### B.4 Qwen3 Thinking Mode Handling

Qwen3 thinking mode generates `<think>...</think>` tags in response. For humor generation, **must disable**:

- **Chat template**: Set `enable_thinking=False` in `tokenizer.apply_chat_template()`
- **System prompt**: Add `/no_think` as the system message content

In SFT data and GRPO prompt construction, **uniformly use `/no_think` system prompt** to maintain consistency.
