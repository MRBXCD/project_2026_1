# RL Module Design Document

## Overview

This document describes the design and implementation of the Reinforcement Learning (RL) module for the SemEval 2026 Task A humor generation project. The RL stage uses Group Relative Policy Optimization (GRPO) to improve the SFT-finetuned model's ability to generate funny, constraint-satisfying jokes.

## Module Structure

```
rl/
├── __init__.py              # Exports: build_reward_fn, compute_reward, reward_relevance
├── rewards.py               # Reward function definitions + RewardStatsRecorder
├── train_grpo.py            # GRPO training script + LoggingGRPOTrainer
├── inference.py             # Inference + rejection sampling
├── train_reward_model.py    # Reward model training (Qwen3-1.7B, Bradley-Terry loss)
├── reward_model.py          # Reward model inference scorers
├── humor_judge.py           # Gemini LLM-as-Judge humor scorer
├── sample_reward_model.ipynb # Reward model usage examples
└── RL_DESIGN.md             # This document
```

## Training Pipeline

```
Base Model (Qwen3-8B)
        │
        ▼
┌─────────────────────────────────┐
│     Load Base Model (Qwen3-8B)  │
│     (SFT merge disabled on      │
│      grpo_without_sft branch)   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     Apply New LoRA (r=32)       │
│     for GRPO Training           │
│     Reference = Base model      │
│     Policy = New LoRA weights   │
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│                  GRPO Training Loop                      │
│                                                          │
│   For each batch of prompts:                             │
│   1. Generate G=16 completions per prompt (sampling)     │
│   2. Score all completions with reward function:         │
│      R = W_format * R_format                             │
│        + W_keyword * R_keyword                           │
│        + W_relevance * R_relevance                       │
│        + W_humor * R_humor                               │
│   3. Compute group-relative advantages:                  │
│      A_i = (r_i - mean(group)) / std(group)              │
│   4. Update policy via PPO-clip + KL penalty (beta)      │
└───────────────┬─────────────────────────────────────────┘
                │
                ▼
        GRPO Checkpoint (checkpoints/grpo/final)
                │
                ▼
┌─────────────────────────────────────────────────────────┐
│              Inference with Rejection Sampling            │
│                                                          │
│   1. Generate N=16 candidates (with bounded retry)       │
│   2. Hard filter: keyword inclusion                      │
│   3. Soft ranking: composite reward score                │
│      (includes headline relevance as soft constraint)    │
│   4. Return highest-scoring valid candidate              │
└─────────────────────────────────────────────────────────┘
```

## Reward Function Design (rewards.py)

### Composite Reward Formula

```
total = WEIGHT_FORMAT    * reward_format(response)        [1.0]
      + WEIGHT_KEYWORD   * reward_keyword(response, kws)  [2.0]
      + WEIGHT_RELEVANCE * reward_relevance(headline, response)  [0.5]
      + WEIGHT_HUMOR     * reward_humor(prompt, response, scorer) [1.5]
```

Short-circuit: if `reward_format <= -1.0`, return immediately (degenerate output).

### Sub-reward 1: Format Compliance (`reward_format`)

Checks format quality with additive penalty accumulation:

| Check | Penalty | Condition |
|---|---|---|
| Empty response | -2.0 (early return) | Whitespace-only or empty |
| Too short | -1.0 | Length < 10 characters |
| Too long | -0.5 | Length > 280 characters |
| Repetitive | -1.5 | Trigram uniqueness ratio < 0.5 |
| Base pass | +0.5 | Starting score (penalties subtract from this) |

Compound failures stack: too long + repetitive = 0.5 + (-0.5) + (-1.5) = -1.5.

### Sub-reward 2: Keyword Inclusion (`reward_keyword`)

Case-insensitive substring matching (works across EN/ZH/ES without tokenization):

| Scenario | Score |
|---|---|
| No keywords required (empty list) | 0.0 |
| All N keywords present | N * 1.0 + 0.5 bonus |
| Some keywords present | hits * 1.0 - 0.5 penalty |
| No keywords present | -1.0 |

### Sub-reward 3: Headline Relevance (`reward_relevance`)

Triangular reward curve peaking at ~30% token overlap:

```
reward
 +0.5 │         *          ← target overlap (30%)
      │       /   \
      │     /       \
  0.0 │   /           \
      │ /               \
 -0.5 │*                   *
      └─────────────────────── overlap
      0%      30%         100%
```

Design rationale:
- 0% overlap: response is unrelated to headline → penalize
- ~30% overlap: response references headline topic but adds creative content → reward
- 100% overlap: response is just paraphrasing the headline → penalize

Token extraction is language-agnostic:
- EN/ES: whitespace split → strip punctuation → keep words >= 3 chars
- ZH: extract CJK character bigrams (2-char sliding windows)
- Mixed text: both strategies apply simultaneously

Weight is intentionally low (0.5) because word overlap is a noisy proxy for semantic relevance.

### Sub-reward 4: Humor Quality (`reward_humor`)

Phase 1: always returns 0.0 (no humor scorer provided).

Phase 2 (implemented): Two scorer backends available:
- **Gemini LLM-as-Judge** (`humor_judge.py`): Batch API calls to `gemini-3-flash-preview`, scores 1-5 mapped to [-1, 1] via `(score-1)/4*2-1`. Enabled via `--use_humor_judge`.
- **Trained Reward Model** (`reward_model.py`): On-device forward pass through Qwen3-1.7B reward model, tanh-normalized to [-1, 1]. Enabled via `--use_reward_model`.
- Output clamped to [-1.0, 1.0]
- Exception-safe: returns 0.0 on any scorer failure

### GRPOTrainer Integration (`build_reward_fn`)

The `build_reward_fn(humor_scorer=None, batch_humor_scorer=None, stats_recorder=None)` factory function returns a closure compatible with TRL GRPOTrainer v0.27.1:

```python
reward_fn(prompts, completions, keywords, headline, **kwargs) -> list[float]
```

- `prompts[i]` / `completions[i]`: conversational format `[{"role": "...", "content": "..."}]`
- `keywords[i]` / `headline[i]`: from dataset columns, auto-passed by GRPOTrainer via `**kwargs`
- `stats_recorder`: optional `RewardStatsRecorder` for per-component logging
- Closure captures `humor_scorer` / `batch_humor_scorer` for Phase 2 scoring

## GRPO Training Script (train_grpo.py)

### Model Loading: Base Model + LoRA

1. Load base Qwen3-8B (bf16, flash_attention_2)
2. ~~Load SFT LoRA adapter → `merge_and_unload()`~~ (disabled on `grpo_without_sft` branch; SFT merge code is commented out in `train_grpo.py:186-194`)
3. GRPOTrainer applies new LoRA (r=32) and creates reference model internally

### Key Hyperparameters (A100 80GB)

| Parameter | Value | Rationale |
|---|---|---|
| `num_generations` | 16 | Group size for advantage estimation |
| `per_device_train_batch_size` | 8 | 8 prompts x 16 generations = 128 completions/step |
| `gradient_accumulation_steps` | 2 | Effective batch = 8 x 2 = 16 prompts |
| `num_train_epochs` | 2 | Two passes over the prompt dataset |
| `learning_rate` | 5e-6 | 40x smaller than SFT (2e-4); RL needs cautious updates |
| `beta` | 0.04 | KL penalty; prevents reward hacking |
| `loss_type` | "grpo" | Classic GRPO (not DAPO default) |
| `temperature` | 0.9 | Diversity for meaningful group advantages |
| `max_completion_length` | 256 | Jokes are short |
| `gradient_checkpointing` | False | Fits within A100 80GB without checkpointing |
| `chat_template_kwargs` | `{"enable_thinking": False}` | Disable Qwen3 thinking mode |
| GRPO LoRA rank | 32 | Smaller than SFT (64); fine-grained steering |

### Qwen3 Thinking Mode

Disabled via `chat_template_kwargs={"enable_thinking": False}` in GRPOConfig. Ensures the model generates jokes directly without `<think>...</think>` reasoning blocks.

## Inference Script (inference.py)

### Model Assembly

Three-layer assembly: base → merge SFT → load GRPO adapter.
Optional `--merge_all` flag to merge GRPO adapter too (faster inference).

### Rejection Sampling with Bounded Retry

```
Round 1: generate 16 candidates → check hard constraints
         ├─ valid candidate found → rejection_sample (select best)
         └─ none found ↓
Round 2: generate 16 more (accumulate to 32) → check again
         ├─ valid found → rejection_sample (from all 32)
         └─ none found ↓
Round 3: generate 16 more (accumulate to 48) → check again
         ├─ valid found → rejection_sample (from all 48)
         └─ still none → fallback: pick best-effort candidate
```

Hard constraint (filter): keyword inclusion only.
Soft constraint (ranking): composite reward score (includes headline relevance).

### CLI Usage

```bash
# GRPO training — Phase 1 (rule-based reward only)
python -m rl.train_grpo
python -m rl.train_grpo --beta 0.04 --lr 5e-6 --num_generations 16

# GRPO training — Phase 2 with Gemini LLM-as-Judge
python -m rl.train_grpo --use_humor_judge

# GRPO training — Phase 2 with trained Reward Model
python -m rl.train_grpo --use_reward_model

# Single prompt inference
python -m rl.inference \
    --prompt "Write a joke about tech" \
    --headline "Tech Giants Face AI Regulations" \
    --keywords "penguin,bankruptcy"

# Batch inference
python -m rl.inference \
    --input_file data/grpo/grpo_prompts.jsonl \
    --output_file results.jsonl \
    --n_candidates 16
```

## Phased Training Strategy

| Phase | Reward Components | Activation | Goal |
|---|---|---|---|
| Phase 1 | format + keyword + relevance | Default (no flags) | Learn hard constraints + basic headline grounding |
| Phase 2 | format + keyword + relevance + humor | `--use_humor_judge` or `--use_reward_model` | Improve joke quality via external scorer |

Phase 2 is fully implemented with two scorer backends:

```bash
# Option A: Gemini LLM-as-Judge (requires GEMINI_API_KEY)
python -m rl.train_grpo --use_humor_judge

# Option B: Trained Reward Model (faster, deterministic)
python -m rl.train_grpo --use_reward_model
```

## Reward Model Training (train_reward_model.py)

A lightweight reward model (Qwen3-1.7B + linear score head) trained with Bradley-Terry preference loss on humor preference pair data.

- **Base model**: Qwen3-1.7B (sequence classification)
- **LoRA**: r=16, alpha=32, `modules_to_save=["score"]` (score head must be fully trainable)
- **Loss**: Bradley-Terry: `-log(σ(r_chosen - r_rejected))`
- **Data**: `data/reward/preference_{train,val}.jsonl` (generated by `format_reward_pairs()`)

```bash
python -m rl.train_reward_model
python -m rl.train_reward_model --model_name Qwen/Qwen3-1.7B --lr 1e-5
```

## Humor Judge (humor_judge.py)

Gemini-based humor scoring using `gemini-3-flash-preview`. Supports both single-pair and batch scoring modes.

- **Batch scoring**: Multiple (prompt, response) pairs packed into a single API call for efficiency
- **Score mapping**: Gemini rates 1-5, mapped to [-1, 1] via `(score-1)/4*2-1`
- **Graceful degradation**: Returns 0.0 on API failure

## Reward Monitoring (RewardStatsRecorder + LoggingGRPOTrainer)

`RewardStatsRecorder` (in `rewards.py`) collects per-component reward statistics (format, keyword, relevance, humor) at each training step. `LoggingGRPOTrainer` (in `train_grpo.py`) extends TRL's `GRPOTrainer` to inject these statistics into the trainer's log stream, enabling per-component monitoring via wandb/tensorboard.

## Key Design Decisions

1. **Closure factory pattern** (`build_reward_fn`): Bridges the gap between our plain-text reward functions and TRL's conversational message format, while capturing `humor_scorer` configuration via closure.

2. **Additive format penalty**: `reward_format` accumulates penalties instead of early-returning, so compound failures (too long + repetitive) receive a more negative score than either alone.

3. **Substring keyword matching**: Uses `kw in text` instead of tokenizer-based matching. This is intentionally language-agnostic — it handles Chinese (no word boundaries) and morphological variations (e.g., "penguin" matches "penguins") without external dependencies.

4. **Triangular relevance curve**: Peaks at ~30% headline-response token overlap instead of rewarding maximum overlap. This prevents the model from being rewarded for parroting the headline, encouraging creative development of the topic.

5. **Headline relevance as soft constraint**: Word overlap is too noisy for hard filtering (many false negatives). It influences the ranking score during rejection sampling but does not disqualify candidates.

6. **Bounded retry in inference**: Retries generation up to 3 rounds (accumulating candidates) before falling back to best-effort, balancing constraint satisfaction against compute cost.

## Monitoring (wandb/tensorboard)

Key metrics to watch during GRPO training:

| Metric | Normal | Abnormal |
|---|---|---|
| `reward/mean` | Slowly increasing | Rapid spike → reward hacking |
| `reward/std` | Gradually decreasing | Persistently high → unstable |
| `kl_divergence` | Slowly increasing, moderate | Explosion → reduce lr or increase beta |
| `policy_loss` | Fluctuating, generally decreasing | Flat → lr too small or weak reward signal |
| `completion_length` | Relatively stable | Trending to max → model padding/rambling |
