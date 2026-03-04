# Data Processing Module Design Document

## Table of Contents

- [1. Overview](#1-overview)
- [2. Data Source Inventory](#2-data-source-inventory)
  - [2.1 Data Source Format Details](#21-data-source-format-details)
  - [2.2 SemEval Task A Subtask Structure](#22-semeval-task-a-subtask-structure)
- [3. Processing Architecture](#3-processing-architecture)
  - [3.1 Three-Layer Architecture Overview](#31-three-layer-architecture-overview)
  - [3.2 Layer 1: Source Parsers](#32-layer-1-source-parsers)
  - [3.3 Layer 2: Unified Intermediate Format](#33-layer-2-unified-intermediate-format)
  - [3.4 Layer 3: Formatters](#34-layer-3-formatters)
- [4. SFT Data Design](#4-sft-data-design)
  - [4.1 Type A: General Humor Data](#41-type-a-general-humor-data)
  - [4.2 Type B: Task Formatted Data (Synthesized)](#42-type-b-task-formatted-data-synthesized)
  - [4.3 Type A and Type B Mixing Strategy](#43-type-a-and-type-b-mixing-strategy)
  - [4.4 Qwen3 Thinking Mode Handling](#44-qwen3-thinking-mode-handling)
- [5. GRPO Data Design](#5-grpo-data-design)
  - [5.1 Subtask 1: Headline-Only](#51-subtask-1-headline-only)
  - [5.2 Subtask 2: Keywords-Only](#52-subtask-2-keywords-only)
- [6. Reward Model Preference Pair Data Design](#6-reward-model-preference-pair-data-design)
  - [6.1 Purpose and Training Objective](#61-purpose-and-training-objective)
  - [6.2 Preference Pair Construction Strategy](#62-preference-pair-construction-strategy)
  - [6.3 Data Format](#63-data-format)
  - [6.4 Construction Flow and Sampling Details](#64-construction-flow-and-sampling-details)
  - [6.5 Language Rebalancing Strategy](#65-language-rebalancing-strategy)
  - [6.6 Hard-Negative Synthesis Pipeline](#66-hard-negative-synthesis-pipeline)
  - [6.7 Notes](#67-notes)
- [7. Prompt Template Design](#7-prompt-template-design)
  - [7.1 Type A General Humor Prompt Pool](#71-type-a-general-humor-prompt-pool)
  - [7.2 Type B Task Formatted Prompt Templates](#72-type-b-task-formatted-prompt-templates)
  - [7.3 GRPO Prompt Templates](#73-grpo-prompt-templates)
- [8. Quality Filtering Strategy](#8-quality-filtering-strategy)
- [9. File Organization and Output](#9-file-organization-and-output)
- [10. Pipeline Invocation](#10-pipeline-invocation)

---

## 1. Overview

This module is responsible for processing multiple raw data sources into standard format data required for SFT and GRPO training stages.
Design Principles:

1. **Core on HuggingFace `datasets` library** — Use existing APIs like `load_dataset`, `map`, `filter`, `concatenate_datasets`, avoid reinventing the wheel.
2. **Pipeline Processing** — Raw Data → Unified Intermediate Format → Training Format, each step independently adjustable.
3. **Different Formatters for SFT and GRPO** — Different data structures for each.

---

## 2. Data Source Inventory

### 2.1 Data Source Format Details

| Dataset | Language | Usage | Raw Format | Key Fields | Volume | Path |
|---|---|---|---|---|---|---|
| **rJokes** | EN | SFT (Type A) | TSV.gz | `score` (int), `joke` (str) | ~43K (dev) + train + test | `data/rjoke/` |
| **CFun** | ZH | SFT (Type A) | HF Arrow | `instruction`, `input`, `output` | 164K | `data/cfun/` |
| **HAHA 2019** | ES | SFT (Type A) | CSV | `text`, `is_humor` (0/1), `funniness_average` (float) | ~36K | `data/haha/` |
| **Chinese Humor Multi-Labeled** | ZH | SFT (Type A) + Pref Pairs | TSV (tab-separated) | `Content` (str), `HumorLevel` (1-5) | ~3.3K | `data/Chinese_Humor_Multi-Labeled/` |
| **SemEval Task A** | EN/ZH/ES | GRPO prompts | TSV | `headline`, `word1`, `word2` | Each ~300 (275 headline + 25 keyword) | `data/semeval_task/` |
| **Synthesized Type B Data** | EN/ZH/ES | SFT (Type B) | JSONL (Stored after synthesis) | `messages` | On demand | `data/synthesized/` |

#### rJokes Field Description

```
score (int)  |  joke (str)
─────────────┼──────────────────────────────────
1            |  "I'll have a cheeseburger..."
0            |  "Who is Michael J. Fox's..."
3            |  "A guy calls in sick to work..."
```

- score is Reddit community vote score, higher means more popular
- Distribution concentrated around 0-3, long tail extends to higher scores

#### CFun Field Description

```
instruction (str)  |  input (str)  |  output (str)
───────────────────┼───────────────┼──────────────────────
"请讲一个笑话"     |  ""           |  "有一天小明..."
```

- Already in instruction fine-tuning format, but its native instruction doesn't match this task
- **Only use `output` field** as joke text, re-prompting required

#### HAHA 2019 Field Description

```
id  |  text (str)              |  is_humor (0/1)  |  funniness_average (float)
────┼──────────────────────────┼──────────────────┼───────────────────────────
... |  "Niveles de retraso..." |  1               |  1.5
```

- `is_humor=1` indicates labeled as humorous text
- `funniness_average` is average rating 1-5 (meaningful only when is_humor=1)
- SFT stage only uses `is_humor=1` data
- Keep `is_humor=0` data for subsequent preference pair construction

#### Chinese Humor Multi-Labeled Field Description

```
ID (str)  |  Title (str)  |  Content (str)         |  HumorLevel (1-5)
──────────┼───────────────┼────────────────────────┼──────────────────
L0001     |  要求加薪      |  員工：老闆，你必須...  |  4
L0004     |  職業習慣      |  一天，一位法官的...    |  2
```

- `HumorLevel` 1-5 rating
- Text is **Traditional Chinese**
- SFT stage uses high-quality jokes with `HumorLevel >= 4`
- Full data (including low scores) kept for preference pair construction

### 2.2 SemEval Task A Subtask Structure

SemEval Task A contains **two mutually exclusive subtasks**:

| Subtask | Input | Constraint | Data Range (EN Example) |
|---|---|---|---|
| **Headline-based** | News Headline | Generate headline-related joke | en_2001 ~ en_2275 (275 items) |
| **Keyword-based** | Two Keywords | Joke must contain these two words | en_2276 ~ en_2300 (25 items) |

Data Characteristics:
- Headline-based items: `word1 = "-"`, `word2 = "-"`, `headline` has value
- Keyword-based items: `headline = "-"`, `word1` and `word2` have values
- **Two subtasks are mutually exclusive**, no items have both headline and keywords

---

## 3. Processing Architecture

### 3.1 Three-Layer Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Data Processing Architecture                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Layer 1: Source Parsers                                                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  rJokes   │ │   CFun   │ │   HAHA   │ │ Chinese  │ │ SemEval  │          │
│  │  Parser   │ │  Parser  │ │  Parser  │ │ Humor    │ │  Parser  │          │
│  │          │ │          │ │          │ │  Parser  │ │          │          │
│  └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘ └─────┬────┘          │
│        │            │            │             │            │                │
│        ▼            ▼            ▼             ▼            │                │
│  Layer 2: Unified Intermediate Format                        │                │
│  ┌──────────────────────────────────────────────┐          │                │
│  │ { "text", "lang", "score", "source" }        │          │                │
│  └──────────────────┬───────────────────────────┘          │                │
│                     │                                       │                │
│  Layer 3: Formatters│                                       │                │
│        ┌────────────┼────────────┬──────────┐  ┌───────────┘                │
│        ▼            ▼            ▼          │  ▼                            │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐               │
│  │ SFT      │ │ SFT      │ │ Reward   │ │ GRPO Prompt      │               │
│  │ Type A   │ │ Type B   │ │ Pair     │ │ Formatter        │               │
│  │ Formatter│ │ Formatter│ │ Formatter│ │ (Direct from     │               │
│  │          │ │          │ │          │ │  SemEval)        │               │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬──────────┘               │
│       │            │            │                │                          │
│       │            │            │  ┌─────────────────────────────────┐      │
│       │            │            │  │ Reward Hard-Negative Synthesis  │      │
│       │            │            │  │ (synthesize_reward_data.py)     │      │
│       │            │            │  │                                 │      │
│       │            │            │  │ pure_jokes.csv─→ Chosen (ZH)   │      │
│       │            │            │  │ unified_all  ──→ Chosen (EN/ES)│      │
│       │            │            │  │ Gemini API   ──→ Rejected (all)│      │
│       │            │            │  └────────┬────────────────────────┘      │
│       │            │            │           │                               │
│       │            │            ▼           ▼                               │
│       │            │     ┌──────────────────────────┐                      │
│       │            │     │ format_reward_pairs()     │                      │
│       │            │     │  1. Score-based pairs     │                      │
│       │            │     │  2. Per-lang downsample   │                      │
│       │            │     │  3. Merge synthesized     │                      │
│       │            │     │  4. Shuffle + split       │                      │
│       │            │     └───────────┬──────────────┘                      │
│       │            │                 │                                      │
│       ▼            ▼                 ▼                ▼                     │
│  ┌──────────────────────────────────────────────────────────┐              │
│  │ Output: HuggingFace Dataset (JSONL)                       │              │
│  │ • data/sft/sft_train.jsonl, sft_val.jsonl                 │              │
│  │ • data/reward/preference_train.jsonl, preference_val.jsonl│              │
│  │ • data/grpo/grpo_prompts.jsonl                            │              │
│  └──────────────────────────────────────────────────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

> Note: SemEval data does **not pass through** Unified Intermediate Format, but is directly output to GRPO Formatter by SemEval Parser, because its structure and usage are completely different from SFT data.
> Note: The Reward Pair Formatter has two data sources: (1) score-based preference pairs from the Unified Intermediate Format, and (2) synthesized hard-negative pairs from `synthesize_reward_data.py`. Both are merged inside `format_reward_pairs()` during the `format_reward` pipeline stage.

### 3.2 Layer 1: Source Parsers

Each parser is an independent function taking raw file path input and outputting `datasets.Dataset` object.

| Parser Function | Input File | Processing Logic |
|---|---|---|
| `parse_rjokes(path)` | `data/rjoke/*.tsv.gz` | Read score + joke; filter empty; normalize score |
| `parse_cfun(cache_dir)` | `data/cfun/` | Load from HF cache; take `output` field only |
| `parse_haha(path)` | `data/haha/*.csv` | Read all cols; filter `is_humor=1`; normalize funniness_average |
| `parse_chinese_humor(path)` | `data/Chinese_Humor_Multi-Labeled/mlabel_corpora/JokeHumorLevel.txt` | Read Content + HumorLevel; normalize |
| `parse_semeval(path)` | `data/semeval_task/task-a-*.tsv` | Distinguish headline-only and keyword-only subtasks |

### 3.3 Layer 2: Unified Intermediate Format

All humor data (rJokes, CFun, HAHA, Chinese Humor) unified into the following schema:

```python
{
    "text": str,              # Joke/Humor text body
    "lang": "en"|"zh"|"es",   # Language identifier
    "score": float | None,    # Quality score normalized to [0, 1], None if no score
    "source": str,            # Data source identifier ("rjokes" / "cfun" / "haha" / "chinese_humor")
}
```

**Score Normalization Scheme:**

| Data Source | Original Score | Normalization Method | Effective Range | Description |
|---|---|---|---|---|
| rJokes | int (0 ~ 11) | `min(score, 11) / 11` | [0, 1.0] | Cap matches actual data max (11 upvotes) |
| CFun | None | `None` | — | No score info |
| HAHA 2019 | float (1.0 ~ 5.0) | `funniness_average / 5.0` | [0, 0.95] | `/5` is the annotation scale max; 0.95 means no joke averaged a perfect 5.0 |
| Chinese Humor | int (1 ~ 5) | `HumorLevel / 5.0` | [0.2, 1.0] | Direct linear mapping |

### 3.4 Layer 3: Formatters

#### SFT Type A Formatter

Convert unified intermediate format joke data to SFT training chat format:

```python
# Input: Unified Intermediate Format
{"text": "...", "lang": "en", "score": 0.75, "source": "rjokes"}

# Output: SFT chat format
{
    "messages": [
        {"role": "user", "content": "<Randomly selected from corresponding language prompt pool>"},
        {"role": "assistant", "content": "<text field>"}
    ]
}
```

#### SFT Type B Formatter

Directly load synthesized task formatted data (already stored as JSONL), no extra conversion needed.

#### GRPO Formatter

Convert SemEval data to GRPO training prompt format:

```python
# Input: SemEval Parsed Result
{"id": "en_2001", "headline": "...", "word1": "-", "word2": "-"}

# Output (headline-only):
{
    "prompt": [
        {"role": "user", "content": "<Generated from headline prompt template>"}
    ],
    "headline": "Original headline text",
    "keywords": []
}

# Output (keyword-only):
{
    "prompt": [
        {"role": "user", "content": "<Generated from keyword prompt template>"}
    ],
    "headline": "",
    "keywords": ["word1", "word2"]
}
```

---

## 4. SFT Data Design

### 4.1 Type A: General Humor Data

**Goal**: Teach model humorous language style.

**Data Sources and Quality Filtering**:

| Data Source | Language | Filter Condition | Est. Count after Filter |
|---|---|---|---|
| rJokes | EN | `score >= 5` | ~5K-8K (TBD) |
| CFun | ZH | None (Use all, random sample to control volume) | Sample ~5K |
| HAHA 2019 | ES | `is_humor == 1` | ~10K |
| Chinese Humor | ZH | `HumorLevel >= 4` | ~1K |

> CFun has 164K items, using all would make Chinese data far outweigh En/Es, so **downsampling** is needed to balance languages.

**Final Format**:

```json
{
    "messages": [
        {"role": "user", "content": "Tell me a short joke."},
        {"role": "assistant", "content": "I told my wife she was drawing her eyebrows too high. She looked surprised."}
    ]
}
```

### 4.2 Type B: Task Formatted Data (Synthesized)

**Goal**: Teach model to understand "Headline → Joke" and "Keywords → Joke" input-output mappings.

**Synthesis Flow** (Done by independent script `synthesize_task_data.py`):

1. Extract headlines from news headline dataset (Recommend Babel Briefings)
2. Randomly pair two low-frequency words from vocabulary as keywords
3. Call strong model API (e.g., Gemini) to generate humorous responses satisfying constraints
4. Quality filtering (Check keyword inclusion, reasonable length, etc.)
5. Store as JSONL

**Note:** We do not mix SemEval data here to generate task-specialized data. This prevents data leakage and ensures GRPO exploration space isn't compressed (since SFT would form fixed answers for these headlines).

**Storage Location**: `data/synthesized/type_b_en.jsonl`, `type_b_zh.jsonl`, `type_b_es.jsonl`

**Final Format (Headline Subtask)**:

```json
{
    "messages": [
        {"role": "user", "content": "You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.\n\nHeadline: \"Tech Giants Face New Regulations on AI Safety\"\n\nWrite a humorous one-liner inspired by the headline."},
        {"role": "assistant", "content": "The new AI safety regulations are so strict, even Siri is hiring a lawyer."}
    ]
}
```

**Final Format (Keyword Subtask)**:

```json
{
    "messages": [
        {"role": "user", "content": "You are a witty comedian. Write a short, funny one-liner joke that naturally includes both of the following words: 'hammer' and 'flower'.\n\nWrite a humorous one-liner that contains both required words."},
        {"role": "assistant", "content": "I tried to fix my garden with a hammer, but all I got was a flat flower and a noise complaint."}
    ]
}
```

### 4.3 Type A and Type B Mixing Strategy

| Data Type | Ratio | Description |
|---|---|---|
| Type A (General Humor) | ~60-70% | Establish humor language style foundation |
| Type B (Task Formatted) | ~30-40% | Teach model to understand task input-output mappings |

Mixed then shuffled, split 90/10 into train/val.

### 4.4 Qwen3 Thinking Mode Handling

Qwen3 series models (including Qwen3-8B) **enable thinking mode by default**, generating internal reasoning in `<think>...</think>` tags before response.

For humor generation, thinking mode is unnecessary and harmful (wastes tokens, interferes with reward calculation), needs to be disabled.

**Handling**: **Do not add any special tokens in data**. Thinking mode is uniformly disabled in training script via `tokenizer.apply_chat_template(..., enable_thinking=False)`. Data level remains clean.

---

## 5. GRPO Data Design

GRPO stage "training data" is **prompt collection** (no response), model generates multiple responses itself then scored by reward function.

### 5.1 Subtask 1: Headline-Only

```json
{
    "prompt": [
        {"role": "user", "content": "You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.\n\nHeadline: \"Panamanian lawmakers' Taiwan trip sparks diplomatic row with China\"\n\nWrite a humorous one-liner inspired by the headline."}
    ],
    "headline": "Panamanian lawmakers' Taiwan trip sparks diplomatic row with China",
    "keywords": []
}
```

- `headline` and `keywords` fields **not passed to model**, only used by reward function during calculation
- Empty `keywords` list means no keyword constraint

### 5.2 Subtask 2: Keywords-Only

```json
{
    "prompt": [
        {"role": "user", "content": "You are a witty comedian. Write a short, funny one-liner joke that naturally includes both of the following words: 'hammer' and 'flower'.\n\nWrite a humorous one-liner that contains both required words."}
    ],
    "headline": "",
    "keywords": ["hammer", "flower"]
}
```

- Empty `headline` string means no headline constraint
- `keywords` contains two required words, reward function will check generated text for these words

---

## 6. Reward Model Preference Pair Data Design

### 6.1 Purpose and Training Objective

In GRPO training, we need a reward function to score each response generated by the model. "Humor level" scoring has two implementation methods:

| Method | Pros | Cons |
|---|---|---|
| **External LLM-as-Judge** (API Call) | No extra training needed | High API cost, slow |
| **Train a small Reward Model** | Fast inference, no API cost | Requires preference pair data + extra training step |

If choosing to train a reward model, we need to construct preference pair data. Training objective: given a text, output a scalar score such that "funnier" text gets higher score than "not funny" text.

### 6.2 Preference Pair Construction Strategy

We utilize **existing scored data** to construct preference pairs. Core idea: For the same prompt, select chosen from high-score samples, rejected from low-score samples.

**Available Data Sources:**

| Source | Language | Score Field | Chosen Condition | Rejected Condition | Discard Middle |
|---|---|---|---|---|---|
| rJokes | EN | score (int, 0–11) | Top 30% (by rank, tie-breaking random) | Bottom 30% (by rank, tie-breaking random) | Middle 40% |
| HAHA 2019 | ES | funniness_average + is_humor | Top 30% (by rank, tie-breaking random) | Bottom 30% (by rank, tie-breaking random) | Middle 40% |
| Chinese Humor | ZH | HumorLevel (1-5) | Top 30% (by rank, tie-breaking random) | Bottom 30% (by rank, tie-breaking random) | Middle 40% |

> **About Spanish**: HAHA 2019 `is_humor=0` samples (score=0) naturally fall into the bottom 30% and serve as rejected source (annotators deemed these "not humorous").
>
> **About tie-breaking**: All three datasets have highly discrete scores (rJokes: 12 unique values, Chinese Humor: 5 unique values, HAHA: 61% at score=0). A naive `>= q70` / `<= q30` split would include entire tie groups, violating the 30/40/30 split (e.g., rJokes would select 75% instead of 60%). The implementation uses rank-based selection with random tie-breaking: indices are shuffled, stable-sorted by score, then the bottom/top 30% by index position are selected. This guarantees exactly 30% in each group regardless of score discreteness.

### 6.3 Data Format

Preference pair data uses TRL `RewardTrainer` compatible format:

```json
{
    "prompt": [
        {"role": "user", "content": "Tell me a joke."}
    ],
    "chosen": [
        {"role": "assistant", "content": "High score joke text"}
    ],
    "rejected": [
        {"role": "assistant", "content": "Low score joke text"}
    ]
}
```

### 6.4 Construction Flow and Sampling Details

```
Raw Scored Data (per source)
       │
       ▼
┌──────────────────────────────────┐
│ Step 1: Score-based pairing       │
│ (Within each source independently)│
│  • Sort by score (random tie-     │
│    breaking for discrete scores)  │
│  • Take exactly:                  │
│    high = top 30% by rank         │
│    low  = bottom 30% by rank      │
│    mid  = middle 40% ← Discard   │
│  • Randomly pair high (chosen)    │
│    with low (rejected)            │
│  • Each chosen max 3 times        │
│  • Prompt from Type A pool        │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 2: Apply score_based cap     │
│  • Per REWARD_PAIR_ALLOCATION     │
│    (en:7K, es:7K, zh:None=all)   │
│  • Downsample if over cap         │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 3: Load synthesized pairs    │
│  • Load reward_neg_{lang}.jsonl   │
│    from data/synthesized/         │
│  • Apply synthesized cap per      │
│    REWARD_PAIR_ALLOCATION         │
│    (en:7K, es:7K, zh:13K)        │
│  • Downsample if over cap         │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│ Step 4: Merge, shuffle, split     │
│  • Combine score-based + synth    │
│  • Print composition summary      │
│    (score_based/synth/total       │
│     per language)                 │
│  • Shuffle                        │
│  • Split train / validation (9:1) │
└──────────────────────────────────┘
```

**Raw Preference Pair Count (before rebalancing):**

| Language | Source | High Est. | Low Est. | Raw Pairs |
|---|---|---|---|---|
| EN | rJokes (~43K dev) | ~6K-8K | ~6K-8K | ~6K-8K pairs |
| ZH | Chinese Humor (~3.3K) | ~1K | ~800 | ~800 pairs |
| ES | HAHA 2019 (~36K, inc is_humor=0) | ~3K-5K | ~15K+ | ~3K-5K pairs |

> The severe Chinese underrepresentation (~800 pairs vs ~50K+ EN/ES combined) caused language bias
> in the first reward model iteration — see Section 6.5 for the rebalancing strategy.

### 6.5 Language Rebalancing Strategy

**Problem**: The initial reward model exhibited language bias due to data imbalance. Chinese/English text received inflated humor scores regardless of actual quality, because the model learned a shortcut: "Chinese text → usually chosen" / "English text → usually seen in training".

**Root cause**: ~151K total preference pairs, of which only ~1.1K were Chinese (~0.7%). The model learned spurious language-based correlations instead of genuine humor features.

**Solution — Two-pronged rebalancing**:

| Strategy | Target | Mechanism | Implementation |
|---|---|---|---|
| **Score-based cap** | EN, ES | Limit score-based (joke-vs-joke) pairs per language | `REWARD_PAIR_ALLOCATION[lang]["score_based"]` |
| **Hard-negative synthesis** | EN, ZH, ES | Generate obviously non-humorous rejected texts + pair with high-quality chosen | `synthesize_reward_data.py` → `reward_neg_{lang}.jsonl` |
| **Synthesized cap** | EN, ZH, ES | Limit hard-negative pairs per language | `REWARD_PAIR_ALLOCATION[lang]["synthesized"]` |

**Rebalanced Target Distribution — configured in `REWARD_PAIR_ALLOCATION` (`formatters.py`)**:

| Language | Score-based Pairs | Synthesized Hard-Negative Pairs | Total | Hard-neg % |
|---|---|---|---|---|
| EN | 7,000 | 7,000 | 14,000 | 50% |
| ZH | ~1,230 (all available) | 13,000 | ~14,230 | ~91% |
| ES | 7,000 | 7,000 | 14,000 | 50% |

> **Why fixed allocation instead of a single `max_pairs_per_lang` cap?**
> Score-based pairs (joke vs. less-funny joke) and hard-negative pairs (joke vs. boring text) teach
> the model fundamentally different things. A single cap applied after merging cannot control the
> composition ratio. The `REWARD_PAIR_ALLOCATION` config controls each type independently, ensuring:
> 1. EN/ES hard-negative coverage is raised to 50% (vs. 5.3%/40.7% in the pre-fix dataset).
> 2. ZH uses all available score-based pairs (~1,230, limited by dataset size) plus synthesized pairs.
> 3. Total per-language count stays balanced (~14K each).
>
> **Pre-fix composition (for reference):**
>
> | Language | Total Pairs | Hard-Negative % | Score-based % |
> |---|---|---|---|
> | ZH | ~12,845 | ~92.5% | ~7.5% |
> | ES | ~12,746 | ~40.7% | ~59.3% |
> | EN | ~12,632 | ~5.3% | ~94.7% |
>
> The EN imbalance (only 5.3% hard-negative) was the root cause of the reward model assigning inflated
> scores to boring English text. The fixed `REWARD_PAIR_ALLOCATION` resolves this.

### 6.6 Hard-Negative Synthesis Pipeline

Implemented in `synthesize_reward_data.py`. Generates preference pairs where **chosen** = real high-quality joke, **rejected** = synthetically generated boring text.

**Data Flow:**

```
┌───────────────────────────────────────────────────┐
│ Chosen Sources (Language-specific)                 │
│  • ZH: Pre-extracted CFun jokes from               │
│         data/cfun/pure_jokes.csv                   │
│         (rule-filtered: dedup + length 10-500)     │
│  • EN: rJokes score_normalized >= 5/11 from         │
│         unified_all.jsonl                          │
│         (= raw Reddit upvotes >= 5, top ~8%)       │
│  • ES: HAHA score_normalized >= 0.30 from          │
│         unified_all.jsonl                          │
│         (= funniness_average >= 1.5/5)             │
└────────────────────┬──────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│ Rejected Source (Gemini API — Batch Generation)    │
│  • Prompt requests BATCH_SIZE=100 statements per   │
│    API call using structured JSON output           │
│    (response_mime_type="application/json").        │
│  • Prompts explicitly enforce target language      │
│    (e.g., "IMPORTANT: Every statement MUST be      │
│    written in English.").                          │
│  • Language filter: each returned statement is     │
│    verified with langid.classify() to match the    │
│    expected language; mismatches are discarded.    │
│  • Additional filters: length, refusal patterns.   │
└────────────────────┬──────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│ Assembly                                           │
│  Pair chosen[i] with rejected[i], assign random    │
│  Type A prompt in corresponding language.           │
│  Output: data/synthesized/reward_neg_{lang}.jsonl  │
└───────────────────────────────────────────────────┘
```

**Score Threshold Rationale (per language):**

| Language | Source | Threshold | Rationale |
|---|---|---|---|
| EN | rJokes | `score_normalized >= 5/11 ≈ 0.4545` (raw >= 5) | rJokes scores normalized as `raw / 11`; raw >= 5 selects ~8% (~34K candidates) |
| ES | HAHA | `score_normalized >= 0.30` (funniness >= 1.5/5) | Inclusive threshold retains genuinely humorous tweets; 0.7 was too strict (only 241 jokes passed) |
| ZH | CFun (pure_jokes.csv) | No score filter | Quality ensured by manual curation of the CSV; rule-based length/dedup applied |

> **Threshold history**: The initial threshold for EN/ES was mistakenly set to `0.7`, which is calibrated
> for HAHA's 1–5 scale (where 0.7 = 3.5/5, very strict). For rJokes with the old cap of 20, this
> required raw score >= 14 upvotes — resulting in **zero** qualifying jokes. The per-language
> `_SCORE_THRESHOLDS` dict in `synthesize_reward_data.py` was introduced to encode correct per-source
> defaults. After changing rJokes cap from 20 to 11, the EN threshold was updated from 0.25 to 5/11.

**CFun CSV Rule-based Filtering**:

Chinese chosen samples are loaded from `data/cfun/pure_jokes.csv`, a pre-extracted CSV containing ~37K standalone jokes from the CFun dataset. The following rule-based filters are applied automatically by `load_cfun_jokes()`:

| Filter | Description | Estimated Removal |
|---|---|---|
| Deduplication | Remove exact duplicate rows | ~3.2K rows |
| Min length | Remove texts < 10 characters (punctuation fragments) | ~800 rows |
| Max length | Remove texts > 500 characters (overly long narratives) | ~180 rows |
| Whitespace strip | Trim leading/trailing whitespace | — |

After filtering, ~30K valid jokes remain. The function randomly samples `n_samples` from this pool (default 9K for ZH hard-negative synthesis).

### 6.7 Notes

**1. Prompt Consistency Issue**

Standard RLHF preference pairs require chosen and rejected to be **different responses to the exact same prompt**. But our data are independently collected jokes, not responses to the same prompt.

This is feasible in practice — reward model essentially learns the classification/ranking task "what kind of text is funnier". But note:
- Use same prompt when pairing (select same one from prompt pool)
- This means reward model learns more about **text's inherent humor**, rather than **response quality to a specific prompt**

**2. Reward Model Architecture**

Using Qwen3-1.7B + LoRA + `AutoModelForSequenceClassification` (adds a scalar `score` head). LoRA config includes `modules_to_save=["score"]` to ensure the new classification head is trainable. Raw logits are normalized via `tanh` to `[-1.0, 1.0]` during inference.

Training uses TRL `RewardTrainer` with Bradley-Terry loss. See `rl/train_reward_model.py` for configuration details.

**3. Hard-Negative Design Rationale**

Hard negatives (boring, non-humorous text) serve a different purpose from score-based rejection pairs:
- **Score-based rejected**: Jokes that are less funny — model learns fine-grained humor ranking
- **Hard negatives**: Plainly non-humorous text — model learns the fundamental boundary between "humor" and "not humor"

This is especially important for preventing the reward model from assigning high scores to non-humorous text that superficially resembles humor (e.g., short Chinese sentences).

**4. Language Detection for Synthesized Rejected Texts**

Gemini API occasionally produces responses in the wrong language despite explicit instructions (e.g., generating English statements when asked for Chinese). A `langid`-based post-filter (`_check_language_match` in `synthesize_reward_data.py`) discards any returned statement whose detected language does not match the requested language. `langid` is configured via `langid.set_languages(["en", "zh", "es"])` before use.

Note: `langid` returns log-probability confidence scores (negative values). The magnitude scales with text length; very negative values (e.g., -200) for long texts are normal and do not indicate low confidence. Only the detected language label is used for filtering — the confidence value is not thresholded.

**5. Batch Generation**

The rejected text generation uses batched Gemini API calls (BATCH_SIZE=100 statements per call) with structured JSON output (`response_mime_type="application/json"`, `response_json_schema`). This reduces API overhead ~100× compared to single-item calls and is the recommended approach for generating large numbers of samples.

---

## 7. Prompt Template Design

### 7.1 Type A General Humor Prompt Pool

Used for SFT Type A data user-side prompt. Randomly **select one** from corresponding language pool during training sample construction.

**English Prompt Pool:**

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

**Chinese Prompt Pool:**

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

**Spanish Prompt Pool:**

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

### 7.2 Type B Task Formatted Prompt Templates

Used for SFT Type B data and synthesis script. Distinguished by subtask and language.

#### Headline Subtask Templates

**English:**

```
You are a witty comedian. Given the following news headline, write a short, funny one-liner joke inspired by it.

Headline: "{headline}"

Write a humorous one-liner inspired by the headline.
```

**Chinese:**

```
你是一位机智的喜剧演员。根据以下新闻标题，写一个简短有趣的笑话。

新闻标题：「{headline}」

写一句幽默的段子。
```

**Spanish:**

```
Eres un comediante ingenioso. Dado el siguiente titular de noticias, escribe un chiste corto y divertido inspirado en él.

Titular: "{headline}"

Escribe un chiste divertido de una línea inspirado en el titular.
```

#### Keyword Subtask Templates

**English:**

```
You are a witty comedian. Write a short, funny one-liner joke that naturally includes both of the following words: '{word1}' and '{word2}'.

Write a humorous one-liner that contains both required words.
```

**Chinese:**

```
你是一位机智的喜剧演员。写一个简短有趣的笑话，其中必须自然地包含以下两个词：「{word1}」和「{word2}」。

写一句包含以上两个词语的幽默段子。
```

**Spanish:**

```
Eres un comediante ingenioso. Escribe un chiste corto y divertido que incluya naturalmente las siguientes dos palabras: '{word1}' y '{word2}'.

Escribe un chiste divertido de una línea que contenga ambas palabras.
```

### 7.3 GRPO Prompt Templates

GRPO stage uses same template structure as Type B, but data source is SemEval official data. Templates reused from Section 6.2.

---

## 8. Quality Filtering Strategy

### SFT Stage Filtering

| Data Source | Filter Condition | Description |
|---|---|---|
| rJokes | `score >= 5` | Pick high score jokes (approx top 20-30%), threshold adjustable later |
| CFun | No filter + Downsample | 164K all usable, but need downsample to ~5K for balance |
| HAHA 2019 | `is_humor == 1` | Exclude non-humorous text |
| Chinese Humor | `HumorLevel >= 4` | Pick relatively high score jokes |

### General Text Quality Filtering

Applied uniformly to all sources:

1. **Non-empty Check** — Filter empty text or whitespace-only samples
2. **Min Length** — Text length >= 10 chars
3. **Max Length** — Text length <= 2000 chars (Too long text might be noise)
4. **Deduplication** — Deduplicate based on exact text match

### Preference Pair Reserve

The following data kept intact (including low score/non-humorous), not used in SFT, but available for preference pairs:

- rJokes: Full data (including low scores)
- HAHA 2019: `is_humor=0` data
- Chinese Humor: `HumorLevel <= 2` data

---

## 9. File Organization and Output

### Code File Organization

```
proj_2026_1/
├── data_preprocessing/
│   ├── DATA_PIPELINE_DESIGN.md        # This design document
│   ├── parsers.py                     # Layer 1: Parser functions for each source
│   ├── prompt_templates.py            # Multi-lang prompt pool + task templates
│   ├── formatters.py                  # Layer 3: SFT / GRPO / Preference Pair Formatters
│   ├── pipeline.py                    # End-to-end pipeline (Parse → Filter → Format → Save)
│   ├── synthesize_task_data.py        # Type B SFT Data Synthesis Script (Requires Gemini API)
│   ├── synthesize_reward_data.py      # Reward Hard-Negative Synthesis Script (Requires Gemini API)
│   └── visulization.ipynb             # Data Visualization & Inspection (langid-based lang detection)
│
├── tests/
│   ├── test_formatters.py             # Tests for format_reward_pairs() rebalancing logic
│   └── test_synthesize_reward_data.py # Tests for synthesize_reward_data.py pure functions
│
├── rl/
│   ├── train_reward_model.py          # Reward model training (Qwen3-1.7B + LoRA + RewardTrainer)
│   ├── reward_model.py                # Reward model inference (scoring functions)
│   ├── sample_reward_model.ipynb      # Interactive reward model case study / debugging
│   └── ...
```

### Data Output Structure

```
proj_2026_1/
├── data/
│   ├── raw/                        # Raw Data (User organized)
│   │   ├── rjoke/
│   │   ├── cfun/
│   │   ├── haha/
│   │   ├── Chinese_Humor_Multi-Labeled/
│   │   └── semeval_task/
│   │
│   ├── preprocessed/               # Unified Intermediate Format (Contains full scores, reusable)
│   │   ├── unified_en.jsonl
│   │   ├── unified_zh.jsonl
│   │   └── unified_es.jsonl
│   │
│   ├── synthesized/                # Synthesized Data
│   │   ├── type_b_en.jsonl         #   SFT Type B (Gemini-generated task data)
│   │   ├── type_b_zh.jsonl
│   │   ├── type_b_es.jsonl
│   │   ├── reward_neg_en.jsonl     #   Reward hard-negative pairs (Gemini-generated)
│   │   ├── reward_neg_zh.jsonl
│   │   └── reward_neg_es.jsonl
│   │
│   ├── sft/                        # Final SFT Training Data
│   │   ├── sft_train.jsonl
│   │   └── sft_val.jsonl
│   │
│   ├── reward/                     # Reward Model Preference Pair Data
│   │   ├── preference_train.jsonl
│   │   └── preference_val.jsonl
│   │
│   └── grpo/                       # Final GRPO Training Data
│       └── grpo_prompts.jsonl
```

### Output File Format Description

**`preprocessed/unified_*.jsonl`** — Unified Intermediate Format, one JSON per line:

```json
{"text": "...", "lang": "en", "score": 0.75, "source": "rjokes"}
```

**`sft/sft_train.jsonl`** — SFT Training Data (Type A + Type B mixed & shuffled), one JSON per line:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**`reward/preference_train.jsonl`** — Reward Model Preference Pair Data, one JSON per line:

```json
{"prompt": [{"role": "user", "content": "..."}], "chosen": [{"role": "assistant", "content": "High score joke"}], "rejected": [{"role": "assistant", "content": "Low score joke"}]}
```

**`grpo/grpo_prompts.jsonl`** — GRPO prompt data, one JSON per line:

```json
{"prompt": [{"role": "user", "content": "..."}], "headline": "...", "keywords": [...]}
```

---

## 10. Pipeline Invocation

All commands executed from project root, using `python -m` module mode.

### 10.1 One-Click Completion (Recommended)

```bash
# Method 1: Full pipeline (Inc. Type B synthesis, requires Gemini API)
#   Order: parse → synthesize → format_sft → format_grpo → format_reward
export GEMINI_API_KEY='your-api-key'
python -m data_preprocessing.pipeline --stage full

# Specify synthesis count:
python -m data_preprocessing.pipeline --stage full --n_headline 300 --n_keyword 150

# Method 2: Pipeline without synthesis (No API needed, if Type B ready)
#   Order: parse → format_sft → format_grpo → format_reward
python -m data_preprocessing.pipeline --stage all
```

### 10.2 Step-by-Step Execution (For Debugging)

```bash
# Step 1: Parse Raw Data → Unified Intermediate Format
#   Input: data/raw/*
#   Output: data/preprocessed/unified_all.jsonl, semeval.jsonl
python -m data_preprocessing.pipeline --stage parse

# Step 2: Synthesize Type B SFT Data (Requires Gemini API)
#   Input: Babel Briefings (Auto download) + Built-in keyword vocab
#   Output: data/synthesized/type_b_{en,zh,es}.jsonl
export GEMINI_API_KEY='your-api-key'
python -m data_preprocessing.pipeline --stage synthesize
python -m data_preprocessing.pipeline --stage synthesize --n_headline 300 --n_keyword 150

# Or use independent script to synthesize specific language:
python -m data_preprocessing.synthesize_task_data --lang en --n_headline 200 --n_keyword 100

# Step 2b: Synthesize Reward Hard-Negative Data (Requires Gemini API)
#   Input: pure_jokes.csv (ZH) + unified_all.jsonl (EN/ES) + Gemini API
#   Output: data/synthesized/reward_neg_{en,zh,es}.jsonl
#
#   Recommended n_samples (targeting ~7K hard-negative pairs per language after filtering):
#     ZH: 9000  (CFun pool is large, high throughput)
#     EN: 7000  (rJokes pool is large; extra headroom for langid filtering)
#     ES: 7000  (HAHA pool is smaller; extra headroom needed)
#
#   IMPORTANT: Must use `uv run` to pick up langid + google-genai from the uv venv.
export GEMINI_API_KEY='your-api-key'
uv run python -m data_preprocessing.synthesize_reward_data --lang zh --n_samples 9000
uv run python -m data_preprocessing.synthesize_reward_data --lang en --n_samples 7000
uv run python -m data_preprocessing.synthesize_reward_data --lang es --n_samples 7000

# Step 3: Unified Intermediate Format → SFT Training Data
#   Input: data/preprocessed/unified_all.jsonl + data/synthesized/type_b_*.jsonl (Optional)
#   Output: data/sft/sft_train.jsonl, sft_val.jsonl
python -m data_preprocessing.pipeline --stage format_sft

# Step 4: SemEval → GRPO Prompt Data
#   Input: data/preprocessed/semeval.jsonl
#   Output: data/grpo/grpo_prompts.jsonl
python -m data_preprocessing.pipeline --stage format_grpo

# Step 5: Unified Intermediate Format → Reward Model Preference Pairs
#   Input: data/preprocessed/unified_all.jsonl + data/synthesized/reward_neg_*.jsonl (Optional)
#   Output: data/reward/preference_train.jsonl, preference_val.jsonl
#
#   Allocation (score_based / synthesized caps) is configured in
#   REWARD_PAIR_ALLOCATION in data_preprocessing/formatters.py.
#   No CLI flags needed — edit the constant directly to change allocation.
python -m data_preprocessing.pipeline --stage format_reward
```

### 10.3 Stage Comparison

| stage | Inc Parse | Inc SFT Synth | Inc Format | Need API | Scenario |
|---|---|---|---|---|---|
| `full` | Yes | Yes | Yes | Yes | One-click start to finish (exc. reward synthesis) |
| `all` | Yes | No | Yes | No | Type B ready, or not needed yet |
| Individual stage | - | - | - | synthesize only | Debug, partial regeneration |

> Note: Reward hard-negative synthesis (`synthesize_reward_data.py`) is always run as a standalone script, not included in `full` or `all`, because it requires Gemini API and generates a large number of API calls.

### 10.4 Recommended Execution Order

**First Run (full pipeline including reward rebalancing)**:

```bash
# 1. Parse raw data
python -m data_preprocessing.pipeline --stage parse

# 2. (Optional) Synthesize SFT Type B data
export GEMINI_API_KEY='your-api-key'
python -m data_preprocessing.pipeline --stage synthesize

# 3. Synthesize reward hard-negatives (requires data/cfun/pure_jokes.csv for ZH)
#    Use uv run to pick up langid + google-genai from the uv venv.
uv run python -m data_preprocessing.synthesize_reward_data --lang zh --n_samples 9000
uv run python -m data_preprocessing.synthesize_reward_data --lang en --n_samples 7000
uv run python -m data_preprocessing.synthesize_reward_data --lang es --n_samples 7000

# 4. Format all training data (with language rebalancing)
python -m data_preprocessing.pipeline --stage format_sft
python -m data_preprocessing.pipeline --stage format_grpo
python -m data_preprocessing.pipeline --stage format_reward
```

**Subsequent Adjustment** (e.g., modified filter threshold, no re-parse/re-synthesis needed):
1. `--stage format_sft` Re-generate SFT data
2. Edit `REWARD_PAIR_ALLOCATION` in `formatters.py`, then `--stage format_reward`
3. Other stages as needed

**Running Tests:**

```bash
python -m pytest tests/test_formatters.py tests/test_synthesize_reward_data.py -v
```
