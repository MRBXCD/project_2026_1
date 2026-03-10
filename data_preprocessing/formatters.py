"""
Layer 3: Formatters
===================

This module is responsible for converting Unified Intermediate Format data into final formats required for various training stages.

Provides 4 formatters, serving different training stages:

1. format_sft_type_a  — General humor data → SFT training format
2. format_sft         — Combines Type A + Type B, splits train/val, outputs complete SFT dataset
3. format_grpo        — SemEval data → GRPO prompt format
4. format_reward_pairs — Unified Intermediate Format → Reward Model preference pairs format

Input/Output relationships of formatters:

    parsers.py output Dataset(s)
           │
           ├──→ format_sft_type_a() ──→ Dataset {"messages": [...]}
           │         │
           │         ├── + Type B JSONL (Synthesized) ──→ format_sft() ──→ DatasetDict {"train", "validation"}
           │
           ├──→ format_grpo() ──→ Dataset {"prompt", "headline", "keywords"}
           │
           └──→ format_reward_pairs() ──→ DatasetDict {"train", "validation"}

Dependencies:
    - datasets (HuggingFace)
    - prompt_templates (Local)
"""

import random
import re
from pathlib import Path

import datasets

from data_preprocessing.prompt_templates import (
    get_random_type_a_prompt,
    build_headline_prompt,
    build_keyword_prompt,
)


# ============================================================
# Reward Pair Allocation Config
# ============================================================
# Per-language allocation for reward model preference pairs.
#   score_based: max number of score-based (joke-vs-joke) pairs to keep.
#                None means use all available.
#   synthesized: max number of synthesized hard-negative pairs to keep.
#                None means use all available.
REWARD_PAIR_ALLOCATION = {
    "en": {"score_based": 7_000, "synthesized": 7_000},
    "es": {"score_based": 7_000, "synthesized": 7_000},
    "zh": {"score_based": 7_000, "synthesized": 7_000},
}

# Fine-grained score-bucket configuration for reward pair construction
REWARD_SCORE_BUCKETS = 4
REWARD_PAIR_TEMPLATES = [(3, 2), (2, 1), (1, 0), (3, 1), (2, 0), (3, 0)]
REWARD_TEMPLATE_RATIOS = {
    (3, 2): 0.20,
    (2, 1): 0.18,
    (1, 0): 0.12,
    (3, 1): 0.16,
    (2, 0): 0.14,
    (3, 0): 0.20,
}
# Deprecated in current reward sampling policy.
# Synthesized data is now controlled only by explicit per-language
# allocation caps in REWARD_PAIR_ALLOCATION[*]["synthesized"].
MAX_SYNTH_RATIO_BY_LANG: dict[str, float] = {}
MAX_REUSE_PER_CHOSEN_BY_LANG = {"en": 4, "es": 5, "zh": 6}
MAX_REUSE_PER_REJECTED_BY_LANG = {"en": 4, "es": 5, "zh": 6}
MAX_PAIR_REUSE_ROUNDS = 3
REUSE_MONITOR_ENABLED = True
REUSE_WARN_RATIO = 0.8
REUSE_STATS_EXPORT_PATH = None


# ============================================================
# Formatter 1: SFT Type A (Unified Intermediate Format → SFT chat format)
# ============================================================

def format_sft_type_a(
    unified_datasets: dict[str, datasets.Dataset],
    score_thresholds: dict[str, float | None] | None = None,
    max_samples_per_source: dict[str, int | None] | None = None,
    exclude_texts: set[str] | None = None,
    seed: int = 42,
) -> datasets.Dataset:
    """Convert Unified Intermediate Format humor data to SFT Type A training format.

    Processing Flow:
        1. Quality filtering for each data source (by score threshold)
        2. Optional downsampling for each data source (to control language balance)
        3. Assign a random Type A prompt to each joke to construct messages format
        4. Merge all data sources

    Args:
        unified_datasets: Dictionary of Unified Intermediate Format datasets output by parser.
            Keys are source names ("rjokes", "cfun", "haha", "chinese_humor"),
            Values are Datasets, schema: {text, lang, score, source}.
            Note: Should not contain "semeval".

        score_thresholds: Minimum score threshold for each source (after normalization).
            Samples below threshold are filtered. None or missing source means no filtering.
            Defaults:
                rjokes: 0.25 (corresponds to raw score >= 5)
                cfun: None (no score, no filtering)
                haha: 0.01 (exclude data with is_humor=0, i.e., score > 0)
                chinese_humor: 0.8 (corresponds to HumorLevel >= 4)

        max_samples_per_source: Maximum number of samples per source (downsampling).
            None or missing source means no limit.
            Defaults:
                cfun: 5000 (164K is too many, downsampling needed to balance languages)
                Others: None (no limit)

        exclude_texts: Optional set of assistant texts to exclude from SFT.
            Matching uses normalized whitespace (strip + collapse inner spaces).

        seed: Random seed for reproducibility of downsampling and prompt selection

    Returns:
        datasets.Dataset: SFT Type A data
            - messages (list[dict]): Chat format dialogue
              [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    # 1. Define default thresholds and downsampling counts
    default_thresholds = {
        "rjokes": 0.25,         # Raw score >= 5
        "cfun": None,           # No score, no filtering
        "haha": 0.01,           # Exclude is_humor=0 (score=0.0)
        "chinese_humor": 0.8,   # HumorLevel >= 4
    }
    default_max_samples = {
        "rjokes": None,
        "cfun": 5000,           # 164K is too many, downsample to balance languages
        "haha": None,
        "chinese_humor": None,
    }
    
    # This is a kind of new parameter handling method. 
    # It is used to merge the default parameters and the custom parameters.
    # useful when we have too many parameters to handle.
    thresholds = {**default_thresholds, **(score_thresholds or {})}
    max_samples = {**default_max_samples, **(max_samples_per_source or {})}
    normalized_exclude_texts = {
        re.sub(r"\s+", " ", text).strip()
        for text in (exclude_texts or set())
        if text is not None
    }

    rng = random.Random(seed)
    filtered_parts = []

    # 2. Filter and downsample each data source
    # Only process unified intermediate format sources, skip semeval
    unified_source_names = [k for k in unified_datasets if k != "semeval"]

    for source_name in unified_source_names:
        ds = unified_datasets[source_name]

        # 2a. Filter by score threshold
        threshold = thresholds.get(source_name)
        if threshold is not None:
            ds = ds.filter(
                lambda x, t=threshold: x["score"] is not None and x["score"] >= t
            )

        # 2b. Downsample
        cap = max_samples.get(source_name)
        if cap is not None and len(ds) > cap:
            ds = ds.shuffle(seed=seed).select(range(cap))

        before_exclude = len(ds)
        if normalized_exclude_texts:
            ds = ds.filter(
                lambda x, blocked=normalized_exclude_texts: re.sub(r"\s+", " ", x["text"]).strip() not in blocked
            )
        after_exclude = len(ds)
        removed = before_exclude - after_exclude
        print(
            f"  [sft_type_a] {source_name}: before={before_exclude} after={after_exclude} removed={removed}"
        )
        filtered_parts.append(ds)

    # 3. Merge all data sources
    merged = datasets.concatenate_datasets(filtered_parts)
    print(f"  [sft_type_a] Total after merge: {len(merged)} rows")

    # 4. Construct messages format for each sample
    def _to_messages(example, idx):
        # Use idx as additional seed to ensure different prompts for each sample
        # while overall reproducibility is still controlled by seed
        sample_rng = random.Random(seed + idx)
        prompt_text = get_random_type_a_prompt(example["lang"], sample_rng)
        return {
            "messages": [
                {"role": "user", "content": prompt_text},
                {"role": "assistant", "content": example["text"]},
            ]
        }

    result = merged.map(
        _to_messages,
        with_indices=True,
        remove_columns=merged.column_names,  # Remove all intermediate format columns
    )

    return result


# ============================================================
# Formatter 2: SFT Complete Dataset (Type A + Type B → train/val)
# ============================================================

def format_sft(
    unified_datasets: dict[str, datasets.Dataset],
    synthesized_dir: str | Path | None = None,
    type_a_ratio: float = 0.7,
    val_ratio: float = 0.1,
    seed: int = 42,
    **type_a_kwargs,
) -> datasets.DatasetDict:
    """Combine Type A and Type B data, split into train/val, output complete SFT dataset.

    Processing Flow:
        1. Call format_sft_type_a() to generate Type A data
        2. Load synthesized Type B JSONL files (if exist)
        3. Control A/B mixing ratio by type_a_ratio
        4. Merge and shuffle
        5. Split into train / validation by val_ratio

    Args:
        unified_datasets: Same as format_sft_type_a parameters

        synthesized_dir: Directory path for synthesized Type B data, expected to contain
            type_b_en.jsonl, type_b_zh.jsonl, type_b_es.jsonl.
            If None or directory not found/empty, only Type A data is used.

        type_a_ratio: Target ratio of Type A data in the final dataset (0.0 ~ 1.0).
            If Type B data is insufficient to reach the target ratio, all available Type B data is used.
            Default 0.7 (70% Type A, 30% Type B).

        val_ratio: Validation set ratio. Default 0.1 (90% train, 10% val).

        seed: Random seed

        **type_a_kwargs: Additional parameters passed to format_sft_type_a
            (score_thresholds, max_samples_per_source, etc.)

    Returns:
        datasets.DatasetDict: Contains "train" and "validation" splits
            Schema for each split: {messages: list[dict]}
    """
    # 1. Generate Type A data
    type_a_ds = format_sft_type_a(
        unified_datasets, seed=seed, **type_a_kwargs
    )
    print(f"  [sft] Type A: {len(type_a_ds)} rows")

    # 2. Load Type B data (if directory exists)
    type_b_ds = None
    if synthesized_dir is not None:
        synthesized_dir = Path(synthesized_dir)
        if synthesized_dir.exists():
            jsonl_files = list(synthesized_dir.glob("type_b_*.jsonl"))
            if jsonl_files:
                type_b_ds = datasets.load_dataset(
                    "json",
                    data_files=[str(f) for f in jsonl_files],
                    split="train",
                )
                print(f"  [sft] Type B: {len(type_b_ds)} rows (loaded from {len(jsonl_files)} files)")
            else:
                print("  [sft] Type B: Directory exists but no type_b_*.jsonl files, skipping")
        else:
            print(f"  [sft] Type B: Directory {synthesized_dir} does not exist, skipping")

    # 3. Mix Type A and Type B by ratio
    if type_b_ds is not None and len(type_b_ds) > 0:
        # Calculate target: type_a takes type_a_ratio, type_b takes 1 - type_a_ratio
        # Calculate how many Type B items based on Type A count
        target_b_count = int(len(type_a_ds) * (1 - type_a_ratio) / type_a_ratio)

        # If Type B data is less than target, use all
        if len(type_b_ds) <= target_b_count:
            actual_b = type_b_ds
        else:
            actual_b = type_b_ds.shuffle(seed=seed).select(range(target_b_count))

        combined = datasets.concatenate_datasets([type_a_ds, actual_b])
        actual_ratio = len(type_a_ds) / len(combined)
        print(f"  [sft] After mixing: {len(combined)} rows (Type A actual ratio: {actual_ratio:.1%})")
    else:
        combined = type_a_ds
        print(f"  [sft] No Type B data, using Type A only: {len(combined)} rows")

    # 4. Shuffle + Split train / validation
    split = combined.shuffle(seed=seed).train_test_split(
        test_size=val_ratio,
        seed=seed,
    )

    result = datasets.DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    print(f"  [sft] Final: train={len(result['train'])}, validation={len(result['validation'])}")
    return result


# ============================================================
# Formatter 3: GRPO Prompt (SemEval Data → GRPO prompt format)
# ============================================================

def format_grpo(
    semeval_dataset: datasets.Dataset,
    eval_ratio: float = 0.0,
    seed: int = 42,
) -> tuple[datasets.Dataset, datasets.Dataset | None]:
    """Convert SemEval parsed results to GRPO prompt format, with optional train/eval split.

    Processing Flow:
        1. Optionally split the dataset into train/eval (stratified by language)
        2. Select template based on subtask type for each SemEval item
        3. Fill template to generate complete user prompt
        4. Wrap in chat messages format (user turn only, no assistant)
        5. Keep headline, keywords, and lang fields

    Args:
        semeval_dataset: Output of parse_semeval().
            schema: {id, headline, keywords, lang, subtask}
        eval_ratio: Fraction of data reserved for evaluation (0.0 = no split).
            Split is stratified by language so each language contributes
            proportionally to both train and eval sets.
        seed: Random seed for reproducible splitting.

    Returns:
        tuple: (train_dataset, eval_dataset)
            eval_dataset is None when eval_ratio == 0.0.
            Each dataset has columns:
            - prompt (list[dict]): [{"role": "user", "content": "..."}]
            - headline (str)
            - keywords (list[str])
            - lang (str)
    """
    def _to_grpo_format(example):
        lang = example["lang"]
        subtask = example["subtask"]

        if subtask == "headline":
            prompt_text = build_headline_prompt(example["headline"], lang)
        else:  # keyword
            kws = example["keywords"]
            prompt_text = build_keyword_prompt(kws[0], kws[1], lang)

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "headline": example["headline"],
            "keywords": example["keywords"],
            "lang": lang,
        }

    if eval_ratio > 0.0:
        rng = random.Random(seed)

        langs = sorted(set(semeval_dataset["lang"]))
        train_indices, eval_indices = [], []

        for lang in langs:
            lang_indices = [
                i for i, l in enumerate(semeval_dataset["lang"]) if l == lang
            ]
            rng.shuffle(lang_indices)
            n_eval = max(1, int(len(lang_indices) * eval_ratio))
            eval_indices.extend(lang_indices[:n_eval])
            train_indices.extend(lang_indices[n_eval:])

        train_ds = semeval_dataset.select(train_indices).map(
            _to_grpo_format, remove_columns=semeval_dataset.column_names,
        )
        eval_ds = semeval_dataset.select(eval_indices).map(
            _to_grpo_format, remove_columns=semeval_dataset.column_names,
        )

        for split_name, ds in [("train", train_ds), ("eval", eval_ds)]:
            per_lang = {}
            for lang_val in ds["lang"]:
                per_lang[lang_val] = per_lang.get(lang_val, 0) + 1
            dist_str = ", ".join(f"{k}: {v}" for k, v in sorted(per_lang.items()))
            print(f"  [grpo] {split_name}: {len(ds)} prompts ({dist_str})")

        return train_ds, eval_ds

    result = semeval_dataset.map(
        _to_grpo_format, remove_columns=semeval_dataset.column_names,
    )
    print(f"  [grpo] Generated {len(result)} GRPO prompts (no eval split)")
    return result, None


# ============================================================
# Formatter 4: Reward Model Preference Pairs
# ============================================================

def _weighted_pick_index(
    candidate_indices: list[int],
    remaining_quota: list[int],
    rng: random.Random,
) -> int | None:
    """Pick one index using remaining quota as weights."""
    if not candidate_indices:
        return None
    total_weight = 0
    for idx in candidate_indices:
        total_weight += max(remaining_quota[idx], 0)
    if total_weight <= 0:
        return rng.choice(candidate_indices)

    threshold = rng.uniform(0, total_weight)
    cumulative = 0.0
    for idx in candidate_indices:
        cumulative += max(remaining_quota[idx], 0)
        if cumulative >= threshold:
            return idx
    return candidate_indices[-1]


def _percentile_nearest_rank(values: list[int], percentile: float) -> int:
    """Return nearest-rank percentile from non-empty integer values."""
    if not values:
        return 0
    sorted_values = sorted(values)
    n = len(sorted_values)
    rank = max(1, int(round(percentile * n)))
    rank = min(rank, n)
    return sorted_values[rank - 1]


def _summarize_reuse_counts(counts: dict[str, int]) -> dict[str, float | int]:
    """Compute reuse summary stats for logging."""
    values = list(counts.values())
    if not values:
        return {
            "n_items": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    return {
        "n_items": len(values),
        "mean": sum(values) / len(values),
        "p50": _percentile_nearest_rank(values, 0.50),
        "p90": _percentile_nearest_rank(values, 0.90),
        "p95": _percentile_nearest_rank(values, 0.95),
        "p99": _percentile_nearest_rank(values, 0.99),
        "max": max(values),
    }


def _make_bucket_indices(scores: list[float], bucket_count: int, rng: random.Random) -> dict[int, list[int]]:
    """Split score indices into quantile-like buckets with random tie-breaking."""
    n = len(scores)
    indices = list(range(n))
    rng.shuffle(indices)
    indices.sort(key=lambda i: scores[i])
    buckets: dict[int, list[int]] = {}
    for bucket_idx in range(bucket_count):
        start = int(n * bucket_idx / bucket_count)
        end = int(n * (bucket_idx + 1) / bucket_count)
        buckets[bucket_idx] = indices[start:end]
    return buckets


def _allocate_target_by_ratios(
    total_target: int,
    templates: list[tuple[int, int]],
    ratios: dict[tuple[int, int], float],
) -> dict[tuple[int, int], int]:
    """Allocate integer template targets by ratios with largest remainder."""
    if total_target <= 0:
        return {tpl: 0 for tpl in templates}
    raw = {tpl: total_target * ratios.get(tpl, 0.0) for tpl in templates}
    base = {tpl: int(raw[tpl]) for tpl in templates}
    remainder = total_target - sum(base.values())
    order = sorted(
        templates,
        key=lambda tpl: (raw[tpl] - base[tpl], ratios.get(tpl, 0.0)),
        reverse=True,
    )
    for tpl in order[:remainder]:
        base[tpl] += 1
    return base


def _build_pairs_with_limits(
    chosen_texts: list[str],
    rejected_texts: list[str],
    target_pairs: int,
    max_reuse_per_chosen: int,
    max_reuse_per_rejected: int,
    max_attempts: int,
    rng: random.Random,
) -> tuple[list[tuple[str, str]], dict[str, int], dict[str, int], int]:
    """Build unique chosen/rejected pairs under bidirectional reuse limits."""
    if target_pairs <= 0 or not chosen_texts or not rejected_texts:
        return [], {}, {}, 0

    chosen_count = [0] * len(chosen_texts)
    rejected_count = [0] * len(rejected_texts)
    chosen_used: dict[str, int] = {}
    rejected_used: dict[str, int] = {}
    used_edges: set[tuple[int, int]] = set()
    pairs: list[tuple[str, str]] = []

    attempts = 0
    while len(pairs) < target_pairs and attempts < max_attempts:
        attempts += 1
        chosen_candidates = [
            i for i, used in enumerate(chosen_count) if used < max_reuse_per_chosen
        ]
        rejected_candidates = [
            i for i, used in enumerate(rejected_count) if used < max_reuse_per_rejected
        ]
        if not chosen_candidates or not rejected_candidates:
            break

        chosen_remaining = [max_reuse_per_chosen - used for used in chosen_count]
        rejected_remaining = [max_reuse_per_rejected - used for used in rejected_count]

        i = _weighted_pick_index(chosen_candidates, chosen_remaining, rng)
        j = _weighted_pick_index(rejected_candidates, rejected_remaining, rng)
        if i is None or j is None:
            break
        if (i, j) in used_edges:
            continue

        used_edges.add((i, j))
        chosen_count[i] += 1
        rejected_count[j] += 1
        c_text = chosen_texts[i]
        r_text = rejected_texts[j]
        chosen_used[c_text] = chosen_used.get(c_text, 0) + 1
        rejected_used[r_text] = rejected_used.get(r_text, 0) + 1
        pairs.append((c_text, r_text))

    return pairs, chosen_used, rejected_used, attempts

def format_reward_pairs(
    unified_datasets: dict[str, datasets.Dataset],
    high_quantile: float = 0.7,
    low_quantile: float = 0.3,
    max_pairs_per_chosen: int = 3,
    allocation: dict[str, dict[str, int | None]] | None = None,
    synthesized_reward_dir: str | Path | None = None,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Construct Reward Model preference pairs from Unified Intermediate Format data.

    Processing Flow:
        1. Generate score-based pairs per source (quantile split with tie-breaking)
        2. Load synthesized hard-negative pairs
        3. Apply per-language allocation caps (score_based and synthesized independently)
        4. Merge all languages, shuffle, split train/val

    Args:
        unified_datasets: Dictionary of Unified Intermediate Format datasets output by parser.
            Only uses sources with scores (rjokes, haha, chinese_humor),
            Automatically skips sources with score=None (cfun) and non-unified format (semeval).

        high_quantile: Minimum percentile for high score group. Default 0.7 (Top 30% are chosen).
        low_quantile: Maximum percentile for low score group. Default 0.3 (Bottom 30% are rejected).
        max_pairs_per_chosen: Max number of pairs per chosen sample.
            Prevents reusing the same good joke too many times. Default 3.
        allocation: Per-language allocation config.  Each language maps to
            {"score_based": int|None, "synthesized": int|None} where None means
            "use all available".  Defaults to REWARD_PAIR_ALLOCATION module constant.
        synthesized_reward_dir: Directory containing synthesized hard-negative preference
            pair files (reward_neg_en.jsonl, reward_neg_zh.jsonl, reward_neg_es.jsonl).
            If set and directory exists, these files are loaded and merged into the
            preference pairs. None means no synthesized data is loaded.
        val_ratio: Validation set ratio. Default 0.1.
        seed: Random seed.

    Returns:
        datasets.DatasetDict: Contains "train" and "validation" splits
            Schema for each split:
            - prompt (list[dict]):   [{"role": "user", "content": "..."}]
            - chosen (list[dict]):   [{"role": "assistant", "content": "High score joke"}]
            - rejected (list[dict]): [{"role": "assistant", "content": "Low score joke"}]
    """
    if allocation is None:
        allocation = REWARD_PAIR_ALLOCATION
    rng = random.Random(seed)
    import json

    scored_sources = ["rjokes", "haha", "chinese_humor"]
    source_lang = {"rjokes": "en", "haha": "es", "chinese_humor": "zh"}
    active_sources = [s for s in scored_sources if s in unified_datasets]
    lang_source_counts = {"en": 0, "es": 0, "zh": 0}
    for source_name in active_sources:
        lang_source_counts[source_lang[source_name]] += 1

    # Allocate language score budget to source budget (future-safe if multiple sources per language)
    source_budget: dict[str, int | None] = {}
    for lang in ["en", "es", "zh"]:
        sources = [s for s in active_sources if source_lang[s] == lang]
        cap = allocation.get(lang, {}).get("score_based")
        if cap is None or not sources:
            for s in sources:
                source_budget[s] = None
            continue
        base = cap // len(sources)
        remainder = cap % len(sources)
        for idx, s in enumerate(sorted(sources)):
            source_budget[s] = base + (1 if idx < remainder else 0)

    # Collect pairs grouped by language for per-language capping
    lang_pairs: dict[str, dict[str, list]] = {
        "en": {"prompt": [], "chosen": [], "rejected": []},
        "es": {"prompt": [], "chosen": [], "rejected": []},
        "zh": {"prompt": [], "chosen": [], "rejected": []},
    }
    reuse_monitor_records: list[dict] = []

    template_ratios = REWARD_TEMPLATE_RATIOS.copy()
    ratio_sum = sum(template_ratios.values()) or 1.0
    template_ratios = {k: v / ratio_sum for k, v in template_ratios.items()}

    for source_name in active_sources:
        ds = unified_datasets[source_name].filter(lambda x: x["score"] is not None)
        if len(ds) == 0:
            continue

        lang = source_lang[source_name]
        scores = ds["score"]
        texts = ds["text"]
        bucket_indices = _make_bucket_indices(scores, REWARD_SCORE_BUCKETS, rng)
        bucket_texts = {
            bucket: [texts[i] for i in indices]
            for bucket, indices in bucket_indices.items()
        }

        print(
            f"  [reward] {source_name} ({lang}) buckets: "
            + ", ".join(f"Q{b}={len(bucket_texts[b])}" for b in sorted(bucket_texts))
        )

        config_reuse_chosen = MAX_REUSE_PER_CHOSEN_BY_LANG.get(lang, max_pairs_per_chosen)
        config_reuse_rejected = MAX_REUSE_PER_REJECTED_BY_LANG.get(lang, max_pairs_per_chosen)
        # Keep backward compatibility: caller-provided max_pairs_per_chosen limits both sides.
        max_reuse_per_chosen = min(config_reuse_chosen, max_pairs_per_chosen)
        max_reuse_per_rejected = min(config_reuse_rejected, max_pairs_per_chosen)

        # Estimate template capacities under current reuse constraints.
        template_capacity: dict[tuple[int, int], int] = {}
        for tpl in REWARD_PAIR_TEMPLATES:
            hi, lo = tpl
            chosen_size = len(bucket_texts.get(hi, []))
            rejected_size = len(bucket_texts.get(lo, []))
            cap = min(
                chosen_size * max_reuse_per_chosen,
                rejected_size * max_reuse_per_rejected,
                chosen_size * rejected_size,
            )
            template_capacity[tpl] = max(cap, 0)

        budget = source_budget.get(source_name)
        if budget is None:
            template_target = template_capacity.copy()
        else:
            template_target = _allocate_target_by_ratios(budget, REWARD_PAIR_TEMPLATES, template_ratios)
            for tpl in template_target:
                template_target[tpl] = min(template_target[tpl], template_capacity[tpl])

        template_pairs: dict[tuple[int, int], list[tuple[str, str]]] = {
            tpl: [] for tpl in REWARD_PAIR_TEMPLATES
        }
        template_pair_sets: dict[tuple[int, int], set[tuple[str, str]]] = {
            tpl: set() for tpl in REWARD_PAIR_TEMPLATES
        }
        template_deficit: dict[tuple[int, int], int] = {tpl: 0 for tpl in REWARD_PAIR_TEMPLATES}

        for tpl in REWARD_PAIR_TEMPLATES:
            hi, lo = tpl
            chosen_pool = bucket_texts.get(hi, [])
            rejected_pool = bucket_texts.get(lo, [])
            target_count = template_target.get(tpl, 0)
            if target_count <= 0 or not chosen_pool or not rejected_pool:
                template_deficit[tpl] = target_count
                continue

            # Compact very large pools to keep pair sampling tractable.
            chosen_needed = max(1, target_count // max(max_reuse_per_chosen, 1) + 2)
            rejected_needed = max(1, target_count // max(max_reuse_per_rejected, 1) + 2)
            chosen_limit = min(len(chosen_pool), chosen_needed * 3)
            rejected_limit = min(len(rejected_pool), rejected_needed * 3)
            if len(chosen_pool) > chosen_limit:
                chosen_pool = rng.sample(chosen_pool, chosen_limit)
            if len(rejected_pool) > rejected_limit:
                rejected_pool = rng.sample(rejected_pool, rejected_limit)

            sampled_pairs, _, _, _ = _build_pairs_with_limits(
                chosen_pool,
                rejected_pool,
                target_pairs=target_count,
                max_reuse_per_chosen=max_reuse_per_chosen,
                max_reuse_per_rejected=max_reuse_per_rejected,
                max_attempts=max(target_count * 20 * MAX_PAIR_REUSE_ROUNDS, 200),
                rng=rng,
            )
            for pair in sampled_pairs:
                if pair not in template_pair_sets[tpl]:
                    template_pair_sets[tpl].add(pair)
                    template_pairs[tpl].append(pair)

            template_deficit[tpl] = max(0, target_count - len(template_pairs[tpl]))

            if REUSE_MONITOR_ENABLED:
                chosen_counts: dict[str, int] = {}
                rejected_counts: dict[str, int] = {}
                for c_text, r_text in template_pairs[tpl]:
                    chosen_counts[c_text] = chosen_counts.get(c_text, 0) + 1
                    rejected_counts[r_text] = rejected_counts.get(r_text, 0) + 1
                chosen_stats = _summarize_reuse_counts(chosen_counts)
                rejected_stats = _summarize_reuse_counts(rejected_counts)
                reuse_monitor_records.append(
                    {
                        "lang": lang,
                        "source": source_name,
                        "template": f"Q{hi}>Q{lo}",
                        "chosen": chosen_stats,
                        "rejected": rejected_stats,
                    }
                )
                print(
                    f"  [reward][{lang}][Q{hi}>Q{lo}] target={target_count}, "
                    f"actual={len(template_pairs[tpl])}, deficit={template_deficit[tpl]}"
                )
                print(
                    f"    chosen_reuse p95={chosen_stats['p95']} p99={chosen_stats['p99']} "
                    f"max={chosen_stats['max']}"
                )
                print(
                    f"    rejected_reuse p95={rejected_stats['p95']} p99={rejected_stats['p99']} "
                    f"max={rejected_stats['max']}"
                )
                if chosen_stats["p99"] > max_reuse_per_chosen:
                    print("    [ERROR] chosen reuse p99 exceeds configured cap")
                elif chosen_stats["p95"] > max_reuse_per_chosen * REUSE_WARN_RATIO:
                    print("    [WARN] chosen reuse p95 near cap")
                if rejected_stats["p99"] > max_reuse_per_rejected:
                    print("    [ERROR] rejected reuse p99 exceeds configured cap")
                elif rejected_stats["p95"] > max_reuse_per_rejected * REUSE_WARN_RATIO:
                    print("    [WARN] rejected reuse p95 near cap")

        # Phase 3: in-language redistribution after resampling
        if budget is not None:
            generated = sum(len(template_pairs[tpl]) for tpl in REWARD_PAIR_TEMPLATES)
            shortage = max(0, budget - generated)
            if shortage > 0:
                extra_targets = _allocate_target_by_ratios(shortage, REWARD_PAIR_TEMPLATES, template_ratios)
                for tpl in REWARD_PAIR_TEMPLATES:
                    hi, lo = tpl
                    chosen_pool = bucket_texts.get(hi, [])
                    rejected_pool = bucket_texts.get(lo, [])
                    extra_target = extra_targets.get(tpl, 0)
                    if extra_target <= 0 or not chosen_pool or not rejected_pool:
                        continue
                    chosen_needed = max(1, extra_target // max(max_reuse_per_chosen, 1) + 2)
                    rejected_needed = max(1, extra_target // max(max_reuse_per_rejected, 1) + 2)
                    chosen_limit = min(len(chosen_pool), chosen_needed * 3)
                    rejected_limit = min(len(rejected_pool), rejected_needed * 3)
                    if len(chosen_pool) > chosen_limit:
                        chosen_pool = rng.sample(chosen_pool, chosen_limit)
                    if len(rejected_pool) > rejected_limit:
                        rejected_pool = rng.sample(rejected_pool, rejected_limit)
                    sampled_pairs, _, _, _ = _build_pairs_with_limits(
                        chosen_pool,
                        rejected_pool,
                        target_pairs=extra_target,
                        max_reuse_per_chosen=max_reuse_per_chosen,
                        max_reuse_per_rejected=max_reuse_per_rejected,
                        max_attempts=max(extra_target * 20, 200),
                        rng=rng,
                    )
                    before = len(template_pairs[tpl])
                    for pair in sampled_pairs:
                        if pair not in template_pair_sets[tpl]:
                            template_pair_sets[tpl].add(pair)
                            template_pairs[tpl].append(pair)
                    added = len(template_pairs[tpl]) - before
                    if added > 0:
                        print(f"  [reward][{lang}][Q{hi}>Q{lo}] redistribution added {added} pairs")

        # Flatten template pairs into language pairs
        source_generated = 0
        for tpl in REWARD_PAIR_TEMPLATES:
            for chosen_text, rejected_text in template_pairs[tpl]:
                prompt_text = get_random_type_a_prompt(lang, rng)
                lang_pairs[lang]["prompt"].append([{"role": "user", "content": prompt_text}])
                lang_pairs[lang]["chosen"].append([{"role": "assistant", "content": chosen_text}])
                lang_pairs[lang]["rejected"].append([{"role": "assistant", "content": rejected_text}])
                source_generated += 1
        print(f"  [reward] {source_name} ({lang}): generated {source_generated} score-based pairs")

    # Apply per-language score_based cap from allocation config
    for lang, pairs in lang_pairs.items():
        n_raw = len(pairs["prompt"])
        cap = allocation.get(lang, {}).get("score_based")
        if cap is not None and n_raw > cap:
            indices = list(range(n_raw))
            rng.shuffle(indices)
            selected = sorted(indices[:cap])
            for key in ("prompt", "chosen", "rejected"):
                pairs[key] = [pairs[key][i] for i in selected]
            print(f"  [reward] Downsampled {lang} score-based: {n_raw} -> {cap}")

    # Optionally export reuse stats
    if REUSE_MONITOR_ENABLED and REUSE_STATS_EXPORT_PATH:
        export_path = Path(REUSE_STATS_EXPORT_PATH)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with open(export_path, "w", encoding="utf-8") as f:
            for record in reuse_monitor_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [reward] Reuse stats exported: {export_path}")

    # Load synthesized hard-negative pairs, apply per-language synthesized cap
    synth_counts: dict[str, int] = {"en": 0, "zh": 0, "es": 0}
    synth_pairs: dict[str, dict[str, list]] = {
        lang: {"prompt": [], "chosen": [], "rejected": []}
        for lang in ["en", "zh", "es"]
    }

    if synthesized_reward_dir is not None:
        synth_dir = Path(synthesized_reward_dir)
        if synth_dir.exists():
            for lang in ["en", "zh", "es"]:
                synth_file = synth_dir / f"reward_neg_{lang}.jsonl"
                if not synth_file.exists():
                    continue
                records = []
                with open(synth_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        records.append(json.loads(line))

                synth_cap_cfg = allocation.get(lang, {}).get("synthesized")
                final_cap = synth_cap_cfg if synth_cap_cfg is not None else len(records)
                final_cap = max(0, final_cap)

                if len(records) > final_cap:
                    rng.shuffle(records)
                    records = records[:final_cap]

                for record in records:
                    synth_pairs[lang]["prompt"].append(record["prompt"])
                    synth_pairs[lang]["chosen"].append(record["chosen"])
                    synth_pairs[lang]["rejected"].append(record["rejected"])

                synth_counts[lang] = len(records)
                print(
                    f"  [reward] Loaded {len(records)} synthesized pairs for {lang} "
                    f"(cfg_cap={synth_cap_cfg}) from {synth_file.name}"
                )
        else:
            print(f"  [reward] Synthesized dir {synth_dir} does not exist, skipping")

    # Merge score-based + synthesized, print composition summary
    all_pairs: dict[str, list] = {"prompt": [], "chosen": [], "rejected": []}

    print()
    print(f"  {'lang':<6} {'score_based':>12} {'synthesized':>12} {'total':>8}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*8}")

    for lang in ["en", "zh", "es"]:
        n_score = len(lang_pairs[lang]["prompt"])
        n_synth = synth_counts[lang]

        for key in ("prompt", "chosen", "rejected"):
            all_pairs[key].extend(lang_pairs[lang][key])
            all_pairs[key].extend(synth_pairs[lang][key])

        total = n_score + n_synth
        print(f"  {lang:<6} {n_score:>12,} {n_synth:>12,} {total:>8,}")

    total_all = len(all_pairs["prompt"])
    print(f"  {'total':<6} {'':<12} {'':<12} {total_all:>8,}")

    if not all_pairs["prompt"]:
        raise ValueError("Failed to generate any preference pairs, please check data and parameters")

    pairs_ds = datasets.Dataset.from_dict(all_pairs)
    split = pairs_ds.train_test_split(test_size=val_ratio, seed=seed)
    result = datasets.DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })
    print(f"  [reward] Final: train={len(result['train'])}, validation={len(result['validation'])}")
    return result
