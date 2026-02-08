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
from pathlib import Path

import datasets

from data_preprocessing.prompt_templates import (
    get_random_type_a_prompt,
    build_headline_prompt,
    build_keyword_prompt,
)


# ============================================================
# Formatter 1: SFT Type A (Unified Intermediate Format → SFT chat format)
# ============================================================

def format_sft_type_a(
    unified_datasets: dict[str, datasets.Dataset],
    score_thresholds: dict[str, float | None] | None = None,
    max_samples_per_source: dict[str, int | None] | None = None,
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

        print(f"  [sft_type_a] {source_name}: {len(ds)} rows (after filtering)")
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

def format_grpo(semeval_dataset: datasets.Dataset) -> datasets.Dataset:
    """Convert SemEval parsed results to GRPO training prompt format.

    Processing Flow:
        1. Select template based on subtask type for each SemEval item
        2. Fill template to generate complete user prompt
        3. Wrap in chat messages format (user turn only, no assistant)
        4. Keep headline and keywords fields (for reward function usage)

    Args:
        semeval_dataset: Output of parse_semeval().
            schema: {id, headline, keywords, lang, subtask}

    Returns:
        datasets.Dataset: GRPO prompt format
            - prompt (list[dict]): [{"role": "user", "content": "..."}]
            - headline (str): News headline, has value for headline subtask, else empty string
            - keywords (list[str]): List of keywords, has value for keyword subtask, else empty list
    """
    def _to_grpo_format(example):
        lang = example["lang"]
        subtask = example["subtask"]

        # Select template based on subtask type and fill
        if subtask == "headline":
            prompt_text = build_headline_prompt(example["headline"], lang)
        else:  # keyword
            kws = example["keywords"]
            prompt_text = build_keyword_prompt(kws[0], kws[1], lang)

        return {
            "prompt": [{"role": "user", "content": prompt_text}],
            "headline": example["headline"],
            "keywords": example["keywords"],
        }

    result = semeval_dataset.map(
        _to_grpo_format,
        remove_columns=semeval_dataset.column_names,  # Remove id, lang, subtask, etc.
    )

    print(f"  [grpo] Generated {len(result)} GRPO prompts")
    return result


# ============================================================
# Formatter 4: Reward Model Preference Pairs
# ============================================================

def format_reward_pairs(
    unified_datasets: dict[str, datasets.Dataset],
    high_quantile: float = 0.7,
    low_quantile: float = 0.3,
    max_pairs_per_chosen: int = 3,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> datasets.DatasetDict:
    """Construct Reward Model preference pairs from Unified Intermediate Format data.

    Processing Flow:
        1. Filter sources with scores (exclude CFun with score=None)
        2. Group by language
        3. Within each language, split into high / low groups by score (discard middle range)
        4. Randomly pair high (chosen) and low (rejected)
        5. Assign the same Type A prompt to each pair
        6. Merge all languages, shuffle, split train/val

    Args:
        unified_datasets: Dictionary of Unified Intermediate Format datasets output by parser.
            Only uses sources with scores (rjokes, haha, chinese_humor),
            Automatically skips sources with score=None (cfun) and non-unified format (semeval).

        high_quantile: Minimum percentile for high score group. Default 0.7 (Top 30% are chosen).
        low_quantile: Maximum percentile for low score group. Default 0.3 (Bottom 30% are rejected).
        max_pairs_per_chosen: Max number of pairs per chosen sample.
            Prevents reusing the same good joke too many times. Default 3.
        val_ratio: Validation set ratio. Default 0.1.
        seed: Random seed.

    Returns:
        datasets.DatasetDict: Contains "train" and "validation" splits
            Schema for each split:
            - prompt (list[dict]):   [{"role": "user", "content": "..."}]
            - chosen (list[dict]):   [{"role": "assistant", "content": "High score joke"}]
            - rejected (list[dict]): [{"role": "assistant", "content": "Low score joke"}]
    """
    rng = random.Random(seed)

    # Scored sources (skip cfun and semeval)
    scored_sources = ["rjokes", "haha", "chinese_humor"]

    all_pairs = {"prompt": [], "chosen": [], "rejected": []}

    # 1. Calculate quantiles and construct preference pairs independently for each source
    #    Different sources have different score distributions and normalizations
    #    (rJokes uses Reddit score, HAHA uses funniness_average, Chinese Humor uses HumorLevel),
    #    so quantiles must be calculated within each source, cannot be calculated after merging.
    for source_name in scored_sources:
        if source_name not in unified_datasets:
            continue

        ds = unified_datasets[source_name]

        # Filter out rows with score=None (defensive check)
        ds = ds.filter(lambda x: x["score"] is not None)

        if len(ds) == 0:
            continue

        lang = ds[0]["lang"]  # Each source corresponds to a single language
        scores = ds["score"]

        # 1a. Calculate quantile thresholds within the source
        sorted_scores = sorted(scores)
        n = len(sorted_scores)
        high_threshold = sorted_scores[int(n * high_quantile)]
        low_threshold = sorted_scores[int(n * low_quantile)]

        # 1b. Split into high / low groups
        high_texts = [ds[i]["text"] for i, s in enumerate(scores) if s >= high_threshold]
        low_texts = [ds[i]["text"] for i, s in enumerate(scores) if s <= low_threshold]

        if not high_texts or not low_texts:
            print(
                f"  [reward] {source_name} ({lang}): "
                f"high={len(high_texts)}, low={len(low_texts)}, Skipping (insufficient data)"
            )
            continue

        print(
            f"  [reward] {source_name} ({lang}): "
            f"high={len(high_texts)} (score>={high_threshold:.3f}), "
            f"low={len(low_texts)} (score<={low_threshold:.3f})"
        )

        # 1c. Random pairing: each chosen used max max_pairs_per_chosen times
        # Construct chosen pool: each chosen repeated max_pairs_per_chosen times
        chosen_pool = high_texts * max_pairs_per_chosen
        rng.shuffle(chosen_pool)

        # rejected pool: copy low_texts
        rejected_pool = low_texts.copy()
        rng.shuffle(rejected_pool)

        # Number of pairs = min(chosen_pool, rejected_pool)
        n_pairs = min(len(chosen_pool), len(rejected_pool))

        for i in range(n_pairs):
            # 1d. Select a common Type A prompt for each pair
            prompt_text = get_random_type_a_prompt(lang, rng)

            all_pairs["prompt"].append(
                [{"role": "user", "content": prompt_text}]
            )
            all_pairs["chosen"].append(
                [{"role": "assistant", "content": chosen_pool[i]}]
            )
            all_pairs["rejected"].append(
                [{"role": "assistant", "content": rejected_pool[i]}]
            )

        print(f"  [reward] {source_name} ({lang}): Generated {n_pairs} preference pairs")

    if not all_pairs["prompt"]:
        raise ValueError("Failed to generate any preference pairs, please check data and parameters")

    # 2. Convert to Dataset
    pairs_ds = datasets.Dataset.from_dict(all_pairs)
    print(f"  [reward] Total preference pairs: {len(pairs_ds)}")

    # 3. Shuffle + Split train / validation
    #    train_test_split defaults to shuffle=True, no need to shuffle manually again
    split = pairs_ds.train_test_split(test_size=val_ratio, seed=seed)

    result = datasets.DatasetDict({
        "train": split["train"],
        "validation": split["test"],
    })

    print(f"  [reward] Final: train={len(result['train'])}, validation={len(result['validation'])}")
    return result
