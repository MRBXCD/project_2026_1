"""
Layer 1: Source Parsers
=======================

This module is responsible for parsing various raw data sources into a Unified Intermediate Format.

Unified Intermediate Format Schema:
    {
        "text": str,              # Joke/Humor text body
        "lang": "en"|"zh"|"es",   # Language identifier
        "score": float | None,    # Quality score normalized to [0, 1], None if no score
        "source": str,            # Data source identifier
    }

Conventions for each parser function:
    - Input: Raw data file path (str or Path)
    - Output: datasets.Dataset object, schema matches the unified format above
    - Most parsers perform "read + field extraction + score normalization".
      CFun parser additionally performs instruction-based extraction and quality filtering
      to keep CFun cleaning rules in one place.

SemEval parser is an exception:
    - Its output schema differs from the Unified Intermediate Format
    - It directly serves the GRPO Formatter
    - Output schema:
      {
          "id": str,
          "headline": str,       # News headline, empty string if no headline
          "keywords": list[str], # List of keywords, empty list if no keywords
          "lang": str,
          "subtask": str,        # "headline" or "keyword"
      }

Dependencies:
    - datasets (HuggingFace)
    - pandas
"""

import csv
import re
from pathlib import Path

import datasets
import pandas as pd

from data_preprocessing.config import (
    RJOKES_SCORE_CAP,
    HUMOR_SCORE_MAX,
    CFUN_MIN_LEN,
    CFUN_MAX_LEN,
)


# ============================================================
# Constant Definitions
# ============================================================

# CFun extraction and cleaning rules
INSTR_HUMOR_DETECT = (
    "以下是一段文本，请分析它是否具有幽默性。幽默性指该文本是否可能引起读者发笑，"
    "或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外）营造幽默效果。只需要输出“幽默”或“不幽默”。"
)
INSTR_HUMOR_REASON = (
    "请阅读以下文字，分析其幽默的原因。幽默性指该文本是否可能引起读者发笑，"
    "或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外等等方式）营造幽默效果。请你写出以下文字幽默的原因："
)
INSTR_FIRST_SENTENCE_COMPLETION = (
    "我将给你笑话的第一句话，请你生成整个笑话。笑话的第一句话如下："
)
CFUN_REMOVE_LABEL_PATTERN = re.compile(r"(?:标题|内容)\s*[：:]\s*")
CFUN_WHITESPACE_PATTERN = re.compile(r"\s+")


# ============================================================
# Parser 1: rJokes (English)
# ============================================================

def parse_rjokes(data_dir: str | Path) -> datasets.Dataset:
    """Parse rJokes dataset and output Unified Intermediate Format.

    rJokes data comes from Reddit r/Jokes, containing joke text and community votes.
    Original format is TSV.gz, two columns per row: score (int) and joke (str).
    The dataset is split into train / dev / test files, this function merges them.

    Args:
        data_dir: rJokes raw data directory path, expected to contain train.tsv.gz, dev.tsv.gz, test.tsv.gz

    Returns:
        datasets.Dataset: Unified Intermediate Format
            - text (str): Joke text
            - lang (str): Fixed as "en"
            - score (float): Normalized to [0, 1], calculation: min(raw_score, 20) / 20
            - source (str): Fixed as "rjokes"
    """
    data_dir = Path(data_dir)

    # 1. Read and merge train / dev / test TSV.gz files
    #    Note: skip preprocessed.csv.gz (different format)
    target_files = ["train.tsv.gz", "dev.tsv.gz", "test.tsv.gz"]
    dfs = []
    for filename in target_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  [rjokes] Warning: {filepath} does not exist, skipping")
            continue
        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["score", "joke"],
            encoding="utf-8",
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
            engine="python",
        )
        dfs.append(df)
        print(f"  [rjokes] Loaded {filename}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)

    # 2. Basic cleaning: filter rows where joke is empty or NaN
    df = df.dropna(subset=["joke"])
    df = df[df["joke"].str.strip().astype(bool)]

    # 3. Ensure score column is numeric
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)

    # 4. 构建统一中间格式
    # datasets.Dataset.from_dict() is a method that creates a Dataset from a dictionary of columns, i.e.:
    # records = {
    # "text":   ["joke1", "joke2", "joke3"],      # 所有行的 text 列
    # "lang":   ["en",    "en",    "en"],          # 所有行的 lang 列
    # "score":  [0.25,    0.5,     1.0],           # 所有行的 score 列
    # "source": ["rjokes","rjokes","rjokes"],      # 所有行的 source 列
    # }
    records = {
        "text": df["joke"].tolist(),
        "lang": ["en"] * len(df),
        "score": [
            min(float(s), RJOKES_SCORE_CAP) / RJOKES_SCORE_CAP
            for s in df["score"]
        ],
        "source": ["rjokes"] * len(df),
    }

    return datasets.Dataset.from_dict(records)


# ============================================================
# Parser 2: CFun (Chinese)
# ============================================================

def parse_cfun(cache_dir: str | Path) -> datasets.Dataset:
    """Parse CFun dataset and output Unified Intermediate Format.

    CFun is pulled directly from HuggingFace and cleaned in-parser to keep a single source
    of truth for all downstream stages.

    Args:
        cache_dir: HuggingFace cache directory path used by datasets.load_dataset.

    Returns:
        datasets.Dataset: Unified Intermediate Format
            - text (str): Cleaned joke text
            - lang (str): Fixed as "zh"
            - score (float | None): Fixed as None (CFun has no score info)
            - source (str): Fixed as "cfun"
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset from HuggingFace cache
    ds = datasets.load_dataset(
        "ZhenghanYU/CFunSet",
        cache_dir=str(cache_dir),
    )
    ds = ds["train"]
    print(f"  [cfun] Raw data: {len(ds)} rows")

    def _extract_candidate(example: dict) -> str | None:
        instruction = (example.get("instruction") or "").strip()
        input_text = (example.get("input") or "").strip()
        output_text = (example.get("output") or "").strip()

        if instruction.startswith("生成一个"):
            return output_text
        if instruction == INSTR_HUMOR_DETECT or instruction == INSTR_HUMOR_REASON:
            return input_text
        if instruction == INSTR_FIRST_SENTENCE_COMPLETION:
            return f"{input_text}{output_text}"
        return None

    def _clean_text(text: str) -> str:
        normalized = CFUN_WHITESPACE_PATTERN.sub(" ", text).strip()
        normalized = CFUN_REMOVE_LABEL_PATTERN.sub("", normalized)
        return CFUN_WHITESPACE_PATTERN.sub(" ", normalized).strip()

    def _has_meaningful_character(text: str) -> bool:
        for char in text:
            if char.isalnum() or "\u4e00" <= char <= "\u9fff":
                return True
        return False

    # 2. Extract and filter records
    stats = {
        "matched_theme_keyword_output": 0,
        "matched_humor_detect_reason_input": 0,
        "matched_first_sentence_concat": 0,
        "drop_unmatched_instruction": 0,
        "drop_empty_candidate": 0,
        "drop_too_short": 0,
        "drop_too_long": 0,
        "drop_non_meaningful_text": 0,
        "drop_duplicate": 0,
    }
    seen_texts = set()
    texts: list[str] = []
    for example in ds:
        candidate = _extract_candidate(example)
        if candidate is None:
            stats["drop_unmatched_instruction"] += 1
            continue

        instruction = (example.get("instruction") or "").strip()
        if instruction.startswith("生成一个"):
            stats["matched_theme_keyword_output"] += 1
        elif instruction == INSTR_HUMOR_DETECT or instruction == INSTR_HUMOR_REASON:
            stats["matched_humor_detect_reason_input"] += 1
        elif instruction == INSTR_FIRST_SENTENCE_COMPLETION:
            stats["matched_first_sentence_concat"] += 1

        if not candidate:
            stats["drop_empty_candidate"] += 1
            continue

        cleaned = _clean_text(candidate)
        if len(cleaned) < CFUN_MIN_LEN:
            stats["drop_too_short"] += 1
            continue
        if len(cleaned) > CFUN_MAX_LEN:
            stats["drop_too_long"] += 1
            continue
        if not _has_meaningful_character(cleaned):
            stats["drop_non_meaningful_text"] += 1
            continue
        if cleaned in seen_texts:
            stats["drop_duplicate"] += 1
            continue

        seen_texts.add(cleaned)
        texts.append(cleaned)

    print(f"  [cfun] matched theme/keyword -> output: {stats['matched_theme_keyword_output']}")
    print(f"  [cfun] matched detect/reason -> input: {stats['matched_humor_detect_reason_input']}")
    print(f"  [cfun] matched first-sentence -> input+output: {stats['matched_first_sentence_concat']}")
    print(f"  [cfun] dropped unmatched instruction: {stats['drop_unmatched_instruction']}")
    print(f"  [cfun] dropped empty candidate: {stats['drop_empty_candidate']}")
    print(f"  [cfun] dropped too short: {stats['drop_too_short']}")
    print(f"  [cfun] dropped too long: {stats['drop_too_long']}")
    print(f"  [cfun] dropped non-meaningful text: {stats['drop_non_meaningful_text']}")
    print(f"  [cfun] dropped duplicate: {stats['drop_duplicate']}")
    print(f"  [cfun] final kept: {len(texts)}")

    records = {
        "text": texts,
        "lang": ["zh"] * len(texts),
        "score": [None] * len(texts),
        "source": ["cfun"] * len(texts),
    }
    return datasets.Dataset.from_dict(records)


# ============================================================
# Parser 3: HAHA 2019 (Spanish)
# ============================================================

def parse_haha(data_dir: str | Path) -> datasets.Dataset:
    """Parse HAHA 2019 dataset and output Unified Intermediate Format.

    HAHA 2019 contains Spanish tweets with humor annotations, including binary label (is_humor)
    and funniness score (funniness_average).

    Note: This parser retains all data (including is_humor=0), no filtering.
    Data with is_humor=0 will be used as rejected source in subsequent preference pair construction.

    Args:
        data_dir: HAHA data directory path, expected to contain haha_2019_train.csv and
                  haha_2019_test_gold.csv

    Returns:
        datasets.Dataset: Unified Intermediate Format
            - text (str): Tweet text
            - lang (str): Fixed as "es"
            - score (float | None):
                - When is_humor=1: funniness_average / 5.0, normalized to [0, 1]
                - When is_humor=0: 0.0 (explicitly marked as not humorous)
            - source (str): Fixed as "haha"
    """
    data_dir = Path(data_dir)

    # 1. Read train and test_gold CSVs and merge
    target_files = ["haha_2019_train.csv", "haha_2019_test_gold.csv"]
    dfs = []
    for filename in target_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  [haha] Warning: {filepath} does not exist, skipping")
            continue
        df = pd.read_csv(filepath, encoding="utf-8")
        dfs.append(df)
        print(f"  [haha] Loaded {filename}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)

    # 2. Basic cleaning: filter rows where text is empty or NaN
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().astype(bool)]

    # 3. Calculate normalized score
    #    - is_humor=1: use funniness_average / 5.0
    #    - is_humor=0: fixed as 0.0
    def _compute_score(row):
        if row["is_humor"] == 1:
            avg = row.get("funniness_average", 0.0)
            # funniness_average might be NaN (incomplete annotation)
            if pd.isna(avg):
                return 0.0
            return float(avg) / HUMOR_SCORE_MAX
        return 0.0

    scores = df.apply(_compute_score, axis=1).tolist()

    # 4. Build Unified Intermediate Format
    records = {
        "text": df["text"].tolist(),
        "lang": ["es"] * len(df),
        "score": scores,
        "source": ["haha"] * len(df),
    }

    return datasets.Dataset.from_dict(records)


# ============================================================
# Parser 4: Chinese Humor Multi-Labeled (Chinese)
# ============================================================

def parse_chinese_humor(data_dir: str | Path) -> datasets.Dataset:
    """Parse Chinese Humor Multi-Labeled dataset and output Unified Intermediate Format.

    This dataset contains Traditional Chinese jokes and HumorLevel (1-5) ratings.
    Original format is TSV (tab-separated), columns: ID, Title, Content, HumorLevel.

    Uses JokeHumorLevel.txt (complete data, including train+test).

    Args:
        data_dir: Chinese_Humor_Multi-Labeled directory path,
                  expected to contain mlabel_corpora/JokeHumorLevel.txt

    Returns:
        datasets.Dataset: Unified Intermediate Format
            - text (str): Joke content (Content field)
            - lang (str): Fixed as "zh"
            - score (float): HumorLevel / 5.0, normalized to [0, 1]
            - source (str): Fixed as "chinese_humor"
    """
    data_dir = Path(data_dir)
    filepath = data_dir / "mlabel_corpora" / "JokeHumorLevel.txt"

    # 1. Read TSV file
    #    Columns: ID, Title, Content, HumorLevel (tab-separated, with header)
    df = pd.read_csv(
        filepath,
        sep="\t",
        encoding="utf-8",
        dtype={"HumorLevel": int},
    )
    print(f"  [chinese_humor] Loaded JokeHumorLevel.txt: {len(df)} rows")

    # 2. Basic cleaning: filter rows where Content is empty or NaN
    df = df.dropna(subset=["Content"])
    df = df[df["Content"].str.strip().astype(bool)]

    # 3. Build Unified Intermediate Format
    records = {
        "text": df["Content"].tolist(),
        "lang": ["zh"] * len(df),
        "score": [float(h) / HUMOR_SCORE_MAX for h in df["HumorLevel"]],
        "source": ["chinese_humor"] * len(df),
    }

    return datasets.Dataset.from_dict(records)


# ============================================================
# Parser 5: SemEval Task A (Multilingual) — Special Schema
# ============================================================

def parse_semeval(data_dir: str | Path) -> datasets.Dataset:
    """Parse SemEval Task A dataset.

    Unlike other parsers, this parser does not output Unified Intermediate Format,
    but outputs a schema dedicated to GRPO Formatter.

    SemEval Task A contains two mutually exclusive subtasks:
    - Headline-based: Has headline, word1/word2 are both "-"
    - Keyword-based: Has word1/word2, headline is "-"

    This parser automatically distinguishes between the two subtasks.

    Args:
        data_dir: semeval_task directory path, expected to contain
                  task-a-en.tsv, task-a-zh.tsv, task-a-es.tsv

    Returns:
        datasets.Dataset: GRPO dedicated schema
            - id (str): Sample ID, e.g., "en_2001"
            - headline (str): News headline, has value for headline subtask, empty string for keyword subtask
            - keywords (list[str]): List of keywords, has value for keyword subtask, empty list for headline subtask
            - lang (str): Language identifier "en" / "zh" / "es"
            - subtask (str): Subtask type "headline" or "keyword"
    """
    data_dir = Path(data_dir)

    all_records = {
        "id": [],
        "headline": [],
        "keywords": [],
        "lang": [],
        "subtask": [],
    }

    # 1. Iterate over task-a-*.tsv files in the directory
    for filepath in sorted(data_dir.glob("task-a-*.tsv")):
        # Extract language code from filename: task-a-en.tsv -> "en"
        lang = filepath.stem.split("-")[-1]

        df = pd.read_csv(filepath, sep="\t", encoding="utf-8", dtype=str)
        print(f"  [semeval] Loaded {filepath.name}: {len(df)} rows (lang={lang})")

        # 2. Determine subtask type and extract fields for each row
        for _, row in df.iterrows():
            row_id = str(row["id"])
            w1 = str(row["word1"]).strip()
            w2 = str(row["word2"]).strip()
            headline = str(row["headline"]).strip()

            if w1 == "-" and w2 == "-":
                # Headline subtask: has headline, no keywords
                all_records["id"].append(row_id)
                all_records["headline"].append(headline)
                all_records["keywords"].append([])
                all_records["lang"].append(lang)
                all_records["subtask"].append("headline")
            elif headline == "-":
                # Keyword subtask: has keywords, no headline
                all_records["id"].append(row_id)
                all_records["headline"].append("")
                all_records["keywords"].append([w1, w2])
                all_records["lang"].append(lang)
                all_records["subtask"].append("keyword")
            else:
                # Should not happen (both headline and keywords exist), print warning and treat as headline
                print(f"  [semeval] Warning: {row_id} has both headline and keywords, treating as headline")
                all_records["id"].append(row_id)
                all_records["headline"].append(headline)
                all_records["keywords"].append([w1, w2])
                all_records["lang"].append(lang)
                all_records["subtask"].append("headline")

    return datasets.Dataset.from_dict(all_records)


# ============================================================
# Unified Entry: Parse All Data Sources
# ============================================================

def parse_all(raw_data_dir: str | Path) -> dict[str, datasets.Dataset]:
    """Parse all raw data sources and return a dictionary.

    This is the main entry function called by pipeline.py. It sequentially calls each parser
    and organizes the results into a dictionary.

    Args:
        raw_data_dir: Root directory of raw data, expected structure:
            raw_data_dir/
            ├── rjoke/
            ├── cfun/
            ├── haha/
            ├── Chinese_Humor_Multi-Labeled/
            └── semeval_task/

    Returns:
        dict: Key is data source name, value is corresponding datasets.Dataset
            {
                "rjokes": Dataset (Unified Intermediate Format),
                "cfun": Dataset (Unified Intermediate Format),
                "haha": Dataset (Unified Intermediate Format),
                "chinese_humor": Dataset (Unified Intermediate Format),
                "semeval": Dataset (GRPO dedicated format),
            }
    """
    raw_data_dir = Path(raw_data_dir)

    results = {}

    # --- Humor Data (Unified Intermediate Format) ---
    results["rjokes"] = parse_rjokes(raw_data_dir / "rjoke")
    results["cfun"] = parse_cfun(raw_data_dir / "cfun")
    results["haha"] = parse_haha(raw_data_dir / "haha")
    results["chinese_humor"] = parse_chinese_humor(
        raw_data_dir / "Chinese_Humor_Multi-Labeled"
    )

    # --- SemEval (GRPO dedicated format) ---
    results["semeval"] = parse_semeval(raw_data_dir / "semeval_task")

    # Print summary of parsing results for each source
    for name, ds in results.items():
        print(f"  [{name}] {len(ds)} samples, columns: {ds.column_names}")

    return results
