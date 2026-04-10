"""
Type B Data Synthesis Script
============================

This script is independent of pipeline.py and is used to synthesize
SFT Type B data (task-formatted training samples).

Supported workflow:
    build_keyword_pool:
      Collect wordfreq candidates -> spaCy POS filter -> SemEval leakage filter
      -> save data/cache/keyword_pools/{lang}.json

    run:
      Build prompts -> call realtime Gemini (multi-task JSON) -> quality filter
      -> save type_b_{lang}.jsonl

Important Design Principles:
    - Do not use SemEval-provided headline/keyword pairs (prevent leakage)
    - Headlines come from external Babel Briefings dataset
    - Keyword pairs are generated from cached language-specific keyword pool files
    - Final SFT Type B data is written to data/synthesized/type_b_{lang}.jsonl
    - pipeline.py format_sft stage automatically loads type_b_*.jsonl

Usage:
    # Build English keyword pool
    python -m data_preprocessing.synthesize_task_data --lang en --build_keyword_pool_only

    # Synthesize English data
    python -m data_preprocessing.synthesize_task_data --lang en

    # Synthesize Chinese data, specify quantity
    python -m data_preprocessing.synthesize_task_data --lang zh --n_headline 300 --n_keyword 150

    # Synthesize all three languages
    python -m data_preprocessing.synthesize_task_data --lang all

Dependencies:
    - google-genai (Google Gemini API)
    - prompt_templates (Local)
    - datasets (HuggingFace, for loading Babel Briefings)
    - wordfreq (For high-frequency keyword candidates)
    - spaCy (For POS-based noun filtering)

Environment Variables:
    - GEMINI_API_KEY: Google Gemini API Key (Must be set)
"""

import argparse
import csv
import itertools
import json
import os
import random
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from data_preprocessing.prompt_templates import (
    build_headline_prompt,
    build_keyword_prompt,
)
from data_preprocessing.config import (
    REALTIME_MULTI_GROUP_SIZE,
    DEFAULT_KEYWORD_POOL_SIZE,
    DEFAULT_KEYWORD_CANDIDATE_LIMIT,
    SPACY_MODEL_NAMES,
    KEYWORD_ALLOWED_POS,
)

# ============================================================
# Path Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SYNTHESIZED_DIR = DATA_DIR / "synthesized"
KEYWORD_POOL_CACHE_DIR = DATA_DIR / "cache" / "keyword_pools"
SEMEVAL_TASK_FILES = {
    "en": DATA_DIR / "raw" / "semeval_task" / "task-a-en.tsv",
    "zh": DATA_DIR / "raw" / "semeval_task" / "task-a-zh.tsv",
    "es": DATA_DIR / "raw" / "semeval_task" / "task-a-es.tsv",
}


def _normalize_keyword(text: str, lang: str) -> str:
    """Normalize keyword text for matching and caching."""
    normalized = unicodedata.normalize("NFKC", text).strip()
    if lang == "zh":
        return "".join(normalized.split())
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.casefold()


def _strip_accents(text: str) -> str:
    """Remove accent marks while preserving base characters."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _keyword_match_keys(word: str, lang: str) -> set[str]:
    """Build normalized matching keys for leakage filtering."""
    normalized = _normalize_keyword(word, lang)
    if not normalized:
        return set()
    keys = {normalized}
    if lang == "es":
        keys.add(_strip_accents(normalized))
    return {item for item in keys if item}


def _keyword_pair_keys(word1: str, word2: str, lang: str) -> set[tuple[str, str]]:
    """Build pair-level keys for SemEval leakage filtering."""
    keys = set()
    for left in _keyword_match_keys(word1, lang):
        for right in _keyword_match_keys(word2, lang):
            if left and right and left != right:
                keys.add(tuple(sorted((left, right))))
    return keys


def _keyword_pool_path(lang: str, cache_dir: Path = KEYWORD_POOL_CACHE_DIR) -> Path:
    """Return cache file path for a language-specific keyword pool."""
    if lang not in SEMEVAL_TASK_FILES:
        raise ValueError(f"Unsupported language code: '{lang}'")
    return Path(cache_dir) / f"{lang}.json"


def _load_semeval_keyword_constraints(lang: str) -> tuple[set[str], set[tuple[str, str]]]:
    """Load SemEval keyword rows as blocked words and blocked pairs."""
    if lang not in SEMEVAL_TASK_FILES:
        raise ValueError(f"Unsupported language code: '{lang}'")

    blocked_words: set[str] = set()
    blocked_pairs: set[tuple[str, str]] = set()
    semeval_path = SEMEVAL_TASK_FILES[lang]
    with open(semeval_path, "r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj, delimiter="\t")
        for row in reader:
            word1 = (row.get("word1") or "").strip()
            word2 = (row.get("word2") or "").strip()
            headline = (row.get("headline") or "").strip()
            if headline != "-" or word1 == "-" or word2 == "-":
                continue

            blocked_words.update(_keyword_match_keys(word1, lang))
            blocked_words.update(_keyword_match_keys(word2, lang))
            blocked_pairs.update(_keyword_pair_keys(word1, word2, lang))
    return blocked_words, blocked_pairs


def _write_keyword_pool_cache(
    lang: str,
    keywords: list[str],
    target_size: int,
    blocked_words_count: int,
    cache_dir: Path = KEYWORD_POOL_CACHE_DIR,
) -> Path:
    """Write keyword pool cache to JSON."""
    output_path = _keyword_pool_path(lang, cache_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "lang": lang,
        "target_size": target_size,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blocked_words_count": blocked_words_count,
        "keywords": keywords,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _load_keyword_pool(
    lang: str,
    cache_dir: Path = KEYWORD_POOL_CACHE_DIR,
    min_size: int = 2,
) -> list[str]:
    """Load and validate keyword pool cache."""
    pool_path = _keyword_pool_path(lang, cache_dir)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Keyword pool cache not found for '{lang}': {pool_path}. "
            "Please build the keyword pool first."
        )

    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    keywords = payload.get("keywords")
    if not isinstance(keywords, list):
        raise ValueError(f"Invalid keyword pool format in {pool_path}: 'keywords' must be a list.")

    cleaned_keywords: list[str] = []
    seen_keywords: set[str] = set()
    for item in keywords:
        if not isinstance(item, str):
            raise ValueError(f"Invalid keyword pool format in {pool_path}: keyword must be a string.")
        normalized = _normalize_keyword(item, lang)
        if not normalized:
            raise ValueError(f"Invalid keyword pool format in {pool_path}: empty keyword is not allowed.")
        if normalized in seen_keywords:
            raise ValueError(f"Invalid keyword pool format in {pool_path}: duplicate keyword '{normalized}'.")
        seen_keywords.add(normalized)
        cleaned_keywords.append(normalized)

    if len(cleaned_keywords) < min_size:
        raise ValueError(
            f"Keyword pool for '{lang}' has only {len(cleaned_keywords)} entries, "
            f"but at least {min_size} are required."
        )
    return cleaned_keywords


def _collect_wordfreq_candidates(lang: str, candidate_limit: int) -> list[str]:
    """Collect high-frequency candidates from wordfreq."""
    from wordfreq import top_n_list  # pyright: ignore[reportMissingImports]

    return top_n_list(lang, candidate_limit, wordlist="best")


def _load_spacy_pipeline(lang: str):
    """Load language-specific spaCy pipeline for POS filtering."""
    import spacy  # pyright: ignore[reportMissingImports]

    if lang not in SPACY_MODEL_NAMES:
        raise ValueError(f"Unsupported language code: '{lang}'")

    model_name = SPACY_MODEL_NAMES[lang]
    try:
        return spacy.load(model_name, disable=["parser", "ner", "lemmatizer"])
    except OSError as error:
        raise RuntimeError(
            f"spaCy model '{model_name}' is not installed. "
            f"Please run: python -m spacy download {model_name}"
        ) from error


def _is_valid_keyword_surface(word: str, lang: str) -> bool:
    """Apply lightweight language-specific surface filters before POS tagging."""
    normalized = _normalize_keyword(word, lang)
    if not normalized:
        return False

    if lang in {"en", "es"}:
        if " " in normalized or len(normalized) < 2 or len(normalized) > 24:
            return False
        return re.fullmatch(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", normalized, flags=re.UNICODE) is not None

    if len(normalized) < 2 or len(normalized) > 8:
        return False
    return re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff]+", normalized) is not None


def _filter_keyword_candidates(
    lang: str,
    candidates: list[str],
    blocked_words: set[str],
    nlp,
    target_size: int,
) -> list[str]:
    """Filter candidates by normalization, leakage, and POS."""
    prefiltered: list[str] = []
    seen_prefiltered: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_keyword(candidate, lang)
        if not _is_valid_keyword_surface(normalized, lang):
            continue
        if any(match_key in blocked_words for match_key in _keyword_match_keys(normalized, lang)):
            continue
        if normalized in seen_prefiltered:
            continue
        seen_prefiltered.add(normalized)
        prefiltered.append(normalized)

    accepted: list[str] = []
    accepted_set: set[str] = set()
    for normalized, doc in zip(prefiltered, nlp.pipe(prefiltered, batch_size=128)):
        if len(doc) != 1:
            continue
        token = doc[0]
        if token.pos_ not in KEYWORD_ALLOWED_POS[lang]:
            continue
        if token.is_punct or token.like_num:
            continue
        if lang in {"en", "es"} and not token.is_alpha:
            continue
        if normalized in accepted_set:
            continue
        accepted.append(normalized)
        accepted_set.add(normalized)
        if len(accepted) >= target_size:
            break
    return accepted


def build_keyword_pool(
    lang: str,
    target_size: int = DEFAULT_KEYWORD_POOL_SIZE,
    cache_dir: Path = KEYWORD_POOL_CACHE_DIR,
    candidate_limit: int | None = None,
) -> Path:
    """Build and cache a language-specific keyword pool."""
    if target_size < 2:
        raise ValueError("target_size must be at least 2.")

    if candidate_limit is None:
        candidate_limit = max(DEFAULT_KEYWORD_CANDIDATE_LIMIT, target_size * 20)

    blocked_words, _blocked_pairs = _load_semeval_keyword_constraints(lang)
    candidates = _collect_wordfreq_candidates(lang, candidate_limit)
    nlp = _load_spacy_pipeline(lang)
    keywords = _filter_keyword_candidates(
        lang=lang,
        candidates=candidates,
        blocked_words=blocked_words,
        nlp=nlp,
        target_size=target_size,
    )
    if len(keywords) < target_size:
        raise ValueError(
            f"Only collected {len(keywords)} keywords for '{lang}', "
            f"which is below target_size={target_size}. Increase candidate_limit or relax filters."
        )
    return _write_keyword_pool_cache(
        lang=lang,
        keywords=keywords,
        target_size=target_size,
        blocked_words_count=len(blocked_words),
        cache_dir=cache_dir,
    )


# ============================================================
# Gemini API Call
# ============================================================

def _init_gemini_client():
    """Initialize Google Gemini API Client.

    Requires environment variable GEMINI_API_KEY.

    Returns:
        google.genai.Client instance

    Raises:
        ValueError: GEMINI_API_KEY environment variable not set
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Please run: export GEMINI_API_KEY='your-api-key'"
        )

    from google import genai
    client = genai.Client(api_key=api_key)
    print("  Gemini API client initialized successfully")
    return client


def _call_gemini(client, prompt: str, max_retries: int = 3) -> str | None:
    """Call Gemini API to generate response, with retry and rate limit handling.

    Args:
        client: Gemini API client
        prompt: Prompt text sent to the model
        max_retries: Maximum number of retries (handling API rate limits or temporary errors)

    Returns:
        str: Text generated by the model, None if failed
    """
    from google.genai import types

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    # Disable thinking mode, just need direct response
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    # Higher temperature encourages creativity
                    temperature=0.9,
                    max_output_tokens=256,
                ),
            )
            text = response.text
            if text:
                return text.strip()
            return None

        except Exception as e:
            error_msg = str(e).lower()

            # API Rate Limit (429 Too Many Requests) or Service Temporarily Unavailable (503)
            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait_time = 2 ** attempt * 5  # Exponential backoff: 10s, 20s, 40s
                print(f"    API Rate Limit, waiting {wait_time}s before retry ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue

            # Other errors
            print(f"    Gemini call failed ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue

    return None


def _build_multi_generation_prompt(task_records: list[dict], lang: str) -> str:
    """Build a single prompt asking model to solve multiple tasks with JSON output."""
    header = (
        f"You are generating humorous responses in language={lang}.\n"
        "Solve each task independently.\n"
        "Return ONLY a valid JSON array. No markdown, no extra text.\n"
        "Each item must be an object with keys: index, output.\n"
        "index: integer task index.\n"
        "output: one joke response for that task.\n"
    )
    lines = [header, "Tasks:"]
    for idx, item in enumerate(task_records):
        lines.append(f"{idx}. input={item.get('prompt','')}")
    return "\n".join(lines)


def _call_gemini_multi(
    client,
    task_records: list[dict],
    lang: str,
    max_retries: int = 3,
) -> list[dict]:
    """Call realtime Gemini once and get multiple structured outputs."""
    if not task_records:
        return []
    from google.genai import types

    response_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "output": {"type": "string"},
            },
            "required": ["index", "output"],
        },
    }
    prompt = _build_multi_generation_prompt(task_records, lang)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-3-flash-preview",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    temperature=0.9,
                    max_output_tokens=min(8192, max(1024, len(task_records) * 220)),
                ),
            )
            if not response.text:
                return []
            parsed = json.loads(response.text)
            if not isinstance(parsed, list):
                return []
            results: list[dict] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                index = item.get("index")
                output = item.get("output")
                if isinstance(index, int) and isinstance(output, str) and output.strip():
                    results.append(
                        {
                            "index": index,
                            "output": output.strip(),
                        }
                    )
            return results
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait_time = 2 ** attempt * 5
                print(f"    API Rate Limit, waiting {wait_time}s before retry ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"    Gemini multi call failed ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue
    return []


def _generate_multi_responses(
    client,
    prompt_records: list[dict],
    lang: str,
    group_size: int = REALTIME_MULTI_GROUP_SIZE,
) -> list[str | None]:
    """Generate responses for many prompts via grouped realtime multi-calls."""
    outputs: list[str | None] = [None] * len(prompt_records)
    if not prompt_records:
        return outputs

    for start, chunk_outputs, _, _ in _iter_multi_response_chunks(
        client=client,
        prompt_records=prompt_records,
        lang=lang,
        group_size=group_size,
    ):
        for i, value in enumerate(chunk_outputs):
            outputs[start + i] = value
    return outputs


def _format_eta_by_passed(passed: int, target: int, elapsed_s: float) -> str:
    """Estimate ETA by current passed rate."""
    if passed <= 0 or elapsed_s <= 0:
        return "N/A"
    remaining = max(target - passed, 0)
    if remaining == 0:
        return "0s"
    rate = passed / elapsed_s
    if rate <= 0:
        return "N/A"
    eta_s = int(remaining / rate)
    minutes, seconds = divmod(eta_s, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes}m{seconds}s"
    if minutes > 0:
        return f"{minutes}m{seconds}s"
    return f"{seconds}s"


def _iter_multi_response_chunks(
    client,
    prompt_records: list[dict],
    lang: str,
    group_size: int = REALTIME_MULTI_GROUP_SIZE,
):
    """Yield grouped outputs and fallback stats for realtime_multi path."""
    total = len(prompt_records)
    for start in range(0, total, group_size):
        chunk = prompt_records[start:start + group_size]
        chunk_outputs: list[str | None] = [None] * len(chunk)
        fallback_count = 0
        multi_success_count = 0

        chunk_results = _call_gemini_multi(client, chunk, lang=lang)
        if not chunk_results:
            # fallback to single-call realtime for this chunk
            for i, rec in enumerate(chunk):
                single = _call_gemini(client, rec.get("prompt", ""))
                chunk_outputs[i] = single
                fallback_count += 1
            yield start, chunk_outputs, fallback_count, multi_success_count
            continue

        seen = set()
        for item in chunk_results:
            idx = item["index"]
            if 0 <= idx < len(chunk):
                chunk_outputs[idx] = item["output"]
                seen.add(idx)
        multi_success_count = len(seen)

        # fill missing indexes with single-call realtime
        for i, rec in enumerate(chunk):
            if i not in seen:
                single = _call_gemini(client, rec.get("prompt", ""))
                chunk_outputs[i] = single
                fallback_count += 1

        yield start, chunk_outputs, fallback_count, multi_success_count


# ============================================================
# News Headline Acquisition (Babel Briefings)
# ============================================================

def _load_headlines(lang: str, n_samples: int, seed: int = 42) -> list[str]:
    """Randomly sample specified number of news headlines from Babel Briefings dataset.

    Babel Briefings is a multilingual news headline dataset (HuggingFace: felixludos/babel-briefings),
    containing 4.7 million news headlines in 30 languages including English, Chinese, Spanish, etc.

    Args:
        lang: Language code "en" / "zh" / "es"
        n_samples: Number of headlines required
        seed: Random seed

    Returns:
        list[str]: List of news headlines
    """
    import datasets as ds_lib

    print(f"  Loading {lang} headlines from Babel Briefings (streaming mode)...")

    # Use streaming=True to avoid downloading full ~5GB dataset
    dataset = ds_lib.load_dataset(
        "felixludos/babel-briefings",
        split="train",
        streaming=True,
    )

    # Filter by language, collect enough candidate headlines
    # Collect more (3x) to have sufficient pool for random sampling later
    collect_target = n_samples * 3
    candidates = []

    for example in dataset:
        if example.get("language") == lang:
            title = example.get("title", "")
            # Basic filter: non-empty, reasonable length (not too short or too long)
            if title and 10 <= len(title) <= 300:
                candidates.append(title)
            if len(candidates) >= collect_target:
                break

    print(f"  Collected {len(candidates)} candidate headlines")

    if len(candidates) == 0:
        raise ValueError(f"Failed to fetch {lang} headlines from Babel Briefings")

    # Random sampling
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n_samples]

    print(f"  Selected {len(selected)} headlines")
    return selected


# ============================================================
# Keyword Pair Generation
# ============================================================

def _generate_keyword_pairs(lang: str, n_pairs: int, seed: int = 42) -> list[tuple[str, str]]:
    """Randomly pair keywords from pool to generate unique keyword pairs.

    Pairing rules:
    - Two words are not identical
    - Same combination not repeated (Order irrelevant: (a,b) and (b,a) are same)

    Args:
        lang: Language code
        n_pairs: Number of pairs to generate
        seed: Random seed

    Returns:
        list[tuple[str, str]]: List of keyword pairs
    """
    pool = _load_keyword_pool(lang, min_size=2)
    _blocked_words, blocked_pairs = _load_semeval_keyword_constraints(lang)

    # Generate all unique combinations of two, excluding blocked SemEval pairs.
    all_combos = [
        (word1, word2)
        for word1, word2 in itertools.combinations(pool, 2)
        if _keyword_pair_keys(word1, word2, lang).isdisjoint(blocked_pairs)
    ]

    if n_pairs > len(all_combos):
        print(
            f"  Warning: Requested {n_pairs} keyword pairs, but vocabulary can only generate {len(all_combos)} combinations, "
            f"using all combinations"
        )
        n_pairs = len(all_combos)

    rng = random.Random(seed)
    selected = rng.sample(all_combos, n_pairs)

    print(f"  Generated {len(selected)} keyword pairs")
    return selected


# ============================================================
# Quality Filtering
# ============================================================

def _filter_headline_response(response: str) -> bool:
    """Check if the response generated for headline subtask is qualified.

    Filter conditions:
    - Non-empty
    - Reasonable length (10 ~ 500 characters)
    - Does not contain common refusal patterns ("I cannot", "I'm sorry", etc.)

    Args:
        response: Response text generated by model

    Returns:
        bool: True if qualified
    """
    if not response or not response.strip():
        return False

    text = response.strip()

    # Length check
    if len(text) < 10 or len(text) > 500:
        return False

    # Refusal pattern check (Model sometimes refuses to generate humor)
    refusal_patterns = [
        "i cannot", "i can't", "i'm sorry", "i apologize",
        "as an ai", "as a language model",
        "不能", "抱歉", "对不起", "作为一个ai", "作为语言模型",
        "no puedo", "lo siento", "disculpa",
    ]
    text_lower = text.lower()
    for pattern in refusal_patterns:
        if pattern in text_lower:
            return False

    return True


def _filter_keyword_response(response: str, word1: str, word2: str) -> bool:
    """Check if the response generated for keyword subtask is qualified.

    Filter conditions:
    - Meets all conditions of _filter_headline_response
    - Extra: Response must contain both word1 and word2 (case-insensitive)

    Args:
        response: Response text generated by model
        word1: First required keyword
        word2: Second required keyword

    Returns:
        bool: True if qualified
    """
    # Check general conditions first
    if not _filter_headline_response(response):
        return False

    # Extra check: both keywords must appear in response (case-insensitive)
    text_lower = response.lower()
    if word1.lower() not in text_lower:
        return False
    if word2.lower() not in text_lower:
        return False

    return True


# ============================================================
# Synthesis Main Logic
# ============================================================


def synthesize_for_language(
    lang: str,
    n_headline: int = 200,
    n_keyword: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Synthesize Type B data for specified language using realtime_multi."""
    oversample_ratio = 1.5
    client = _init_gemini_client()
    all_samples = []

    # ---- Headline Part ----
    if n_headline > 0:
        print(f"\n  --- Headline Subtask (Target: {n_headline}) ---")
        stage_start = time.time()
        n_headline_fetch = int(n_headline * oversample_ratio)
        headlines = _load_headlines(lang, n_headline_fetch, seed=seed)

        passed = 0
        failed = 0
        fallback_total = 0
        multi_success_total = 0
        headline_records = [{"prompt": build_headline_prompt(headline, lang)} for headline in headlines]
        for start, chunk_outputs, fallback_count, multi_success_count in _iter_multi_response_chunks(
            client=client,
            prompt_records=headline_records,
            lang=lang,
        ):
            fallback_total += fallback_count
            multi_success_total += multi_success_count
            for offset, response in enumerate(chunk_outputs):
                if passed >= n_headline:
                    break
                user_prompt = headline_records[start + offset]["prompt"]
                if response and _filter_headline_response(response):
                    all_samples.append(
                        {
                            "messages": [
                                {"role": "user", "content": user_prompt},
                                {"role": "assistant", "content": response},
                            ]
                        }
                    )
                    passed += 1
                else:
                    failed += 1
            processed = min(len(headline_records), start + len(chunk_outputs))
            elapsed_s = time.time() - stage_start
            eta_text = _format_eta_by_passed(passed, n_headline, elapsed_s)
            print(
                f"    Progress: processed={processed}/{len(headline_records)}, "
                f"passed={passed}/{n_headline}, filtered={failed}, "
                f"multi_ok={multi_success_total}, fallback={fallback_total}, eta={eta_text}"
            )
            time.sleep(0.1)
            if passed >= n_headline:
                print(f"    Target {n_headline} reached, stopping early (processed={processed})")
                break

        print(f"  Headline Done: {passed} passed, {failed} filtered")
        if passed < n_headline:
            print(
                f"  Warning: Target not reached (Target {n_headline}, Actual {passed}), "
                "try increasing oversample_ratio or lowering group size."
            )

    # ---- Keyword Part ----
    if n_keyword > 0:
        print(f"\n  --- Keyword Subtask (Target: {n_keyword}) ---")
        stage_start = time.time()
        n_keyword_fetch = int(n_keyword * oversample_ratio)
        keyword_pairs = _generate_keyword_pairs(lang, n_keyword_fetch, seed=seed)

        passed = 0
        failed = 0
        fallback_total = 0
        multi_success_total = 0
        keyword_records = [
            {"word1": w1, "word2": w2, "prompt": build_keyword_prompt(w1, w2, lang)}
            for w1, w2 in keyword_pairs
        ]
        for start, chunk_outputs, fallback_count, multi_success_count in _iter_multi_response_chunks(
            client=client,
            prompt_records=keyword_records,
            lang=lang,
        ):
            fallback_total += fallback_count
            multi_success_total += multi_success_count
            for offset, response in enumerate(chunk_outputs):
                if passed >= n_keyword:
                    break
                record = keyword_records[start + offset]
                if response and _filter_keyword_response(response, record["word1"], record["word2"]):
                    all_samples.append(
                        {
                            "messages": [
                                {"role": "user", "content": record["prompt"]},
                                {"role": "assistant", "content": response},
                            ]
                        }
                    )
                    passed += 1
                else:
                    failed += 1
            processed = min(len(keyword_records), start + len(chunk_outputs))
            elapsed_s = time.time() - stage_start
            eta_text = _format_eta_by_passed(passed, n_keyword, elapsed_s)
            print(
                f"    Progress: processed={processed}/{len(keyword_records)}, "
                f"passed={passed}/{n_keyword}, filtered={failed}, "
                f"multi_ok={multi_success_total}, fallback={fallback_total}, eta={eta_text}"
            )
            time.sleep(0.1)
            if passed >= n_keyword:
                print(f"    Target {n_keyword} reached, stopping early (processed={processed})")
                break

        print(f"  Keyword Done: {passed} passed, {failed} filtered")
        if passed < n_keyword:
            print(
                f"  Warning: Target not reached (Target {n_keyword}, Actual {passed}), "
                "try increasing oversample_ratio or lowering group size."
            )

    print(f"\n  Total synthesized: {len(all_samples)} samples")
    return all_samples


def save_synthesized(samples: list[dict], lang: str) -> Path:
    """Save synthesized samples as JSONL file.

    Args:
        samples: Synthesized SFT sample list
        lang: Language code

    Returns:
        Path: Saved file path
    """
    SYNTHESIZED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SYNTHESIZED_DIR / f"type_b_{lang}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    return output_path


# ============================================================
# Command Line Entry
# ============================================================

def main():
    """Command line entry point, parse arguments and run synthesis."""
    parser = argparse.ArgumentParser(
        description="Synthesize Type B Task Formatted Data (Requires Gemini API)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["en", "zh", "es", "all"],
        help="Target language, or 'all' to synthesize all three languages",
    )
    parser.add_argument(
        "--n_headline",
        type=int,
        default=70,
        help="Number of headline subtask samples per language (default 70)",
    )
    parser.add_argument(
        "--n_keyword",
        type=int,
        default=70,
        help="Number of keyword subtask samples per language (default 70)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )
    parser.add_argument(
        "--build_keyword_pool_only",
        action="store_true",
        help="Only build keyword pool cache file(s) and skip Gemini synthesis",
    )
    parser.add_argument(
        "--keyword_pool_size",
        type=int,
        default=DEFAULT_KEYWORD_POOL_SIZE,
        help=f"Target keyword pool size per language (default {DEFAULT_KEYWORD_POOL_SIZE})",
    )
    parser.add_argument(
        "--candidate_limit",
        type=int,
        default=None,
        help="Optional upper bound for wordfreq candidate collection",
    )

    args = parser.parse_args()

    languages = ["en", "zh", "es"] if args.lang == "all" else [args.lang]
    if args.build_keyword_pool_only:
        for lang in languages:
            print(f"\n{'=' * 60}")
            print(f"Building keyword pool: lang={lang}")
            print(
                f"  target_size: {args.keyword_pool_size}, "
                f"candidate_limit: {args.candidate_limit or 'auto'}"
            )
            print(f"{'=' * 60}")
            output_path = build_keyword_pool(
                lang=lang,
                target_size=args.keyword_pool_size,
                candidate_limit=args.candidate_limit,
            )
            print(f"\nSaved keyword pool: {output_path}")
        print("\nDone.")
        return

    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Synthesizing Type B data: lang={lang}")
        print(f"  headline: {args.n_headline} rows, keyword: {args.n_keyword} rows")
        print(f"{'=' * 60}")

        samples = synthesize_for_language(
            lang=lang,
            n_headline=args.n_headline,
            n_keyword=args.n_keyword,
            seed=args.seed,
        )
        output_path = save_synthesized(samples, lang)
        print(f"\nSaved: {output_path} ({len(samples)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
