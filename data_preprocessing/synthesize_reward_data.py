"""
Reward Model Hard-Negative Data Synthesis Script
==================================================

This script synthesizes preference pairs for reward model training by:
    1. Loading positive (chosen) joke samples:
       - Chinese: Rule-filtered from local pure_jokes.csv (pre-extracted CFun jokes)
       - English: High-score jokes from rJokes (via unified_all.jsonl)
       - Spanish: High-score jokes from HAHA (via unified_all.jsonl)
    2. Generating negative (rejected) samples via Gemini API:
       - Plain, boring, non-humorous statements in each target language
    3. Assembling preference pairs in TRL-compatible format

Output files are saved to data/synthesized/reward_neg_{lang}.jsonl and will be
loaded by format_reward_pairs() during the pipeline's format_reward stage.

Usage:
    python -m data_preprocessing.synthesize_reward_data --lang zh --n_samples 9000
    python -m data_preprocessing.synthesize_reward_data --lang en --n_samples 3000
    python -m data_preprocessing.synthesize_reward_data --lang all --n_samples 3000

Dependencies:
    - google-genai (Google Gemini API)
    - data_preprocessing.prompt_templates (Local)
    - data_preprocessing.synthesize_task_data (Local, for Gemini API utilities)

Environment Variables:
    - GEMINI_API_KEY: Google Gemini API Key (Must be set)
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

from data_preprocessing.prompt_templates import get_random_type_a_prompt
from data_preprocessing.synthesize_task_data import _init_gemini_client


# ============================================================
# Path Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHESIZED_DIR = PROJECT_ROOT / "data" / "synthesized"
PREPROCESSED_DIR = PROJECT_ROOT / "data" / "preprocessed"
UNIFIED_ALL_FILE = PREPROCESSED_DIR / "unified_all.jsonl"
CFUN_JOKES_CSV = PROJECT_ROOT / "data" / "cfun" / "pure_jokes.csv"


# ============================================================
# Gemini Batch Prompts for Non-Humorous Text Generation
# ============================================================

BATCH_SIZE = 100

_BORING_TEXT_BATCH_PROMPTS = {
    "en": (
        f"Generate exactly {BATCH_SIZE} short, plain, boring statements that are NOT funny at all. "
        "Each statement should be 1-2 sentences, like something you'd find in a random "
        "description or observation. No humor, no wordplay, no punchline. "
        "Each statement must be unique and different from the others."
    ),
    "zh": (
        f"生成恰好{BATCH_SIZE}条简短、平淡、无聊的陈述，完全不好笑。"
        "每条1-2句话，像日常描述或观察记录。不要幽默、不要双关、不要笑点。"
        "每条陈述必须互不相同。"
    ),
    "es": (
        f"Genera exactamente {BATCH_SIZE} declaraciones cortas, simples y aburridas "
        "que NO sean graciosas en absoluto. "
        "Cada una debe ser 1-2 oraciones, como una descripción cotidiana. "
        "Sin humor, sin juegos de palabras. "
        "Cada declaración debe ser única y diferente de las demás."
    ),
}

_BATCH_RESPONSE_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}


def _call_gemini_batch(
    client,
    prompt: str,
    max_retries: int = 3,
) -> list[str]:
    """Call Gemini API with structured JSON output to generate a batch of texts.

    Uses response_mime_type='application/json' and response_json_schema
    to guarantee the output is a valid JSON array of strings.

    Args:
        client: Gemini API client.
        prompt: Prompt requesting multiple text items.
        max_retries: Maximum retries for rate limits or transient errors.

    Returns:
        List of generated text strings. Empty list on failure.
    """
    from google.genai import types

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_json_schema=_BATCH_RESPONSE_SCHEMA,
                    temperature=0.9,
                    max_output_tokens=16384,
                ),
            )
            if not response.text:
                return []
            items = json.loads(response.text)
            if isinstance(items, list):
                return [str(item).strip() for item in items if item]
            return []

        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait_time = 2 ** attempt * 5
                print(f"    API Rate Limit, waiting {wait_time}s ({attempt}/{max_retries})")
                time.sleep(wait_time)
                continue
            print(f"    Gemini batch call failed ({attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                time.sleep(2)
                continue

    return []


# ============================================================
# Chosen Sample Loading
# ============================================================

def load_cfun_jokes(n_samples: int, seed: int = 42) -> list[str]:
    """Load and filter CFun jokes from pre-extracted pure_jokes.csv.

    Applies rule-based quality filters: deduplication, length filtering,
    and whitespace stripping. No LLM-as-Judge or instruction prefix
    filtering needed — the CSV already contains standalone joke texts.

    Args:
        n_samples: Number of joke texts to return.
        seed: Random seed for reproducible sampling.

    Returns:
        List of joke text strings.

    Raises:
        FileNotFoundError: If pure_jokes.csv does not exist.
        ValueError: If no jokes remain after filtering.
    """
    if not CFUN_JOKES_CSV.exists():
        raise FileNotFoundError(
            f"{CFUN_JOKES_CSV} does not exist. "
            "Place the pre-extracted CFun jokes CSV at data/cfun/pure_jokes.csv."
        )

    with open(CFUN_JOKES_CSV, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        raw_jokes = [row[0] for row in reader if row]

    print(f"  CFun CSV raw: {len(raw_jokes)} rows")

    unique_jokes = list(dict.fromkeys(raw_jokes))
    print(f"  CFun after dedup: {len(unique_jokes)} rows")

    filtered = [
        text.strip()
        for text in unique_jokes
        if text.strip() and 10 <= len(text.strip()) <= 500
    ]
    print(f"  CFun after quality filter: {len(filtered)} rows")

    if not filtered:
        raise ValueError(
            "No CFun jokes passed filtering. Check data/cfun/pure_jokes.csv content."
        )

    rng = random.Random(seed)
    rng.shuffle(filtered)

    selected = filtered[:n_samples]
    print(f"  CFun selected: {len(selected)} jokes")
    return selected


def load_high_score_jokes(
    lang: str,
    n_samples: int,
    score_threshold: float = 0.7,
    seed: int = 42,
) -> list[str]:
    """Load high-score jokes for a given language from unified_all.jsonl.

    Args:
        lang: Language code ("en" or "es").
        n_samples: Number of joke texts to return.
        score_threshold: Minimum normalized score to qualify as high-score.
        seed: Random seed for reproducible sampling.

    Returns:
        List of joke text strings.

    Raises:
        FileNotFoundError: If unified_all.jsonl does not exist.
    """
    if not UNIFIED_ALL_FILE.exists():
        raise FileNotFoundError(
            f"{UNIFIED_ALL_FILE} does not exist. Run --stage parse first."
        )

    _LANG_SOURCE = {"en": "rjokes", "es": "haha"}
    source = _LANG_SOURCE.get(lang)
    if source is None:
        raise ValueError(f"load_high_score_jokes only supports en/es, got: {lang}")

    candidates = []
    with open(UNIFIED_ALL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if (
                record.get("source") == source
                and record.get("score") is not None
                and record["score"] >= score_threshold
            ):
                text = record.get("text", "").strip()
                if text and 10 <= len(text) <= 1000:
                    candidates.append(text)

    print(f"  {source} ({lang}): {len(candidates)} jokes with score >= {score_threshold}")

    rng = random.Random(seed)
    rng.shuffle(candidates)

    selected = candidates[:n_samples]
    print(f"  {source} ({lang}): selected {len(selected)} jokes")
    return selected


# ============================================================
# Rejected Sample Generation
# ============================================================

def _filter_boring_response(response: str) -> bool:
    """Check if a generated boring statement passes quality filters."""
    if not response or not response.strip():
        return False
    text = response.strip()
    if len(text) < 5 or len(text) > 500:
        return False
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


def generate_boring_texts(
    lang: str,
    n_samples: int,
    oversample_ratio: float = 1.3,
) -> list[str]:
    """Generate plain, non-humorous texts using Gemini batch API.

    Each API call generates BATCH_SIZE items via structured JSON output,
    drastically reducing the total number of API calls needed.

    Args:
        lang: Language code ("en", "zh", "es").
        n_samples: Target number of boring texts.
        oversample_ratio: Request this many times more items to
            compensate for quality filtering loss.

    Returns:
        List of non-humorous text strings.
    """
    if lang not in _BORING_TEXT_BATCH_PROMPTS:
        raise ValueError(f"Unsupported language: {lang}")

    client = _init_gemini_client()
    prompt = _BORING_TEXT_BATCH_PROMPTS[lang]
    n_target = int(n_samples * oversample_ratio)
    n_batches = (n_target + BATCH_SIZE - 1) // BATCH_SIZE

    results: list[str] = []
    total_generated = 0
    total_filtered = 0

    print(f"\n  --- Generating boring texts for {lang} ---")
    print(f"  Target: {n_samples}, oversample target: {n_target}, "
          f"batch size: {BATCH_SIZE}, batches: {n_batches}")

    for batch_idx in range(n_batches):
        if len(results) >= n_samples:
            print(f"    Target reached after {batch_idx} batches")
            break

        raw_items = _call_gemini_batch(client, prompt)
        total_generated += len(raw_items)

        passed = [item for item in raw_items if _filter_boring_response(item)]
        batch_filtered = len(raw_items) - len(passed)
        total_filtered += batch_filtered
        results.extend(passed)

        print(f"    Batch {batch_idx + 1}/{n_batches}: "
              f"received {len(raw_items)}, passed {len(passed)}, "
              f"total so far: {len(results)}/{n_samples}")

        time.sleep(1.0)

    results = results[:n_samples]
    print(f"  Done: {len(results)} passed, "
          f"{total_generated} generated, {total_filtered} filtered")
    if len(results) < n_samples:
        print(
            f"  Warning: target not reached (target {n_samples}, "
            f"actual {len(results)}). Try increasing oversample_ratio."
        )

    return results


# ============================================================
# Preference Pair Assembly
# ============================================================

def assemble_preference_pairs(
    chosen_texts: list[str],
    rejected_texts: list[str],
    lang: str,
    seed: int = 42,
) -> list[dict]:
    """Pair chosen and rejected texts into TRL-compatible preference pairs.

    Each pair gets a random Type A prompt in the corresponding language.
    The number of pairs is min(len(chosen_texts), len(rejected_texts)).

    Args:
        chosen_texts: List of high-quality joke texts.
        rejected_texts: List of non-humorous texts (hard negatives).
        lang: Language code for prompt selection.
        seed: Random seed.

    Returns:
        List of preference pair dicts with keys: prompt, chosen, rejected.
    """
    rng = random.Random(seed)
    n_pairs = min(len(chosen_texts), len(rejected_texts))

    pairs = []
    for i in range(n_pairs):
        prompt_text = get_random_type_a_prompt(lang, rng)
        pairs.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "chosen": [{"role": "assistant", "content": chosen_texts[i]}],
            "rejected": [{"role": "assistant", "content": rejected_texts[i]}],
        })

    return pairs


# ============================================================
# Save
# ============================================================

def save_reward_pairs(pairs: list[dict], lang: str) -> Path:
    """Save synthesized reward preference pairs as JSONL file.

    Args:
        pairs: List of preference pair dicts.
        lang: Language code.

    Returns:
        Path to the saved file.
    """
    SYNTHESIZED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SYNTHESIZED_DIR / f"reward_neg_{lang}.jsonl"

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    return output_path


# ============================================================
# Synthesis Main Logic
# ============================================================

def synthesize_for_language(
    lang: str,
    n_samples: int,
    seed: int = 42,
) -> list[dict]:
    """Synthesize hard-negative preference pairs for one language.

    Args:
        lang: Language code ("en", "zh", "es").
        n_samples: Number of preference pairs to generate.
        seed: Random seed.

    Returns:
        List of preference pair dicts.
    """
    # 1. Load chosen (positive) samples
    if lang == "zh":
        chosen_texts = load_cfun_jokes(n_samples, seed=seed)
    else:
        chosen_texts = load_high_score_jokes(lang, n_samples, seed=seed)

    # 2. Generate rejected (negative) samples via Gemini
    rejected_texts = generate_boring_texts(lang, n_samples)

    # 3. Assemble preference pairs
    pairs = assemble_preference_pairs(chosen_texts, rejected_texts, lang, seed=seed)

    print(f"\n  Assembled {len(pairs)} preference pairs for {lang}")
    return pairs


# ============================================================
# Command Line Entry
# ============================================================

def main():
    """Command line entry point for reward data synthesis."""
    parser = argparse.ArgumentParser(
        description="Synthesize Hard-Negative Preference Pairs for Reward Model (Requires Gemini API)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        choices=["en", "zh", "es", "all"],
        help="Target language, or 'all' for all three languages",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        required=True,
        help="Number of preference pairs to generate per language",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default 42)",
    )

    args = parser.parse_args()

    languages = ["en", "zh", "es"] if args.lang == "all" else [args.lang]

    for lang in languages:
        print(f"\n{'=' * 60}")
        print(f"Synthesizing reward data: lang={lang}, n_samples={args.n_samples}")
        print(f"{'=' * 60}")

        pairs = synthesize_for_language(
            lang=lang,
            n_samples=args.n_samples,
            seed=args.seed,
        )

        output_path = save_reward_pairs(pairs, lang)
        print(f"\nSaved: {output_path} ({len(pairs)} pairs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
