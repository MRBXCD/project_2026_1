"""
LLM-as-Judge Pairwise Evaluation (Tier 2)
==========================================

Compares model outputs pairwise using Gemini as judge: for each eval
prompt, the judge sees two responses (A and B) and decides which is
funnier, or declares a TIE.

To mitigate position bias, each pair is judged TWICE with A/B swapped.
Only consistent verdicts count; inconsistent pairs are recorded as TIE.

Comparison pairs:
    - Base vs GRPO   (primary result)
    - SFT  vs GRPO   (GRPO-stage incremental gain)
    - Base vs SFT    (SFT-stage incremental gain)

Usage:
    python -m evaluation.eval_llm_judge
    python -m evaluation.eval_llm_judge --pairs "base:grpo,sft:grpo"

Dependencies:
    - google-genai (Google Gemini API)
    - GEMINI_API_KEY environment variable
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

JUDGE_MODEL = "gemini-3-flash-preview"
API_CALL_DELAY = 0.2
MAX_RETRIES = 3


# ============================================================
# Judge Prompt
# ============================================================

JUDGE_PROMPT = """You are an expert judge of humor quality.

Given the following news headline and two joke responses (A and B), determine which joke is funnier.

Headline: "{headline}"

Response A: "{response_a}"
Response B: "{response_b}"

Consider:
1. Is the joke genuinely funny (not just random or nonsensical)?
2. Does it relate to the headline?
3. Is it a single, well-formed sentence?

Which response is funnier? Reply with ONLY "A" or "B" or "TIE"."""


# ============================================================
# Gemini API
# ============================================================

def _init_gemini_client():
    """Initialize Google Gemini API Client."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Please run: export GEMINI_API_KEY='your-api-key'"
        )
    from google import genai
    return genai.Client(api_key=api_key)


def _call_gemini(client, prompt: str) -> str | None:
    """Call Gemini API with retry logic."""
    from google.genai import types

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=JUDGE_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.1,
                    max_output_tokens=16,
                ),
            )
            text = response.text
            return text.strip() if text else None
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "resource" in error_msg or "quota" in error_msg:
                wait = 2 ** attempt * 5
                print(f"    Rate limit, waiting {wait}s ({attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            print(f"    API error ({attempt}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)
    return None


def _parse_verdict(text: str | None) -> str:
    """Parse judge response into 'A', 'B', or 'TIE'."""
    if text is None:
        return "TIE"
    text = text.strip().upper()
    if text in ("A", "B", "TIE"):
        return text
    if "TIE" in text:
        return "TIE"
    match = re.search(r"\b([AB])\b", text)
    return match.group(1) if match else "TIE"


# ============================================================
# Pairwise Judging
# ============================================================

def judge_pair(
    client,
    headline: str,
    response_model_a: str,
    response_model_b: str,
) -> dict:
    """Judge one pair with position-bias mitigation (two passes, A/B swapped).

    Returns dict with:
        verdict: 'model_a' | 'model_b' | 'tie'
        pass1: raw verdict of first pass (model_a=A, model_b=B)
        pass2: raw verdict of second pass (model_a=B, model_b=A)
        consistent: whether both passes agree
    """
    prompt_1 = JUDGE_PROMPT.format(
        headline=headline,
        response_a=response_model_a,
        response_b=response_model_b,
    )
    time.sleep(API_CALL_DELAY)
    raw_1 = _call_gemini(client, prompt_1)
    verdict_1 = _parse_verdict(raw_1)

    prompt_2 = JUDGE_PROMPT.format(
        headline=headline,
        response_a=response_model_b,
        response_b=response_model_a,
    )
    time.sleep(API_CALL_DELAY)
    raw_2 = _call_gemini(client, prompt_2)
    verdict_2 = _parse_verdict(raw_2)

    # Normalize: translate verdicts into "who won" from model_a's perspective
    # Pass 1: A=model_a, B=model_b → verdict_1="A" means model_a wins
    # Pass 2: A=model_b, B=model_a → verdict_2="B" means model_a wins
    model_a_wins_1 = verdict_1 == "A"
    model_b_wins_1 = verdict_1 == "B"
    tie_1 = verdict_1 == "TIE"

    model_a_wins_2 = verdict_2 == "B"
    model_b_wins_2 = verdict_2 == "A"
    tie_2 = verdict_2 == "TIE"

    if model_a_wins_1 and model_a_wins_2:
        final = "model_a"
        consistent = True
    elif model_b_wins_1 and model_b_wins_2:
        final = "model_b"
        consistent = True
    elif tie_1 and tie_2:
        final = "tie"
        consistent = True
    else:
        final = "tie"
        consistent = False

    return {
        "verdict": final,
        "pass1_raw": verdict_1,
        "pass2_raw": verdict_2,
        "consistent": consistent,
    }


def run_pairwise_comparison(
    client,
    results_a: list[dict],
    results_b: list[dict],
    model_a_name: str,
    model_b_name: str,
) -> dict:
    """Run pairwise comparison between two models on all prompts.

    Returns summary dict with win rates and detailed per-prompt results.
    """
    n = min(len(results_a), len(results_b))
    wins_a, wins_b, ties = 0, 0, 0
    consistent_count = 0
    details = []

    for i in range(n):
        ra = results_a[i]
        rb = results_b[i]

        headline = ra.get("headline", rb.get("headline", ""))
        resp_a = ra["best_response"]
        resp_b = rb["best_response"]

        result = judge_pair(client, headline, resp_a, resp_b)

        if result["verdict"] == "model_a":
            wins_a += 1
        elif result["verdict"] == "model_b":
            wins_b += 1
        else:
            ties += 1

        if result["consistent"]:
            consistent_count += 1

        details.append({
            "index": i,
            "headline": headline,
            "lang": ra.get("lang", ""),
            f"response_{model_a_name}": resp_a,
            f"response_{model_b_name}": resp_b,
            "verdict": result["verdict"].replace("model_a", model_a_name).replace("model_b", model_b_name),
            "pass1_raw": result["pass1_raw"],
            "pass2_raw": result["pass2_raw"],
            "consistent": result["consistent"],
        })

        status_char = {"model_a": model_a_name[0].upper(), "model_b": model_b_name[0].upper(), "tie": "="}
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f"    [{i + 1}/{n}] {model_a_name}={wins_a} {model_b_name}={wins_b} tie={ties}")

    summary = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n_comparisons": n,
        f"win_{model_a_name}": wins_a,
        f"win_{model_b_name}": wins_b,
        "tie": ties,
        f"win_rate_{model_a_name}": round(wins_a / n, 4) if n else 0,
        f"win_rate_{model_b_name}": round(wins_b / n, 4) if n else 0,
        "tie_rate": round(ties / n, 4) if n else 0,
        "consistency_rate": round(consistent_count / n, 4) if n else 0,
    }

    return {"summary": summary, "details": details}


# ============================================================
# Display
# ============================================================

def print_summary(all_summaries: list[dict]):
    """Print a formatted summary table of all pairwise comparisons."""
    print(f"\n{'Comparison':<22} {'Win A':>10} {'Win B':>10} {'Tie':>10} {'Consist.':>10}")
    print("-" * 62)
    for s in all_summaries:
        ma = s["model_a"]
        mb = s["model_b"]
        label = f"{ma} vs {mb}"
        wa = s[f"win_rate_{ma}"]
        wb = s[f"win_rate_{mb}"]
        t = s["tie_rate"]
        c = s["consistency_rate"]
        print(f"{label:<22} {wa:>9.1%} {wb:>9.1%} {t:>9.1%} {c:>9.1%}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge pairwise evaluation")
    parser.add_argument(
        "--pairs", type=str, default="base:grpo,sft:grpo,base:sft",
        help="Comma-separated model pairs to compare, e.g. 'base:grpo,sft:grpo'",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    client = _init_gemini_client()

    pairs = [p.strip().split(":") for p in args.pairs.split(",")]

    model_cache: dict[str, list[dict]] = {}
    for ma, mb in pairs:
        for m in (ma, mb):
            if m not in model_cache:
                path = output_dir / f"{m}.jsonl"
                if not path.exists():
                    print(f"Warning: {path} not found, skipping pairs involving {m}")
                    model_cache[m] = []
                    continue
                with open(path, encoding="utf-8") as f:
                    model_cache[m] = [json.loads(line) for line in f if line.strip()]
                print(f"Loaded {len(model_cache[m])} results for {m}")

    all_summaries = []
    all_details = []

    for model_a, model_b in pairs:
        ra = model_cache.get(model_a, [])
        rb = model_cache.get(model_b, [])
        if not ra or not rb:
            print(f"\nSkipping {model_a} vs {model_b} (missing data)")
            continue

        print(f"\n{'=' * 60}")
        print(f"Judging: {model_a} vs {model_b}")
        print(f"{'=' * 60}")

        result = run_pairwise_comparison(client, ra, rb, model_a, model_b)
        all_summaries.append(result["summary"])
        all_details.extend(result["details"])

    if all_summaries:
        print(f"\n{'=' * 60}")
        print("LLM Judge Summary")
        print(f"{'=' * 60}")
        print_summary(all_summaries)

        summary_path = RESULTS_DIR / "llm_judge.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to {summary_path}")

        details_path = RESULTS_DIR / "llm_judge_details.jsonl"
        with open(details_path, "w", encoding="utf-8") as f:
            for d in all_details:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Details saved to {details_path}")


if __name__ == "__main__":
    main()
