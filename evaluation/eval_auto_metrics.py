"""
Automated Evaluation Metrics (Tier 1)
=====================================

Computes rule-based quality metrics on generated outputs from
generate_outputs.py, comparing Base / SFT / GRPO models side by side.

Metrics:
    - Format Compliance: non-empty, length in [10, 280], no trigram degeneracy
    - Keyword Satisfaction: all required keywords present (keyword subtask only)
    - Distinct-1 / Distinct-2: unigram / bigram diversity across all responses
    - Degeneracy Rate: fraction of responses with trigram uniqueness < 0.5
    - Length Statistics: mean, median, min, max response character length

Usage:
    python -m evaluation.eval_auto_metrics
    python -m evaluation.eval_auto_metrics --models base,grpo
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

TRIGRAM_UNIQUENESS_THRESHOLD = 0.5
FORMAT_MIN_LENGTH = 10
FORMAT_MAX_LENGTH = 280


# ============================================================
# Metric Functions
# ============================================================

def _is_format_compliant(text: str) -> bool:
    """Check if response passes format constraints."""
    text = text.strip()
    if not text or len(text) < FORMAT_MIN_LENGTH:
        return False
    if len(text) > FORMAT_MAX_LENGTH:
        return False
    words = text.split()
    if len(words) >= 4:
        trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
        if len(set(trigrams)) / len(trigrams) < TRIGRAM_UNIQUENESS_THRESHOLD:
            return False
    return True


def _is_degenerate(text: str) -> bool:
    """Check if response has excessive trigram repetition."""
    words = text.strip().split()
    if len(words) < 4:
        return False
    trigrams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    return len(set(trigrams)) / len(trigrams) < TRIGRAM_UNIQUENESS_THRESHOLD


def _keyword_satisfied(text: str, keywords: list[str]) -> bool:
    """Check if all required keywords appear in the response."""
    text_lower = text.lower()
    return all(kw.lower() in text_lower for kw in keywords)


def _distinct_n(responses: list[str], n: int) -> float:
    """Compute Distinct-N: unique n-grams / total n-grams across all responses."""
    total_ngrams = 0
    unique_ngrams: set[tuple[str, ...]] = set()
    for resp in responses:
        tokens = resp.strip().lower().split()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            unique_ngrams.add(ngram)
            total_ngrams += 1
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0.0


def compute_metrics(results: list[dict]) -> dict:
    """Compute all automated metrics for a list of generation results."""
    responses = [r["best_response"] for r in results]
    n = len(responses)
    if n == 0:
        return {}

    format_pass = sum(1 for r in responses if _is_format_compliant(r))
    degenerate_count = sum(1 for r in responses if _is_degenerate(r))

    kw_items = [r for r in results if r.get("keywords")]
    kw_total = len(kw_items)
    kw_pass = sum(
        1 for r in kw_items
        if _keyword_satisfied(r["best_response"], r["keywords"])
    )

    lengths = [len(r.strip()) for r in responses]

    metrics = {
        "n_samples": n,
        "format_compliance": round(format_pass / n, 4),
        "degeneracy_rate": round(degenerate_count / n, 4),
        "distinct_1": round(_distinct_n(responses, 1), 4),
        "distinct_2": round(_distinct_n(responses, 2), 4),
        "avg_length": round(statistics.mean(lengths), 1),
        "median_length": round(statistics.median(lengths), 1),
        "min_length": min(lengths),
        "max_length": max(lengths),
    }

    if kw_total > 0:
        metrics["keyword_satisfaction"] = round(kw_pass / kw_total, 4)
        metrics["keyword_total"] = kw_total
    else:
        metrics["keyword_satisfaction"] = None
        metrics["keyword_total"] = 0

    return metrics


def compute_per_lang_metrics(results: list[dict]) -> dict[str, dict]:
    """Compute metrics broken down by language."""
    by_lang: dict[str, list[dict]] = {}
    for r in results:
        lang = r.get("lang", "unknown")
        by_lang.setdefault(lang, []).append(r)

    return {lang: compute_metrics(items) for lang, items in sorted(by_lang.items())}


# ============================================================
# Display
# ============================================================

def print_comparison_table(all_metrics: dict[str, dict]):
    """Print a formatted comparison table across models."""
    models = list(all_metrics.keys())
    metric_keys = [
        "n_samples", "format_compliance", "degeneracy_rate",
        "distinct_1", "distinct_2", "keyword_satisfaction",
        "avg_length", "median_length",
    ]
    labels = {
        "n_samples": "Samples",
        "format_compliance": "Format Compliance",
        "degeneracy_rate": "Degeneracy Rate",
        "distinct_1": "Distinct-1",
        "distinct_2": "Distinct-2",
        "keyword_satisfaction": "Keyword Satisfaction",
        "avg_length": "Avg Length",
        "median_length": "Median Length",
    }

    header = f"{'Metric':<22}" + "".join(f"{m:>14}" for m in models)
    print(header)
    print("-" * len(header))

    for key in metric_keys:
        row = f"{labels[key]:<22}"
        for m in models:
            val = all_metrics[m].get(key)
            if val is None:
                row += f"{'N/A':>14}"
            elif isinstance(val, float) and val <= 1.0:
                row += f"{val:>13.1%} "
            elif isinstance(val, float):
                row += f"{val:>14.1f}"
            else:
                row += f"{val:>14}"
        print(row)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Compute automated evaluation metrics")
    parser.add_argument(
        "--models", type=str, default="base,sft,grpo",
        help="Comma-separated model names (must have corresponding output files)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(OUTPUT_DIR),
        help="Directory containing model output JSONL files",
    )
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]
    output_dir = Path(args.output_dir)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    all_per_lang: dict[str, dict] = {}

    for model_name in model_names:
        path = output_dir / f"{model_name}.jsonl"
        if not path.exists():
            print(f"Warning: {path} not found, skipping {model_name}")
            continue

        with open(path, encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]

        print(f"[{model_name}] Loaded {len(results)} results from {path}")

        all_metrics[model_name] = compute_metrics(results)
        all_per_lang[model_name] = compute_per_lang_metrics(results)

    if not all_metrics:
        print("No model outputs found. Run generate_outputs.py first.")
        return

    print(f"\n{'=' * 60}")
    print("Overall Metrics Comparison")
    print(f"{'=' * 60}\n")
    print_comparison_table(all_metrics)

    for lang in ["en", "zh", "es"]:
        lang_metrics = {}
        for m in all_metrics:
            if lang in all_per_lang.get(m, {}):
                lang_metrics[m] = all_per_lang[m][lang]
        if lang_metrics:
            print(f"\n--- Language: {lang} ---\n")
            print_comparison_table(lang_metrics)

    report = {
        "overall": all_metrics,
        "per_language": all_per_lang,
    }
    report_path = RESULTS_DIR / "auto_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()
