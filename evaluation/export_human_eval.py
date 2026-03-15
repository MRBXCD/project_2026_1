"""
Human Evaluation Export (Tier 3)
================================

Randomly samples prompts from model outputs and exports them as a blind
A/B comparison table for human evaluators. The source model for each
response is hidden; A/B assignment is randomized per row.

Output: CSV file with columns:
    id, headline, lang, response_a, response_b, your_verdict

Evaluators fill in the ``your_verdict`` column with A, B, or TIE.

Usage (standalone):
    python -m evaluation.export_human_eval
    python -m evaluation.export_human_eval --n_samples 40 --pairs "base:grpo"

Usage (via pipeline):
    python -m evaluation.pipeline --steps human_eval
"""

import argparse
import csv
import json
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def export_blind_table(
    results_a: list[dict],
    results_b: list[dict],
    model_a: str,
    model_b: str,
    n_samples: int = 36,
    seed: int = 42,
) -> list[dict]:
    """Create a blind A/B comparison table.

    Randomly samples n_samples prompts, stratified by language (equal per
    language when possible). For each row, randomly assigns one model's
    response to column A and the other to column B.

    Returns list of row dicts and a separate key mapping row_id → which
    model was assigned to A.
    """
    rng = random.Random(seed)
    n = min(len(results_a), len(results_b))

    by_lang: dict[str, list[int]] = {}
    for i in range(n):
        lang = results_a[i].get("lang", "unknown")
        by_lang.setdefault(lang, []).append(i)

    per_lang = max(1, n_samples // len(by_lang)) if by_lang else n_samples
    selected_indices = []
    for lang in sorted(by_lang):
        indices = by_lang[lang][:]
        rng.shuffle(indices)
        selected_indices.extend(indices[:per_lang])

    rng.shuffle(selected_indices)
    selected_indices = selected_indices[:n_samples]

    rows = []
    answer_key = []

    for row_id, idx in enumerate(selected_indices, 1):
        ra = results_a[idx]
        rb = results_b[idx]
        resp_a_model = ra["best_response"]
        resp_b_model = rb["best_response"]

        if rng.random() < 0.5:
            col_a, col_b = resp_a_model, resp_b_model
            a_source = model_a
        else:
            col_a, col_b = resp_b_model, resp_a_model
            a_source = model_b

        rows.append({
            "id": row_id,
            "headline": ra.get("headline", ""),
            "lang": ra.get("lang", ""),
            "response_a": col_a,
            "response_b": col_b,
            "your_verdict": "",
        })
        answer_key.append({
            "id": row_id,
            "response_a_source": a_source,
            "response_b_source": model_b if a_source == model_a else model_a,
        })

    return rows, answer_key


# ============================================================
# Programmatic Entry Point (for pipeline)
# ============================================================

def run(
    pair: tuple[str, str] | None = None,
    n_samples: int = 36,
    seed: int = 42,
    output_dir: str | Path | None = None,
    results_dir: str | Path | None = None,
) -> dict | None:
    """Programmatic entry point for the human_eval export step.

    Returns:
        Dict with "n_exported", "csv_path", "key_path" on success, None on failure.
    """
    if pair is None:
        pair = ("base", "grpo")
    model_a, model_b = pair
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR
    res_dir = Path(results_dir) if results_dir else RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)

    path_a = out_dir / f"{model_a}.jsonl"
    path_b = out_dir / f"{model_b}.jsonl"

    for p, m in [(path_a, model_a), (path_b, model_b)]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found. Run generate step for {m} first.")

    with open(path_a, encoding="utf-8") as f:
        results_a = [json.loads(line) for line in f if line.strip()]
    with open(path_b, encoding="utf-8") as f:
        results_b = [json.loads(line) for line in f if line.strip()]

    rows, answer_key = export_blind_table(
        results_a, results_b, model_a, model_b,
        n_samples=n_samples, seed=seed,
    )

    csv_path = res_dir / "human_eval_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "headline", "lang", "response_a", "response_b", "your_verdict"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} blind A/B samples to {csv_path}")

    key_path = res_dir / "human_eval_answer_key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump(answer_key, f, ensure_ascii=False, indent=2)
    print(f"Answer key saved to {key_path} (DO NOT share with evaluators)")

    return {
        "n_exported": len(rows),
        "csv_path": str(csv_path),
        "key_path": str(key_path),
    }


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export blind A/B table for human evaluation")
    parser.add_argument(
        "--pairs", type=str, default="base:grpo",
        help="Model pair for human eval (default: base:grpo)",
    )
    parser.add_argument("--n_samples", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pair_parts = args.pairs.strip().split(":")
    if len(pair_parts) != 2:
        raise ValueError("--pairs must be exactly two models separated by ':'")

    run(pair=tuple(pair_parts), n_samples=args.n_samples, seed=args.seed)


if __name__ == "__main__":
    main()
