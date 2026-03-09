"""
Evaluation Report Generator
============================

Aggregates results from automated metrics (Tier 1) and LLM-as-Judge
(Tier 2) into a single Markdown report.

Usage:
    python -m evaluation.generate_report
"""

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def _fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val:.1%}"


def _fmt_num(val):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def build_report(auto_metrics: dict | None, judge_results: list | None) -> str:
    lines = []
    lines.append("# Humor Generation — Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")

    # ---- Tier 1: Automated Metrics ----
    lines.append("## Tier 1: Automated Metrics")
    lines.append("")

    if auto_metrics and "overall" in auto_metrics:
        overall = auto_metrics["overall"]
        models = list(overall.keys())

        metric_rows = [
            ("Samples", "n_samples", False),
            ("Format Compliance", "format_compliance", True),
            ("Degeneracy Rate", "degeneracy_rate", True),
            ("Distinct-1", "distinct_1", True),
            ("Distinct-2", "distinct_2", True),
            ("Keyword Satisfaction", "keyword_satisfaction", True),
            ("Avg Length", "avg_length", False),
            ("Median Length", "median_length", False),
        ]

        header = "| Metric |" + " | ".join(f" {m} " for m in models) + " |"
        sep = "|---|" + " | ".join("---" for _ in models) + " |"
        lines.append(header)
        lines.append(sep)

        for label, key, is_pct in metric_rows:
            row = f"| {label} |"
            for m in models:
                val = overall[m].get(key)
                row += f" {_fmt_pct(val) if is_pct else _fmt_num(val)} |"
            lines.append(row)
        lines.append("")

        # Per-language breakdown
        per_lang = auto_metrics.get("per_language", {})
        for lang in ["en", "zh", "es"]:
            lang_data = {}
            for m in models:
                if lang in per_lang.get(m, {}):
                    lang_data[m] = per_lang[m][lang]

            if not lang_data:
                continue

            lines.append(f"### Language: {lang}")
            lines.append("")
            lm = list(lang_data.keys())
            header = "| Metric |" + " | ".join(f" {m} " for m in lm) + " |"
            sep = "|---|" + " | ".join("---" for _ in lm) + " |"
            lines.append(header)
            lines.append(sep)
            for label, key, is_pct in metric_rows:
                row = f"| {label} |"
                for m in lm:
                    val = lang_data[m].get(key)
                    row += f" {_fmt_pct(val) if is_pct else _fmt_num(val)} |"
                lines.append(row)
            lines.append("")
    else:
        lines.append("*Auto metrics not available. Run eval_auto_metrics.py first.*")
        lines.append("")

    # ---- Tier 2: LLM-as-Judge ----
    lines.append("## Tier 2: LLM-as-Judge Pairwise Comparison")
    lines.append("")

    if judge_results:
        lines.append("| Comparison | Win A | Win B | Tie | Consistency |")
        lines.append("|---|---|---|---|---|")
        for s in judge_results:
            ma = s["model_a"]
            mb = s["model_b"]
            wa = s.get(f"win_rate_{ma}", 0)
            wb = s.get(f"win_rate_{mb}", 0)
            tie = s.get("tie_rate", 0)
            cons = s.get("consistency_rate", 0)
            lines.append(
                f"| {ma} vs {mb} | "
                f"{ma}: {wa:.1%} | "
                f"{mb}: {wb:.1%} | "
                f"{tie:.1%} | "
                f"{cons:.1%} |"
            )
        lines.append("")

        for s in judge_results:
            ma = s["model_a"]
            mb = s["model_b"]
            n = s.get("n_comparisons", 0)
            wa_n = s.get(f"win_{ma}", 0)
            wb_n = s.get(f"win_{mb}", 0)
            tie_n = s.get("tie", 0)
            lines.append(f"**{ma} vs {mb}** ({n} pairs): "
                         f"{ma} wins {wa_n}, {mb} wins {wb_n}, tie {tie_n}")
            lines.append("")
    else:
        lines.append("*LLM judge results not available. Run eval_llm_judge.py first.*")
        lines.append("")

    # ---- Tier 3: Human Evaluation ----
    lines.append("## Tier 3: Human Evaluation")
    lines.append("")
    lines.append("*To be filled in after human evaluation is completed.*")
    lines.append("")
    lines.append("| Evaluator | Model A Wins | Model B Wins | Ties |")
    lines.append("|---|---|---|---|")
    lines.append("| Evaluator 1 | | | |")
    lines.append("| Evaluator 2 | | | |")
    lines.append("| Agreement (Cohen's kappa) | | | |")
    lines.append("")

    # ---- Notes ----
    lines.append("## Notes")
    lines.append("")
    lines.append("- Position bias mitigation: each pairwise comparison is run twice "
                 "with A/B order swapped; only consistent verdicts are counted as wins.")
    lines.append("- Keyword Satisfaction only applies to prompts with keyword constraints.")
    lines.append("- Distinct-N is computed across all responses in the evaluation set.")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    auto_path = results_dir / "auto_metrics.json"
    auto_metrics = None
    if auto_path.exists():
        with open(auto_path, encoding="utf-8") as f:
            auto_metrics = json.load(f)

    judge_path = results_dir / "llm_judge.json"
    judge_results = None
    if judge_path.exists():
        with open(judge_path, encoding="utf-8") as f:
            judge_results = json.load(f)

    report = build_report(auto_metrics, judge_results)

    report_path = results_dir / "evaluation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
