"""
Evaluation Report Generator
============================

Aggregates all available evaluation results into a single Markdown report:
    - Section 1: General Capability Benchmark (lm-eval comparison)
    - Section 2: Task-Specific Auto Metrics (format, keywords, diversity)
    - Section 3: LLM-as-Judge Pairwise Comparison
    - Section 4: Human Evaluation (pending or completed)
    - Appendix: run metadata

Each section is rendered only if its results file exists in the results
directory. Missing sections show a placeholder message. This enables
incremental report generation -- run ``--steps report`` again after
completing human evaluation to update the report.

Usage (standalone):
    python -m evaluation.generate_report
    python -m evaluation.generate_report --results_dir evaluation/results

Usage (via pipeline):
    python -m evaluation.pipeline --steps report
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


# ============================================================
# Formatting Helpers
# ============================================================

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


# ============================================================
# Section 1: Benchmark Comparison
# ============================================================

def _build_benchmark_section(results_dir: Path) -> str:
    lines = ["## 1. General Capability Benchmark", ""]

    benchmark_json = results_dir / "benchmark_comparison.json"
    benchmark_md = results_dir / "benchmark_comparison.md"

    if benchmark_md.exists():
        content = benchmark_md.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("# "):
                continue
            lines.append(line)
        lines.append("")
        return "\n".join(lines)

    if benchmark_json.exists():
        with open(benchmark_json, encoding="utf-8") as f:
            data = json.load(f)

        models = data.get("models", [])
        tasks = data.get("tasks", [])
        if not models or not tasks:
            lines.append("*Benchmark data is empty.*")
            lines.append("")
            return "\n".join(lines)

        overall = [t for t in tasks if t.get("level") == "overall"]
        groups = [t for t in tasks if t.get("level") == "group"]

        if overall or groups:
            display = overall + groups
            header = "| Task | Metric |" + " | ".join(f" {m} " for m in models) + " |"
            sep = "|---|---|" + " | ".join("---:" for _ in models) + " |"
            lines.append(header)
            lines.append(sep)
            for t in display:
                scores = t.get("scores", {})
                row = f"| {t['alias']} | `{t['metric_name']}` |"
                for m in models:
                    val = scores.get(m)
                    row += f" {val * 100:.2f}% |" if val is not None else " N/A |"
                lines.append(row)
            lines.append("")

        return "\n".join(lines)

    lines.append("*Benchmark results not available. Run the benchmark step first.*")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# Section 2: Auto Metrics
# ============================================================

def _build_auto_metrics_section(results_dir: Path) -> str:
    lines = ["## 2. Task-Specific Automated Metrics", ""]

    auto_path = results_dir / "auto_metrics.json"
    if not auto_path.exists():
        lines.append("*Auto metrics not available. Run the auto_metrics step first.*")
        lines.append("")
        return "\n".join(lines)

    with open(auto_path, encoding="utf-8") as f:
        auto_metrics = json.load(f)

    overall = auto_metrics.get("overall", {})
    if not overall:
        lines.append("*Auto metrics data is empty.*")
        lines.append("")
        return "\n".join(lines)

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
    sep = "|---|" + " | ".join("---:" for _ in models) + " |"
    lines.append(header)
    lines.append(sep)
    for label, key, is_pct in metric_rows:
        row = f"| {label} |"
        for m in models:
            val = overall[m].get(key)
            row += f" {_fmt_pct(val) if is_pct else _fmt_num(val)} |"
        lines.append(row)
    lines.append("")

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
        sep = "|---|" + " | ".join("---:" for _ in lm) + " |"
        lines.append(header)
        lines.append(sep)
        for label, key, is_pct in metric_rows:
            row = f"| {label} |"
            for m in lm:
                val = lang_data[m].get(key)
                row += f" {_fmt_pct(val) if is_pct else _fmt_num(val)} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Section 3: LLM-as-Judge
# ============================================================

def _build_llm_judge_section(results_dir: Path) -> str:
    lines = ["## 3. LLM-as-Judge Pairwise Comparison", ""]

    judge_path = results_dir / "llm_judge.json"
    if not judge_path.exists():
        lines.append("*LLM judge results not available. Run the llm_judge step first.*")
        lines.append("")
        return "\n".join(lines)

    with open(judge_path, encoding="utf-8") as f:
        judge_results = json.load(f)

    if not judge_results:
        lines.append("*LLM judge data is empty.*")
        lines.append("")
        return "\n".join(lines)

    lines.append("| Comparison | Win A | Win B | Tie | Consistency |")
    lines.append("|---|---:|---:|---:|---:|")
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
        lines.append(
            f"**{ma} vs {mb}** ({n} pairs): "
            f"{ma} wins {wa_n}, {mb} wins {wb_n}, tie {tie_n}"
        )
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Section 4: Human Evaluation
# ============================================================

def _build_human_eval_section(results_dir: Path) -> str:
    lines = ["## 4. Human Evaluation", ""]

    csv_path = results_dir / "human_eval_samples.csv"
    key_path = results_dir / "human_eval_answer_key.json"

    if not csv_path.exists():
        lines.append("*Human evaluation samples not exported yet. "
                      "Run the human_eval step first.*")
        lines.append("")
        return "\n".join(lines)

    filled_rows = []
    total_rows = 0
    try:
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_rows += 1
                verdict = row.get("your_verdict", "").strip().upper()
                if verdict in ("A", "B", "TIE"):
                    filled_rows.append(verdict)
    except Exception:
        lines.append("*Error reading human evaluation CSV.*")
        lines.append("")
        return "\n".join(lines)

    if not filled_rows:
        lines.append(f"Blind A/B samples exported ({total_rows} pairs). "
                      "**Pending**: awaiting evaluator responses.")
        lines.append("")
        lines.append(f"- CSV file: `{csv_path}`")
        lines.append("- Fill in the `your_verdict` column with A, B, or TIE")
        lines.append("- Then re-run: `python -m evaluation.pipeline --steps report`")
        lines.append("")
        return "\n".join(lines)

    if not key_path.exists():
        lines.append("*Answer key not found. Cannot decode blind verdicts.*")
        lines.append("")
        return "\n".join(lines)

    with open(key_path, encoding="utf-8") as f:
        answer_key = json.load(f)

    key_map = {entry["id"]: entry for entry in answer_key}
    model_wins: Counter = Counter()

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            verdict = row.get("your_verdict", "").strip().upper()
            row_id = int(row.get("id", 0))
            key_entry = key_map.get(row_id)
            if not key_entry or verdict not in ("A", "B", "TIE"):
                continue

            if verdict == "A":
                model_wins[key_entry["response_a_source"]] += 1
            elif verdict == "B":
                model_wins[key_entry["response_b_source"]] += 1
            else:
                model_wins["tie"] += 1

    n_filled = len(filled_rows)
    lines.append(f"Completed: {n_filled}/{total_rows} pairs evaluated.")
    lines.append("")

    if model_wins:
        lines.append("| Model/Outcome | Count | Rate |")
        lines.append("|---|---:|---:|")
        for label, count in model_wins.most_common():
            rate = count / n_filled if n_filled else 0
            lines.append(f"| {label} | {count} | {rate:.1%} |")
        lines.append("")

    return "\n".join(lines)


# ============================================================
# Appendix
# ============================================================

def _build_appendix() -> str:
    lines = [
        "## Appendix",
        "",
        f"- Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "- Position bias mitigation: each pairwise comparison is run twice "
        "with A/B order swapped; only consistent verdicts are counted as wins.",
        "- Keyword Satisfaction only applies to prompts with keyword constraints.",
        "- Distinct-N is computed across all responses in the evaluation set.",
        "",
    ]
    return "\n".join(lines)


# ============================================================
# Build Full Report
# ============================================================

def build_report(results_dir: Path) -> str:
    """Build the full evaluation report by scanning available results."""
    sections = [
        "# Humor Generation — Evaluation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        _build_benchmark_section(results_dir),
        "---",
        "",
        _build_auto_metrics_section(results_dir),
        "---",
        "",
        _build_llm_judge_section(results_dir),
        "---",
        "",
        _build_human_eval_section(results_dir),
        "---",
        "",
        _build_appendix(),
    ]
    return "\n".join(sections)


# ============================================================
# Programmatic Entry Point (for pipeline)
# ============================================================

def run(results_dir: str | Path | None = None) -> str:
    """Programmatic entry point for the report step.

    Returns:
        The generated report markdown string.
    """
    res_dir = Path(results_dir) if results_dir else RESULTS_DIR
    res_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(res_dir)

    report_path = res_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    return report


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--results_dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    report = run(results_dir=args.results_dir)
    print()
    print(report)


if __name__ == "__main__":
    main()
