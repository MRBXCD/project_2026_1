"""
Compare SFT and Base lm-eval results and render readable markdown tables.

Usage:
    python -m utils.compare_sft_base
    python -m utils.compare_sft_base --top-k 15
    python -m utils.compare_sft_base --base-json /path/to/base.json --sft-json /path/to/sft.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_BASE_DIR = EVAL_DIR / "benchmark_base"
DEFAULT_SFT_DIR = EVAL_DIR / "benchmark_sft"
DEFAULT_OUTPUT = EVAL_DIR / "compare_sft_base.md"


@dataclass
class TaskComparison:
    task: str
    alias: str
    metric_name: str
    base: float
    sft: float

    @property
    def delta(self) -> float:
        return self.sft - self.base

    @property
    def winner(self) -> str:
        if self.delta > 1e-12:
            return "SFT"
        if self.delta < -1e-12:
            return "Base"
        return "Tie"

    @property
    def level(self) -> str:
        # lm-eval alias convention:
        # "mmlu" -> overall, " - xxx" -> group, "  - xxx" -> subtask
        if self.alias.startswith("  - "):
            return "subtask"
        if self.alias.startswith(" - "):
            return "group"
        return "overall"


def find_latest_result_json(result_dir: Path) -> Path:
    files = sorted(result_dir.glob("**/results_*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No result json found under: {result_dir}")
    return files[-1]


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_primary_metric(task_result: dict) -> tuple[str, float]:
    # Prefer acc, then exact_match, otherwise first non-stderr metric.
    preferred_prefixes = ("acc,", "exact_match,", "f1,", "bleu,", "rouge")
    candidates = [
        (k, v)
        for k, v in task_result.items()
        if isinstance(v, (int, float)) and "stderr" not in k
    ]
    if not candidates:
        raise ValueError("No numeric metric found in task result.")

    for prefix in preferred_prefixes:
        for k, v in candidates:
            if k.startswith(prefix):
                return k, float(v)
    return candidates[0][0], float(candidates[0][1])


def build_comparisons(base_data: dict, sft_data: dict) -> list[TaskComparison]:
    base_results = base_data.get("results", {})
    sft_results = sft_data.get("results", {})
    common_tasks = sorted(set(base_results) & set(sft_results))
    if not common_tasks:
        raise ValueError("No common tasks between base and sft results.")

    comparisons: list[TaskComparison] = []
    for task in common_tasks:
        base_task = base_results[task]
        sft_task = sft_results[task]

        metric_name, base_metric = pick_primary_metric(base_task)
        sft_metric = sft_task.get(metric_name)
        if not isinstance(sft_metric, (int, float)):
            # Fall back if metric key naming differs unexpectedly.
            _, sft_metric = pick_primary_metric(sft_task)

        alias = str(base_task.get("alias", task))
        comparisons.append(
            TaskComparison(
                task=task,
                alias=alias,
                metric_name=metric_name,
                base=float(base_metric),
                sft=float(sft_metric),
            )
        )
    return comparisons


def fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def fmt_pp(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}pp"


def build_table(rows: Iterable[TaskComparison]) -> str:
    header = (
        "| Task | Metric | Base | SFT | Delta (SFT-Base) | Better |\n"
        "|---|---|---:|---:|---:|---|\n"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            f"| {r.alias} | `{r.metric_name}` | {fmt_pct(r.base)} | {fmt_pct(r.sft)} | {fmt_pp(r.delta)} | {r.winner} |"
        )
    return header + "\n".join(body_lines) + ("\n" if body_lines else "")


def build_summary(comparisons: list[TaskComparison]) -> str:
    deltas = [c.delta for c in comparisons]
    improved = sum(1 for d in deltas if d > 1e-12)
    degraded = sum(1 for d in deltas if d < -1e-12)
    tied = len(deltas) - improved - degraded
    avg_delta = sum(deltas) / len(deltas)
    best = max(comparisons, key=lambda c: c.delta)
    worst = min(comparisons, key=lambda c: c.delta)
    return "\n".join(
        [
            "## Overall Summary",
            "",
            f"- Compared tasks: **{len(comparisons)}**",
            f"- SFT better on: **{improved}** tasks",
            f"- Base better on: **{degraded}** tasks",
            f"- Tie: **{tied}** tasks",
            f"- Mean delta (SFT-Base): **{fmt_pp(avg_delta)}**",
            f"- Best gain: **{best.alias} ({fmt_pp(best.delta)})**",
            f"- Largest drop: **{worst.alias} ({fmt_pp(worst.delta)})**",
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare base and sft lm-eval JSON results and render markdown table."
    )
    parser.add_argument("--base-json", type=str, default=None, help="Path to base result JSON.")
    parser.add_argument("--sft-json", type=str, default=None, help="Path to sft result JSON.")
    parser.add_argument("--base-dir", type=str, default=str(DEFAULT_BASE_DIR), help="Base result directory.")
    parser.add_argument("--sft-dir", type=str, default=str(DEFAULT_SFT_DIR), help="SFT result directory.")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output markdown file path.")
    parser.add_argument("--top-k", type=int, default=10, help="Top improved/regressed subtasks to show.")
    args = parser.parse_args()

    base_json = Path(args.base_json) if args.base_json else find_latest_result_json(Path(args.base_dir))
    sft_json = Path(args.sft_json) if args.sft_json else find_latest_result_json(Path(args.sft_dir))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_data = load_json(base_json)
    sft_data = load_json(sft_json)
    comparisons = build_comparisons(base_data, sft_data)

    overall_rows = [c for c in comparisons if c.level == "overall"]
    group_rows = [c for c in comparisons if c.level == "group"]
    subtask_rows = [c for c in comparisons if c.level == "subtask"]
    top_k = max(1, args.top_k)
    top_improved = sorted(subtask_rows, key=lambda c: c.delta, reverse=True)[:top_k]
    top_regressed = sorted(subtask_rows, key=lambda c: c.delta)[:top_k]

    markdown = "\n".join(
        [
            "# SFT vs Base Evaluation Comparison",
            "",
            f"- Base JSON: `{base_json}`",
            f"- SFT JSON: `{sft_json}`",
            "",
            build_summary(comparisons),
            "## Overall Tasks",
            "",
            build_table(overall_rows),
            "## Task Groups",
            "",
            build_table(group_rows),
            f"## Top {top_k} Improved Subtasks (SFT > Base)",
            "",
            build_table(top_improved),
            f"## Top {top_k} Regressed Subtasks (Base > SFT)",
            "",
            build_table(top_regressed),
        ]
    )

    output_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nSaved comparison markdown to: {output_path}")


if __name__ == "__main__":
    main()
