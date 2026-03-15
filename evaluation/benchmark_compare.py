"""
Benchmark Comparison using lm-evaluation-harness
=================================================

Runs lm-eval benchmarks for multiple models (base / SFT / GRPO) and
compares results side by side. Supports running benchmarks, comparing
existing results, or both.

This module replaces:
    - sft/eval_sft.py (benchmark mode)
    - utils/compare_sft_base.py

Usage:
    python -m evaluation.benchmark_compare
    python -m evaluation.benchmark_compare --mode compare
    python -m evaluation.benchmark_compare --models base,sft
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = EVAL_DIR / "results"

DEFAULT_BENCHMARK_DIRS = {
    "base": EVAL_DIR / "benchmark_base",
    "sft": EVAL_DIR / "benchmark_sft",
    "grpo": EVAL_DIR / "benchmark_grpo",
}


# ============================================================
# Data Structures
# ============================================================

@dataclass
class TaskResult:
    """Benchmark result for a single task across multiple models."""

    task: str
    alias: str
    metric_name: str
    scores: dict[str, float] = field(default_factory=dict)

    @property
    def level(self) -> str:
        if self.alias.startswith("  - "):
            return "subtask"
        if self.alias.startswith(" - "):
            return "group"
        return "overall"


# ============================================================
# Running lm-eval
# ============================================================

def invoke_lm_eval(
    model_args: str,
    tasks: str,
    output_dir: str,
    num_fewshot: int = 5,
    batch_size: str = "4",
    device: str = "cuda:0",
    limit: str | None = None,
):
    """Run lm-evaluation-harness CLI."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", tasks,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--device", device,
        "--output_path", output_dir,
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    print(f"  Command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _prepare_merged_model(
    base_model: str,
    sft_repo: str,
    save_dir: Path,
) -> Path:
    """Merge SFT adapter into base model and save to disk for lm-eval."""
    if save_dir.exists():
        print(f"  Merged SFT model already exists at {save_dir}, skipping merge")
        return save_dir

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Merging SFT adapter into base model -> {save_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, sft_repo)
    model = model.merge_and_unload()

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    del model, tokenizer
    gc.collect()
    print("  Merge complete")
    return save_dir


def _prepare_grpo_merged_model(
    base_model: str,
    sft_repo: str,
    grpo_repo: str,
    save_dir: Path,
) -> Path:
    """Merge SFT + GRPO adapters into base model and save to disk."""
    if save_dir.exists():
        print(f"  Merged GRPO model already exists at {save_dir}, skipping merge")
        return save_dir

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Merging SFT+GRPO adapters into base model -> {save_dir}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model = PeftModel.from_pretrained(model, sft_repo)
    model = model.merge_and_unload()
    model = PeftModel.from_pretrained(model, grpo_repo)
    model = model.merge_and_unload()

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    del model, tokenizer
    gc.collect()
    print("  Merge complete")
    return save_dir


def run_benchmarks(
    models: list[str],
    base_model: str = "Qwen/Qwen3-8B",
    sft_repo: str = "MRBSTUDIO/Humor-Qwen3-8B-SFT",
    grpo_repo: str = "MRBSTUDIO/Humor-Qwen3-8B-GRPO",
    tasks: str = "mmlu",
    num_fewshot: int = 5,
    batch_size: str = "4",
    device: str = "cuda:0",
    limit: str | None = None,
    sft_eval_mode: str = "peft",
):
    """Run lm-eval benchmarks for the specified models."""
    for model_name in models:
        output_dir = str(DEFAULT_BENCHMARK_DIRS.get(
            model_name, EVAL_DIR / f"benchmark_{model_name}"
        ))

        print(f"\n{'=' * 60}")
        print(f"Running benchmark for: {model_name}")
        print(f"{'=' * 60}")

        if model_name == "base":
            model_args = (
                f"pretrained={base_model},"
                "dtype=bfloat16,trust_remote_code=True"
            )
        elif model_name == "sft":
            if sft_eval_mode == "peft":
                model_args = (
                    f"pretrained={base_model},peft={sft_repo},"
                    "dtype=bfloat16,trust_remote_code=True"
                )
            else:
                merged_dir = _prepare_merged_model(
                    base_model, sft_repo,
                    EVAL_DIR / "merged_sft_model",
                )
                model_args = (
                    f"pretrained={merged_dir},"
                    "dtype=bfloat16,trust_remote_code=True"
                )
        elif model_name == "grpo":
            merged_dir = _prepare_grpo_merged_model(
                base_model, sft_repo, grpo_repo,
                EVAL_DIR / "merged_grpo_model",
            )
            model_args = (
                f"pretrained={merged_dir},"
                "dtype=bfloat16,trust_remote_code=True"
            )
        else:
            print(f"  Unknown model type: {model_name}, skipping")
            continue

        invoke_lm_eval(
            model_args=model_args,
            tasks=tasks,
            output_dir=output_dir,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
        )
        print(f"  Results saved to {output_dir}")


# ============================================================
# Comparing Results
# ============================================================

def find_latest_result_json(result_dir: Path) -> Path | None:
    """Find the most recent lm-eval results JSON in a directory."""
    files = sorted(
        result_dir.glob("**/results_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def _pick_primary_metric(task_result: dict) -> tuple[str, float]:
    """Select the primary metric from an lm-eval task result dict."""
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


def compare_results(
    model_results: dict[str, Path | None],
) -> list[TaskResult]:
    """Compare lm-eval results across multiple models.

    Args:
        model_results: {model_name: path_to_results_json} mapping.
            None values are skipped.

    Returns:
        List of TaskResult with scores for all available models.
    """
    loaded: dict[str, dict] = {}
    for model_name, path in model_results.items():
        if path is None or not path.exists():
            print(f"  Warning: no results for {model_name}, skipping")
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded[model_name] = data.get("results", {})

    if len(loaded) < 2:
        print("  Need at least 2 models with results to compare")
        return []

    all_tasks = set()
    for results in loaded.values():
        all_tasks.update(results.keys())
    common_tasks = sorted(
        t for t in all_tasks
        if all(t in results for results in loaded.values())
    )

    if not common_tasks:
        print("  No common tasks found across models")
        return []

    comparisons: list[TaskResult] = []
    reference_model = list(loaded.keys())[0]

    for task in common_tasks:
        ref_task = loaded[reference_model][task]
        metric_name, _ = _pick_primary_metric(ref_task)
        alias = str(ref_task.get("alias", task))

        scores = {}
        for model_name, results in loaded.items():
            task_data = results[task]
            val = task_data.get(metric_name)
            if isinstance(val, (int, float)):
                scores[model_name] = float(val)
            else:
                _, fallback = _pick_primary_metric(task_data)
                scores[model_name] = fallback

        comparisons.append(TaskResult(
            task=task,
            alias=alias,
            metric_name=metric_name,
            scores=scores,
        ))

    return comparisons


# ============================================================
# Markdown Formatting
# ============================================================

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.2f}%"


def _fmt_pp(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}pp"


def build_comparison_table(
    rows: Iterable[TaskResult],
    models: list[str],
    reference_model: str | None = None,
) -> str:
    """Build a markdown comparison table.

    If reference_model is set, shows delta columns relative to that model.
    """
    rows = list(rows)
    if not rows:
        return ""

    if reference_model and reference_model in models:
        header_parts = ["| Task | Metric"]
        for m in models:
            header_parts.append(f" | {m}")
            if m != reference_model:
                header_parts.append(f" | Delta vs {reference_model}")
        header_parts.append(" |\n")

        sep_parts = ["|---|---"]
        for m in models:
            sep_parts.append("|---:")
            if m != reference_model:
                sep_parts.append("|---:")
        sep_parts.append("|\n")

        header = "".join(header_parts) + "".join(sep_parts)
    else:
        header = (
            "| Task | Metric |"
            + " | ".join(f" {m} " for m in models)
            + " |\n"
            + "|---|---|"
            + " | ".join("---:" for _ in models)
            + " |\n"
        )

    lines = []
    for r in rows:
        ref_score = r.scores.get(reference_model) if reference_model else None
        parts = [f"| {r.alias} | `{r.metric_name}`"]
        for m in models:
            score = r.scores.get(m)
            if score is not None:
                parts.append(f" | {_fmt_pct(score)}")
                if reference_model and m != reference_model and ref_score is not None:
                    parts.append(f" | {_fmt_pp(score - ref_score)}")
            else:
                parts.append(" | N/A")
                if reference_model and m != reference_model:
                    parts.append(" | N/A")
        parts.append(" |")
        lines.append("".join(parts))

    return header + "\n".join(lines) + ("\n" if lines else "")


def build_comparison_summary(comparisons: list[TaskResult], models: list[str]) -> str:
    """Build a summary of benchmark comparison across models."""
    if not comparisons or len(models) < 2:
        return ""

    reference = models[0]
    lines = [
        "### Summary",
        "",
        f"- Reference model: **{reference}**",
        f"- Compared tasks: **{len(comparisons)}**",
    ]

    for m in models[1:]:
        deltas = []
        for c in comparisons:
            ref_score = c.scores.get(reference)
            m_score = c.scores.get(m)
            if ref_score is not None and m_score is not None:
                deltas.append(m_score - ref_score)

        if not deltas:
            continue

        improved = sum(1 for d in deltas if d > 1e-12)
        degraded = sum(1 for d in deltas if d < -1e-12)
        avg_delta = sum(deltas) / len(deltas)
        lines.append(f"- **{m}** vs {reference}: "
                     f"{improved} improved, {degraded} degraded, "
                     f"mean delta {_fmt_pp(avg_delta)}")

    lines.append("")
    return "\n".join(lines)


def build_full_markdown(
    comparisons: list[TaskResult],
    models: list[str],
    model_json_paths: dict[str, Path | None] | None = None,
    top_k: int = 10,
) -> str:
    """Build a full markdown comparison report."""
    reference = models[0] if models else "base"
    overall = [c for c in comparisons if c.level == "overall"]
    groups = [c for c in comparisons if c.level == "group"]
    subtasks = [c for c in comparisons if c.level == "subtask"]

    sorted_by_delta: list[TaskResult] = []
    if subtasks and len(models) >= 2:
        target = models[-1]
        sorted_by_delta = sorted(
            subtasks,
            key=lambda c: c.scores.get(target, 0) - c.scores.get(reference, 0),
            reverse=True,
        )

    top_k = max(1, top_k)
    top_improved = sorted_by_delta[:top_k]
    top_regressed = list(reversed(sorted_by_delta[-top_k:])) if sorted_by_delta else []

    sections = [
        "# Benchmark Comparison Report",
        "",
    ]

    if model_json_paths:
        for m, p in model_json_paths.items():
            sections.append(f"- {m}: `{p}`")
        sections.append("")

    sections.append(build_comparison_summary(comparisons, models))

    sections.extend([
        "### Overall Tasks",
        "",
        build_comparison_table(overall, models, reference),
        "### Task Groups",
        "",
        build_comparison_table(groups, models, reference),
    ])

    if top_improved:
        target = models[-1]
        sections.extend([
            f"### Top {top_k} Improved Subtasks ({target} vs {reference})",
            "",
            build_comparison_table(top_improved, models, reference),
        ])

    if top_regressed:
        target = models[-1]
        sections.extend([
            f"### Top {top_k} Regressed Subtasks ({reference} vs {target})",
            "",
            build_comparison_table(top_regressed, models, reference),
        ])

    return "\n".join(sections)


# ============================================================
# Programmatic Entry Point (for pipeline)
# ============================================================

def run(
    models: list[str] | None = None,
    mode: str = "all",
    base_model: str = "Qwen/Qwen3-8B",
    sft_repo: str = "MRBSTUDIO/Humor-Qwen3-8B-SFT",
    grpo_repo: str = "MRBSTUDIO/Humor-Qwen3-8B-GRPO",
    tasks: str = "mmlu",
    num_fewshot: int = 5,
    batch_size: str = "4",
    device: str = "cuda:0",
    limit: str | None = None,
    sft_eval_mode: str = "peft",
    top_k: int = 10,
    results_dir: Path | None = None,
) -> dict | None:
    """Programmatic entry point for the benchmark step.

    Args:
        mode: "run" to only run benchmarks, "compare" to only compare,
              "all" to run then compare.

    Returns:
        Comparison data dict saved to results, or None if compare not run.
    """
    if models is None:
        models = ["base", "sft", "grpo"]
    if results_dir is None:
        results_dir = RESULTS_DIR

    results_dir.mkdir(parents=True, exist_ok=True)

    if mode in ("run", "all"):
        run_benchmarks(
            models=models,
            base_model=base_model,
            sft_repo=sft_repo,
            grpo_repo=grpo_repo,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=device,
            limit=limit,
            sft_eval_mode=sft_eval_mode,
        )

    if mode in ("compare", "all"):
        model_json_paths: dict[str, Path | None] = {}
        for m in models:
            bench_dir = DEFAULT_BENCHMARK_DIRS.get(m, EVAL_DIR / f"benchmark_{m}")
            model_json_paths[m] = find_latest_result_json(bench_dir)

        available = {m: p for m, p in model_json_paths.items() if p is not None}
        if len(available) < 2:
            print("Not enough benchmark results to compare. Run benchmarks first.")
            return None

        comparisons = compare_results(model_json_paths)
        if not comparisons:
            return None

        ordered_models = [m for m in models if m in available]
        markdown = build_full_markdown(
            comparisons, ordered_models, model_json_paths, top_k,
        )

        md_path = results_dir / "benchmark_comparison.md"
        md_path.write_text(markdown, encoding="utf-8")
        print(f"\nBenchmark comparison saved to {md_path}")

        raw_data = {
            "models": ordered_models,
            "tasks": [
                {
                    "task": c.task,
                    "alias": c.alias,
                    "metric_name": c.metric_name,
                    "level": c.level,
                    "scores": c.scores,
                }
                for c in comparisons
            ],
        }
        json_path = results_dir / "benchmark_comparison.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(raw_data, f, ensure_ascii=False, indent=2)
        print(f"Benchmark comparison data saved to {json_path}")

        return raw_data

    return None


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run lm-eval benchmarks and compare results across models."
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["run", "compare", "all"],
        help="'run' benchmarks, 'compare' existing results, or 'all' (default: all)",
    )
    parser.add_argument(
        "--models", type=str, default="base,sft,grpo",
        help="Comma-separated model names (default: base,sft,grpo)",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sft_repo", type=str, default="MRBSTUDIO/Humor-Qwen3-8B-SFT")
    parser.add_argument("--grpo_repo", type=str, default="MRBSTUDIO/Humor-Qwen3-8B-GRPO")
    parser.add_argument("--tasks", type=str, default="mmlu")
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--batch_size", type=str, default="4")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=str, default=None)
    parser.add_argument("--sft_eval_mode", type=str, default="peft",
                        choices=["peft", "merge"])
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    run(
        models=models,
        mode=args.mode,
        base_model=args.base_model,
        sft_repo=args.sft_repo,
        grpo_repo=args.grpo_repo,
        tasks=args.tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        sft_eval_mode=args.sft_eval_mode,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
