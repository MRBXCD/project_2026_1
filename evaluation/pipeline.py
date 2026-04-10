"""
Unified Evaluation Pipeline
============================

One-command entry point to run all evaluation steps or any subset:

    1. benchmark  — Run lm-eval benchmarks and compare across models
    2. generate   — Generate model outputs (base / SFT / GRPO)
    3. auto_metrics — Compute automated quality metrics
    4. llm_judge  — LLM-as-Judge pairwise comparison
    5. human_eval — Export blind A/B samples for human evaluation
    6. report     — Aggregate all available results into a Markdown report

Usage:
    # Run everything
    python -m evaluation.pipeline --steps all

    # Run specific steps
    python -m evaluation.pipeline --steps benchmark,auto_metrics
    python -m evaluation.pipeline --steps generate,llm_judge

    # Skip certain steps
    python -m evaluation.pipeline --steps all --skip benchmark

    # Re-generate report after human eval is done
    python -m evaluation.pipeline --steps report
"""

import argparse
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

ALL_STEPS = ["benchmark", "generate", "auto_metrics", "llm_judge", "human_eval", "report"]


def _resolve_steps(steps_str: str, skip_str: str | None) -> list[str]:
    """Parse --steps and --skip arguments into an ordered list of steps."""
    if steps_str.strip().lower() == "all":
        steps = list(ALL_STEPS)
    else:
        steps = [s.strip() for s in steps_str.split(",") if s.strip()]

    invalid = set(steps) - set(ALL_STEPS)
    if invalid:
        raise ValueError(
            f"Unknown step(s): {invalid}. Valid steps: {ALL_STEPS}"
        )

    if skip_str:
        skip = {s.strip() for s in skip_str.split(",") if s.strip()}
        steps = [s for s in steps if s not in skip]

    return steps


def _step_benchmark(args):
    """Step 1: Run lm-eval benchmarks and compare results."""
    from evaluation.benchmark_compare import run as benchmark_run

    models = [m.strip() for m in args.models.split(",")]
    benchmark_run(
        models=models,
        mode="all",
        base_model=args.base_model,
        sft_repo=args.sft_repo,
        grpo_repo=args.grpo_repo,
        tasks=args.benchmark_tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.benchmark_batch_size,
        device=args.device,
        limit=args.benchmark_limit,
        sft_eval_mode=args.sft_eval_mode,
        top_k=args.top_k,
        results_dir=Path(args.results_dir),
        skip_sft_for_grpo=args.skip_sft_for_grpo,
    )


def _step_generate(args):
    """Step 2: Generate model outputs."""
    from evaluation.generate_outputs import run as generate_run

    models = [m.strip() for m in args.models.split(",")]
    generate_run(
        models=models,
        base_model=args.base_model,
        sft_repo=args.sft_repo,
        grpo_repo=args.grpo_repo,
        eval_file=args.eval_file,
        n_candidates=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        skip_sft_for_grpo=args.skip_sft_for_grpo,
    )


def _step_auto_metrics(args):
    """Step 3: Compute automated metrics."""
    from evaluation.eval_auto_metrics import run as auto_metrics_run

    models = [m.strip() for m in args.models.split(",")]
    auto_metrics_run(
        models=models,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )


def _step_llm_judge(args):
    """Step 4: LLM-as-Judge pairwise comparison."""
    from evaluation.eval_llm_judge import run as llm_judge_run

    pairs = [tuple(p.strip().split(":")) for p in args.judge_pairs.split(",")]
    llm_judge_run(
        pairs=pairs,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )


def _step_human_eval(args):
    """Step 5: Export blind A/B samples for human evaluation."""
    from evaluation.export_human_eval import run as human_eval_run

    pair_parts = args.human_eval_pair.strip().split(":")
    if len(pair_parts) != 2:
        raise ValueError("--human_eval_pair must be two models separated by ':'")
    human_eval_run(
        pair=tuple(pair_parts),
        n_samples=args.human_eval_n_samples,
        seed=args.seed,
        output_dir=args.output_dir,
        results_dir=args.results_dir,
    )


def _step_report(args):
    """Step 6: Generate the aggregated evaluation report."""
    from evaluation.generate_report import run as report_run

    report_run(results_dir=args.results_dir)


STEP_HANDLERS = {
    "benchmark": _step_benchmark,
    "generate": _step_generate,
    "auto_metrics": _step_auto_metrics,
    "llm_judge": _step_llm_judge,
    "human_eval": _step_human_eval,
    "report": _step_report,
}


def main():
    parser = argparse.ArgumentParser(
        description="Unified Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps (in execution order):\n"
            "  benchmark     Run lm-eval benchmarks and compare\n"
            "  generate      Generate model outputs for evaluation\n"
            "  auto_metrics  Compute automated quality metrics\n"
            "  llm_judge     LLM-as-Judge pairwise comparison\n"
            "  human_eval    Export blind A/B samples for human eval\n"
            "  report        Aggregate all results into Markdown report\n"
            "\n"
            "Examples:\n"
            "  python -m evaluation.pipeline --steps all\n"
            "  python -m evaluation.pipeline --steps generate,auto_metrics,report\n"
            "  python -m evaluation.pipeline --steps all --skip benchmark\n"
            "  python -m evaluation.pipeline --steps report\n"
        ),
    )

    # --- Step selection ---
    parser.add_argument(
        "--steps", type=str, required=True,
        help="Comma-separated step names or 'all' (required)",
    )
    parser.add_argument(
        "--skip", type=str, default=None,
        help="Comma-separated step names to skip",
    )

    # --- Model configuration ---
    parser.add_argument(
        "--models", type=str, default="base,sft,grpo",
        help="Comma-separated model names (default: base,sft,grpo)",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--sft_repo", type=str, default="MRBSTUDIO/Humor-Qwen3-8B-SFT")
    parser.add_argument("--grpo_repo", type=str, default="MRBSTUDIO/Humor-Qwen3-8B-GRPO")

    # --- Benchmark configuration ---
    parser.add_argument("--benchmark_tasks", type=str, default="mmlu")
    parser.add_argument("--num_fewshot", type=int, default=5)
    parser.add_argument("--benchmark_batch_size", type=str, default="4")
    parser.add_argument("--benchmark_limit", type=str, default=None)
    parser.add_argument("--sft_eval_mode", type=str, default="peft",
                        choices=["peft", "merge"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--skip_sft_for_grpo", action="store_true",
        help="Load GRPO adapter directly on base model without SFT merge (for base->GRPO experiments)",
    )

    # --- Generation configuration ---
    parser.add_argument(
        "--eval_file", type=str,
        default=str(PROJECT_ROOT / "data" / "grpo" / "grpo_prompts_eval.jsonl"),
    )
    parser.add_argument("--n_candidates", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)

    # --- Judge configuration ---
    parser.add_argument(
        "--judge_pairs", type=str, default="base:grpo,sft:grpo,base:sft",
        help="Comma-separated model pairs for LLM judge (default: base:grpo,sft:grpo,base:sft)",
    )

    # --- Human eval configuration ---
    parser.add_argument(
        "--human_eval_pair", type=str, default="base:grpo",
        help="Model pair for human eval export (default: base:grpo)",
    )
    parser.add_argument("--human_eval_n_samples", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)

    # --- Output directories ---
    parser.add_argument(
        "--output_dir", type=str,
        default=str(PROJECT_ROOT / "evaluation" / "outputs"),
        help="Directory for model output JSONL files",
    )
    parser.add_argument(
        "--results_dir", type=str,
        default=str(PROJECT_ROOT / "evaluation" / "results"),
        help="Directory for evaluation results and report",
    )

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    steps = _resolve_steps(args.steps, args.skip)

    if not steps:
        print("No steps to run.")
        return

    print("=" * 60)
    print("Evaluation Pipeline")
    print(f"Steps: {', '.join(steps)}")
    print(f"Models: {args.models}")
    print("=" * 60)

    for step in steps:
        handler = STEP_HANDLERS[step]
        print(f"\n{'=' * 60}")
        print(f"  STEP: {step}")
        print(f"{'=' * 60}\n")

        start = time.time()
        try:
            handler(args)
        except Exception as e:
            print(f"\n  ERROR in step '{step}': {e}")
            print(f"  Continuing with remaining steps...\n")
            continue
        elapsed = time.time() - start
        print(f"\n  Step '{step}' completed in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print("Pipeline finished.")
    print(f"Results directory: {args.results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
