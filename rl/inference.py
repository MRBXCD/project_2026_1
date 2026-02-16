"""
Inference and Rejection Sampling
================================

This module provides inference capabilities for the GRPO-trained model,
including a rejection sampling strategy to guarantee constraint satisfaction
at generation time.

Rejection Sampling Overview:
    Even after GRPO training, the model does not *guarantee* that every
    single output will satisfy all hard constraints (e.g., containing
    required keywords). Rejection sampling addresses this by:

    1. Generating N candidate responses for each prompt (N >> 1).
    2. Filtering out candidates that violate hard constraints (keywords).
    3. Ranking the remaining candidates by a soft quality score (reward).
    4. Returning the highest-scoring valid candidate.

    This is a standard technique in constrained text generation — the
    model handles the "creative" part, and rejection sampling handles
    the "compliance" part.

Model Loading:
    The inference model is assembled from three components:
    1. Base Qwen3-8B weights.
    2. SFT LoRA adapter (merged into base weights).
    3. GRPO LoRA adapter (loaded on top of merged weights).

    This mirrors the training setup: GRPO was trained on a model that
    already had SFT weights merged in.

Usage:
    # Single prompt
    python -m rl.inference --prompt "Write a joke about AI regulations"

    # Single prompt with keywords
    python -m rl.inference \\
        --prompt "Write a joke" \\
        --keywords "penguin,bankruptcy"

    # Batch inference from JSONL file
    python -m rl.inference --input_file data/grpo/grpo_prompts.jsonl

Dependencies:
    - transformers, peft, torch
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.rewards import compute_reward


# ============================================================
# Path Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "sft" / "final"
GRPO_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "grpo" / "final"


# ============================================================
# Model Loading
# ============================================================

def load_model(
    base_model_name: str = "Qwen/Qwen3-8B",
    sft_adapter_path: str | Path = SFT_ADAPTER_DIR,
    grpo_adapter_path: str | Path = GRPO_ADAPTER_DIR,
    merge_all: bool = False,
) -> tuple[AutoModelForCausalLM | PeftModel, AutoTokenizer]:
    """Load the full GRPO-trained model for inference.

    Assembly process:
    1. Load base model in bf16.
    2. Load SFT LoRA adapter and merge into base weights.
    3. Load GRPO LoRA adapter on top of the merged model.
    4. Optionally merge GRPO adapter too (merge_all=True) for
       faster inference at the cost of losing adapter flexibility.

    Args:
        base_model_name: HuggingFace model name or local path.
        sft_adapter_path: Path to the SFT LoRA adapter directory.
        grpo_adapter_path: Path to the GRPO LoRA adapter directory.
        merge_all: If True, merge the GRPO adapter as well, producing
            a plain PreTrainedModel. Slightly faster inference but
            cannot unload the adapter afterwards.
            If False (default), return a PeftModel with the GRPO adapter
            loaded but not merged.

    Returns:
        tuple: (model, tokenizer)
            model is either a PeftModel (merge_all=False) or a plain
            PreTrainedModel (merge_all=True). Both support .generate().

    Raises:
        FileNotFoundError: If adapter paths do not exist.
    """
    sft_adapter_path = Path(sft_adapter_path)
    grpo_adapter_path = Path(grpo_adapter_path)

    if not sft_adapter_path.exists():
        raise FileNotFoundError(
            f"SFT adapter not found at {sft_adapter_path}. "
            f"Run SFT training first: python -m sft.train_sft"
        )
    if not grpo_adapter_path.exists():
        raise FileNotFoundError(
            f"GRPO adapter not found at {grpo_adapter_path}. "
            f"Run GRPO training first: python -m rl.train_grpo"
        )

    # Step 1: Load base model
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Step 2: Load and merge SFT adapter
    print(f"Loading SFT adapter: {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, str(sft_adapter_path))
    model = model.merge_and_unload()
    print("  SFT adapter merged.")

    # Step 3: Load GRPO adapter
    print(f"Loading GRPO adapter: {grpo_adapter_path}")
    model = PeftModel.from_pretrained(model, str(grpo_adapter_path))

    # Step 4: Optionally merge GRPO adapter too
    if merge_all:
        model = model.merge_and_unload()
        print("  GRPO adapter merged. Model is a plain PreTrainedModel.")
    else:
        print("  GRPO adapter loaded (not merged).")

    model.eval()
    return model, tokenizer


# ============================================================
# Candidate Generation
# ============================================================

def generate_candidates(
    model: AutoModelForCausalLM | PeftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    n_candidates: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
) -> list[str]:
    """Generate N candidate responses for a single prompt.

    Uses the model's generate() method with num_return_sequences=N to
    produce multiple diverse candidates in a single forward pass. This
    is more efficient than calling generate() N times individually.

    The prompt is formatted using Qwen3's chat template with thinking
    mode disabled (enable_thinking=False) to produce direct joke outputs.

    Args:
        model: The loaded model (PeftModel or PreTrainedModel).
        tokenizer: The tokenizer with padding_side="left".
        prompt_text: The user prompt in plain text.
            e.g., "Write a funny joke about AI regulations."
        n_candidates: Number of candidate responses to generate.
            More candidates increase the chance of finding a
            constraint-satisfying response, but cost more compute.
            Default 16.
        max_new_tokens: Maximum number of new tokens per candidate.
            Default 256 (jokes are short).
        temperature: Sampling temperature. Higher values produce more
            diverse candidates. Default 0.9.
        top_p: Nucleus sampling threshold. Default 0.95.

    Returns:
        list[str]: N candidate response strings (decoded, stripped).
    """
    messages = [{"role": "user", "content": prompt_text}]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=n_candidates,
        )

    # Decode each candidate, stripping the prompt prefix and special tokens
    candidates = []
    for seq in outputs:
        response = tokenizer.decode(
            seq[prompt_len:], skip_special_tokens=True
        ).strip()
        candidates.append(response)

    return candidates


# ============================================================
# Rejection Sampling
# ============================================================

def rejection_sample(
    candidates: list[str],
    prompt_text: str,
    keywords: list[str] | None = None,
    headline: str | None = None,
    humor_scorer=None,
) -> dict:
    """Apply rejection sampling to select the best candidate.

    Rejection sampling pipeline:
    1. Hard constraint filter: Remove candidates that do not contain
       all required keywords (if any). Note: headline relevance is a
       SOFT constraint — it affects ranking scores but does not filter
       candidates, because word overlap is too noisy for hard filtering.
    2. Soft quality ranking: Score remaining candidates using the
       composite reward function (same as GRPO training reward),
       which includes format, keyword, relevance, and humor scores.
    3. Selection: Return the highest-scoring valid candidate.
    4. Fallback: If no candidate passes the hard filter, return the
       candidate with the most keyword hits (best-effort).

    This function reuses compute_reward() from rl.rewards to ensure
    that the scoring at inference time is consistent with what the
    model was trained on.

    Args:
        candidates: List of candidate response strings (from
            generate_candidates).
        prompt_text: The original user prompt (for reward computation).
        keywords: List of required keywords. None or empty list means
            no keyword constraint (skip hard filtering).
        headline: The news headline text. Passed to compute_reward()
            for relevance scoring (soft constraint). None or empty
            string means no headline.
        humor_scorer: Optional humor scoring function (Phase 2).
            Same interface as in compute_reward().

    Returns:
        dict: Rejection sampling result with keys:
            - "best_response" (str): The selected best response.
            - "best_score" (float): Reward score of the best response.
            - "all_candidates" (list[str]): All generated candidates.
            - "valid_candidates" (list[str]): Candidates passing hard
              constraints.
            - "constraint_pass_rate" (float): Fraction of candidates
              that passed hard constraints (0.0 to 1.0).
            - "scores" (list[float]): Reward scores of valid candidates
              (sorted descending).
    """
    # Step 1: Hard constraint filter (keywords)
    if keywords:
        valid_candidates = [
            c for c in candidates
            if all(kw.lower() in c.lower() for kw in keywords)
        ]
    else:
        valid_candidates = candidates.copy()

    constraint_pass_rate = (
        len(valid_candidates) / len(candidates) if candidates else 0.0
    )

    # Step 2: Fallback if no candidate passes hard constraints
    if not valid_candidates:
        # Best-effort: pick the candidate with the most keyword hits
        if keywords:
            best_fallback = max(
                candidates,
                key=lambda c: sum(
                    kw.lower() in c.lower() for kw in keywords
                ),
            )
        else:
            best_fallback = candidates[0]

        return {
            "best_response": best_fallback,
            "best_score": compute_reward(prompt_text, best_fallback, keywords, headline),
            "all_candidates": candidates,
            "valid_candidates": [],
            "constraint_pass_rate": 0.0,
            "scores": [],
        }

    # Step 3: Score valid candidates using the same reward function as training
    scored = [
        (c, compute_reward(prompt_text, c, keywords, headline, humor_scorer))
        for c in valid_candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_response, best_score = scored[0]
    scores = [s for _, s in scored]

    return {
        "best_response": best_response,
        "best_score": best_score,
        "all_candidates": candidates,
        "valid_candidates": [c for c, _ in scored],
        "constraint_pass_rate": constraint_pass_rate,
        "scores": scores,
    }


# ============================================================
# End-to-End Inference
# ============================================================

def run_inference(
    model: AutoModelForCausalLM | PeftModel,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    keywords: list[str] | None = None,
    headline: str | None = None,
    n_candidates: int = 16,
    max_retries: int = 3,
    humor_scorer=None,
    **generate_kwargs,
) -> dict:
    """Run end-to-end inference with retry: generate -> filter -> retry if needed.

    Chains generate_candidates() and rejection_sample() together, with
    bounded retry logic: if no candidate passes hard constraints on the
    first attempt, generates additional candidates (up to max_retries
    total rounds) before falling back to best-effort.

    Each retry round accumulates candidates (not discards previous ones),
    so the final rejection_sample considers ALL candidates across all
    rounds, maximizing the chance of finding a good valid response.

    Args:
        model: The loaded model.
        tokenizer: The tokenizer.
        prompt_text: The user prompt in plain text.
        keywords: Optional keyword constraints.
        headline: Optional headline text for relevance scoring.
        n_candidates: Number of candidates per generation round.
        max_retries: Maximum number of generation rounds. Default 3.
            Total candidates considered = n_candidates * (number of
            rounds actually executed). Retries stop early as soon as
            at least one valid candidate is found.
        humor_scorer: Optional humor scoring function.
        **generate_kwargs: Additional keyword arguments passed to
            generate_candidates() (e.g., max_new_tokens, temperature).

    Returns:
        dict: Same as rejection_sample() output, plus:
            - "num_rounds" (int): Number of generation rounds executed.
    """
    all_candidates = []

    for round_idx in range(max_retries):
        new_candidates = generate_candidates(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            n_candidates=n_candidates,
            **generate_kwargs,
        )
        all_candidates.extend(new_candidates)

        # Check if any candidate passes hard constraints.
        # If no keywords, first round is always sufficient.
        if not keywords:
            break

        has_valid = any(
            all(kw.lower() in c.lower() for kw in keywords)
            for c in all_candidates
        )
        if has_valid:
            break

        if round_idx < max_retries - 1:
            print(f"    Retry {round_idx + 2}/{max_retries}: "
                  f"no valid candidate yet in {len(all_candidates)} samples")

    result = rejection_sample(
        candidates=all_candidates,
        prompt_text=prompt_text,
        keywords=keywords,
        headline=headline,
        humor_scorer=humor_scorer,
    )
    result["num_rounds"] = round_idx + 1

    return result


# ============================================================
# Main CLI
# ============================================================

def main():
    """Inference CLI entry point.

    Supports two modes:
    1. Single prompt mode (--prompt): Generate a joke for one prompt.
    2. Batch mode (--input_file): Process a JSONL file of prompts.

    Results are printed to stdout and optionally saved to --output_file.
    """
    parser = argparse.ArgumentParser(description="GRPO Model Inference")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-8B",
        help="Base model name or path",
    )
    parser.add_argument(
        "--sft_adapter_path", type=str, default=str(SFT_ADAPTER_DIR),
        help="Path to the SFT LoRA adapter directory",
    )
    parser.add_argument(
        "--grpo_adapter_path", type=str, default=str(GRPO_ADAPTER_DIR),
        help="Path to the GRPO LoRA adapter directory",
    )
    parser.add_argument(
        "--merge_all", action="store_true",
        help="Merge all adapters for faster inference",
    )

    # --- Input modes (mutually exclusive) ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--prompt", type=str,
        help="Single prompt text for inference",
    )
    input_group.add_argument(
        "--input_file", type=str,
        help="Path to JSONL file with prompts for batch inference. "
             "Expected fields: prompt (list of message dicts), "
             "keywords (list of strings, optional)",
    )

    # --- Generation parameters ---
    parser.add_argument(
        "--keywords", type=str, default=None,
        help="Comma-separated keywords for single prompt mode "
             "(e.g., 'penguin,bankruptcy')",
    )
    parser.add_argument(
        "--headline", type=str, default=None,
        help="News headline for single prompt mode "
             "(used for relevance scoring)",
    )
    parser.add_argument(
        "--n_candidates", type=int, default=16,
        help="Number of candidates per prompt (default: 16)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max new tokens per candidate (default: 256)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature (default: 0.9)",
    )

    # --- Output ---
    parser.add_argument(
        "--output_file", type=str, default=None,
        help="Optional path to save results as JSONL",
    )
    args = parser.parse_args()

    # ---- Load Model ----
    print("=" * 60)
    print("Loading model")
    print("=" * 60)
    model, tokenizer = load_model(
        base_model_name=args.model_name,
        sft_adapter_path=args.sft_adapter_path,
        grpo_adapter_path=args.grpo_adapter_path,
        merge_all=args.merge_all,
    )

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    all_results = []

    # ---- Single Prompt Mode ----
    if args.prompt is not None:
        keywords = (
            [kw.strip() for kw in args.keywords.split(",")]
            if args.keywords
            else None
        )

        print(f"\nPrompt: {args.prompt}")
        if args.headline:
            print(f"Headline: {args.headline}")
        if keywords:
            print(f"Keywords: {keywords}")

        result = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt_text=args.prompt,
            keywords=keywords,
            headline=args.headline,
            n_candidates=args.n_candidates,
            **generate_kwargs,
        )

        print(f"\nBest response: {result['best_response']}")
        print(f"Score: {result['best_score']:.3f}")
        print(f"Constraint pass rate: {result['constraint_pass_rate']:.1%}")
        print(f"Total candidates: {len(result['all_candidates'])}")
        print(f"Valid candidates: {len(result['valid_candidates'])}")

        all_results.append({
            "prompt": args.prompt,
            "headline": args.headline or "",
            "keywords": keywords or [],
            **result,
        })

    # ---- Batch Mode ----
    if args.input_file is not None:
        input_path = Path(args.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            prompts_data = [json.loads(line) for line in f if line.strip()]

        print(f"\nBatch inference: {len(prompts_data)} prompts")
        print("-" * 60)

        for i, item in enumerate(prompts_data):
            # Extract prompt text from conversational format
            prompt_messages = item.get("prompt", [])
            prompt_text = prompt_messages[-1]["content"] if prompt_messages else ""
            keywords = item.get("keywords", []) or None
            item_headline = item.get("headline", "") or None

            result = run_inference(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                keywords=keywords,
                headline=item_headline,
                n_candidates=args.n_candidates,
                **generate_kwargs,
            )

            print(f"[{i + 1}/{len(prompts_data)}] "
                  f"pass_rate={result['constraint_pass_rate']:.0%} "
                  f"score={result['best_score']:.3f} "
                  f"| {result['best_response'][:80]}...")

            all_results.append({
                "prompt": prompt_text,
                "headline": item.get("headline", ""),
                "keywords": keywords or [],
                **result,
            })

    # ---- Save Results ----
    if args.output_file and all_results:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for result in all_results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
