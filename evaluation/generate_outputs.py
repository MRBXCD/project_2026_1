"""
Generate Model Outputs for Evaluation
======================================

Generates responses from Base, SFT, and GRPO models on the evaluation
prompt set. Each model is loaded, used for generation, then unloaded
before loading the next one (to fit within a single GPU).

For each prompt, N candidate responses are generated and the best one
is selected via rejection sampling (keyword filtering + reward scoring).

Usage (standalone):
    python -m evaluation.generate_outputs --models base,sft,grpo
    python -m evaluation.generate_outputs --models grpo --n_candidates 32

Usage (via pipeline):
    python -m evaluation.pipeline --steps generate

Output:
    evaluation/outputs/{model_name}.jsonl
"""

import argparse
import gc
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from rl.rewards import compute_reward


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_PROMPTS_FILE = PROJECT_ROOT / "data" / "grpo" / "grpo_prompts_eval.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "outputs"


# ============================================================
# Model Loading / Unloading
# ============================================================

def _unload(model, tokenizer):
    """Release model GPU memory."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_base(model_name: str):
    """Load the raw base model (no fine-tuning)."""
    print(f"  Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    return model, tokenizer


def _load_sft(model_name: str, sft_repo: str):
    """Load base + SFT LoRA adapter (not merged)."""
    print(f"  Loading base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    print(f"  Loading SFT adapter: {sft_repo}")
    model = PeftModel.from_pretrained(model, sft_repo)
    model.eval()
    return model, tokenizer


def _load_grpo(model_name: str, sft_repo: str, grpo_repo: str):
    """Load base + merge SFT + load GRPO adapter."""
    print(f"  Loading base: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    print(f"  Merging SFT adapter: {sft_repo}")
    model = PeftModel.from_pretrained(model, sft_repo)
    model = model.merge_and_unload()
    print(f"  Loading GRPO adapter: {grpo_repo}")
    model = PeftModel.from_pretrained(model, grpo_repo)
    model.eval()
    return model, tokenizer


# ============================================================
# Generation
# ============================================================

def generate_candidates(
    model,
    tokenizer,
    prompt_text: str,
    n_candidates: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
) -> list[str]:
    """Generate N candidate responses for a single prompt."""
    messages = [
        {"role": "system", "content": "/no_think"},
        {"role": "user", "content": prompt_text},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=n_candidates,
        )

    candidates = []
    for seq in outputs:
        resp = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True).strip()
        candidates.append(resp)
    return candidates


def select_best(
    candidates: list[str],
    prompt_text: str,
    keywords: list[str] | None,
    headline: str | None,
) -> tuple[str, float, float]:
    """Select best candidate via rejection sampling.

    Returns (best_response, best_score, constraint_pass_rate).
    """
    if keywords:
        valid = [
            c for c in candidates
            if all(kw.lower() in c.lower() for kw in keywords)
        ]
    else:
        valid = candidates[:]

    pass_rate = len(valid) / len(candidates) if candidates else 0.0

    pool = valid if valid else candidates
    scored = [
        (c, compute_reward(prompt_text, c, keywords, headline))
        for c in pool
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0][0], scored[0][1], pass_rate


# ============================================================
# Per-Model Generation Loop
# ============================================================

def run_generation(
    model,
    tokenizer,
    prompts: list[dict],
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict]:
    """Generate and select best response for every evaluation prompt."""
    results = []
    for i, item in enumerate(prompts):
        prompt_messages = item["prompt"]
        prompt_text = prompt_messages[-1]["content"] if prompt_messages else ""
        keywords = item.get("keywords", []) or None
        headline = item.get("headline", "") or None
        lang = item.get("lang", "")

        candidates = generate_candidates(
            model, tokenizer, prompt_text,
            n_candidates=n_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        best, score, pass_rate = select_best(
            candidates, prompt_text, keywords, headline,
        )

        results.append({
            "headline": headline or "",
            "keywords": keywords or [],
            "lang": lang,
            "prompt_text": prompt_text,
            "best_response": best,
            "all_candidates": candidates,
            "constraint_pass_rate": pass_rate,
            "best_score": round(score, 4),
        })

        if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
            print(f"    [{i + 1}/{len(prompts)}] score={score:.3f} "
                  f"pass_rate={pass_rate:.0%} | {best[:60]}...")

    return results


# ============================================================
# Programmatic Entry Point (for pipeline)
# ============================================================

def run(
    models: list[str] | None = None,
    base_model: str = "Qwen/Qwen3-8B",
    sft_repo: str = "MRBSTUDIO/Humor-Qwen3-8B-SFT",
    grpo_repo: str = "MRBSTUDIO/Humor-Qwen3-8B-GRPO",
    eval_file: str | Path | None = None,
    n_candidates: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 0.9,
    output_dir: str | Path | None = None,
):
    """Programmatic entry point for the generate step."""
    if models is None:
        models = ["base", "sft", "grpo"]
    eval_path = Path(eval_file) if eval_file else EVAL_PROMPTS_FILE
    out_dir = Path(output_dir) if output_dir else OUTPUT_DIR

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation prompts not found at {eval_path}. "
            f"Run: python -m data_preprocessing.pipeline --stage format_grpo --eval_ratio 0.2"
        )

    with open(eval_path, encoding="utf-8") as f:
        prompts = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(prompts)} evaluation prompts from {eval_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "base": lambda: _load_base(base_model),
        "sft": lambda: _load_sft(base_model, sft_repo),
        "grpo": lambda: _load_grpo(base_model, sft_repo, grpo_repo),
    }

    for model_name in models:
        if model_name not in loaders:
            print(f"Unknown model: {model_name}, skipping")
            continue

        print(f"\n{'=' * 60}")
        print(f"Generating with: {model_name}")
        print(f"{'=' * 60}")

        model, tokenizer = loaders[model_name]()

        results = run_generation(
            model, tokenizer, prompts,
            n_candidates=n_candidates,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        _unload(model, tokenizer)

        out_path = out_dir / f"{model_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved {len(results)} results to {out_path}")

    print("\nGeneration complete.")


# ============================================================
# CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate evaluation outputs")
    parser.add_argument(
        "--models", type=str, default="base,sft,grpo",
        help="Comma-separated model names to generate from (default: base,sft,grpo)",
    )
    parser.add_argument(
        "--eval_file", type=str, default=str(EVAL_PROMPTS_FILE),
        help="Path to evaluation prompts JSONL",
    )
    parser.add_argument(
        "--base_model", type=str, default="Qwen/Qwen3-8B",
        help="Base model name or HF repo ID",
    )
    parser.add_argument(
        "--sft_repo", type=str, default="MRBSTUDIO/Humor-Qwen3-8B-SFT",
        help="SFT adapter HF repo ID or local path",
    )
    parser.add_argument(
        "--grpo_repo", type=str, default="MRBSTUDIO/Humor-Qwen3-8B-GRPO",
        help="GRPO adapter HF repo ID or local path",
    )
    parser.add_argument("--n_candidates", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()

    run(
        models=[m.strip() for m in args.models.split(",")],
        base_model=args.base_model,
        sft_repo=args.sft_repo,
        grpo_repo=args.grpo_repo,
        eval_file=args.eval_file,
        n_candidates=args.n_candidates,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
