"""
Reward Model Scorer
====================

This module provides inference wrappers for the trained reward model,
exposing scorer callables that are compatible with the existing reward
function interface in rl/rewards.py.

The reward model (trained by rl/train_reward_model.py) is a sequence
classification model: base LM + linear score head, producing a scalar
reward for each (prompt, response) pair. Compared to the Gemini
LLM-as-Judge (rl/humor_judge.py), the reward model is:

    - Deterministic: same input always produces the same score.
    - Fast: on-device forward pass, no API latency or rate limits.
    - Fine-grained: continuous scalar output instead of 1-5 integers.
    - Task-specific: trained on humor preference data.

Scorer Interface:
    Two factory functions mirror those in rl/humor_judge.py:

    build_reward_model_scorer(model_path)
        -> Callable[[str, str], float]
        Single-pair scorer for inference / rejection sampling.

    build_batch_reward_model_scorer(model_path)
        -> Callable[[list[str], list[str]], list[float]]
        Batch scorer for efficient GRPO training.

    Both return scores in [-1.0, 1.0] via tanh normalization,
    matching the range expected by rl.rewards.compute_reward().

Integration with GRPO:
    from rl.reward_model import build_batch_reward_model_scorer
    from rl.rewards import build_reward_fn

    batch_scorer = build_batch_reward_model_scorer("checkpoints/reward_model/final")
    reward_fn = build_reward_fn(batch_humor_scorer=batch_scorer)

Dependencies:
    - transformers, peft, torch
"""

import math
from pathlib import Path
from typing import Callable

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Constants
# ============================================================

# Batch size for reward model inference during GRPO training.
# Controls how many (prompt, response) pairs are tokenized and
# forwarded through the model at once. Larger = faster but more VRAM.
INFERENCE_BATCH_SIZE = 32


# ============================================================
# Model Loading
# ============================================================

def _load_reward_model(
    model_path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load the trained reward model (base + LoRA adapter).

    The reward model checkpoint directory contains either:
    - A LoRA adapter (adapter_config.json + adapter_model.safetensors)
      that was trained on top of a base model, plus the score head
      weights saved via modules_to_save. In this case, we load the base
      model as AutoModelForSequenceClassification, then load the adapter.
    - A full merged model (if the user merged the adapter).

    The function auto-detects which case applies by checking for
    adapter_config.json.

    Args:
        model_path: Path to the reward model checkpoint directory.
        device: Target device. If None, uses "cuda" if available.

    Returns:
        tuple: (model, tokenizer) with model in eval mode on the target device.

    Raises:
        FileNotFoundError: If model_path does not exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Reward model not found at {model_path}. "
            f"Please train it first: python -m rl.train_reward_model"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    adapter_config = model_path / "adapter_config.json"
    is_peft_adapter = adapter_config.exists()

    if is_peft_adapter:
        import json
        with open(adapter_config) as f:
            config = json.load(f)
        base_model_name = config.get("base_model_name_or_path", "")

        print(f"Loading reward model base: {base_model_name}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )

        print(f"Loading reward model adapter: {model_path}")
        model = PeftModel.from_pretrained(base_model, str(model_path))

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            padding_side="right",
        )
    else:
        print(f"Loading merged reward model: {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            num_labels=1,
            torch_dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            padding_side="right",
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Reward model loaded: {total_params / 1e6:.1f}M parameters on {device}")

    return model, tokenizer


# ============================================================
# Score Normalization
# ============================================================

def _normalize_scores(raw_scores: list[float]) -> list[float]:
    """Normalize raw reward model logits to [-1.0, 1.0] via tanh.

    The reward model outputs unbounded scalar logits. We apply tanh
    to squash them into [-1, 1], which is the range expected by
    rl.rewards.compute_reward(humor_score_override=...).

    Tanh is preferred over linear clamping because:
    - It preserves relative ordering of scores.
    - It smoothly handles outliers (no harsh cutoff).
    - Scores near 0 (ambiguous quality) maintain fine granularity.

    Args:
        raw_scores: Unbounded reward model logits.

    Returns:
        list[float]: Scores in [-1.0, 1.0].
    """
    return [math.tanh(s) for s in raw_scores]


# ============================================================
# Single Scorer (for inference / rejection sampling)
# ============================================================

def build_reward_model_scorer(
    model_path: str | Path,
    device: str | torch.device | None = None,
) -> Callable[[str, str], float]:
    """Build a single-pair scorer from the trained reward model.

    Returns a closure that scores one (prompt, response) pair at a time
    by running a forward pass through the reward model. The model is
    loaded once and reused across all calls.

    This scorer matches the interface expected by build_reward_fn():
        scorer(prompt: str, response: str) -> float

    For GRPO training with many completions per step, use
    build_batch_reward_model_scorer() instead for better throughput.

    Args:
        model_path: Path to the trained reward model checkpoint.
        device: Target device. If None, auto-detects.

    Returns:
        Callable: scorer(prompt, response) -> float in [-1.0, 1.0].
    """
    model, tokenizer = _load_reward_model(model_path, device)

    @torch.no_grad()
    def scorer(prompt: str, response: str) -> float:
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to(model.device)

        logits = model(**inputs).logits
        raw_score = logits.squeeze().item()
        return math.tanh(raw_score)

    return scorer


# ============================================================
# Batch Scorer (for efficient GRPO training)
# ============================================================

def build_batch_reward_model_scorer(
    model_path: str | Path,
    device: str | torch.device | None = None,
    batch_size: int = INFERENCE_BATCH_SIZE,
) -> Callable[[list[str], list[str]], list[float]]:
    """Build a batch scorer from the trained reward model.

    Returns a closure that scores multiple (prompt, response) pairs in
    batched forward passes. This is much more efficient than scoring
    one at a time during GRPO training.

    For 128 completions per GRPO step with batch_size=32:
        128 / 32 = 4 forward passes per step (vs 128 for single scorer,
        or 16 API calls for Gemini batch judge).

    The returned callable has the signature:
        scorer(prompts: list[str], responses: list[str]) -> list[float]

    Args:
        model_path: Path to the trained reward model checkpoint.
        device: Target device. If None, auto-detects.
        batch_size: Number of pairs per forward pass.

    Returns:
        Callable: scorer(prompts, responses) -> list[float].
            Each score is in [-1.0, 1.0].
    """
    model, tokenizer = _load_reward_model(model_path, device)

    @torch.no_grad()
    def scorer(prompts: list[str], responses: list[str]) -> list[float]:
        all_scores: list[float] = []

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]
            batch_responses = responses[start:end]

            texts = []
            for p, r in zip(batch_prompts, batch_responses):
                messages = [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r},
                ]
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                )
                texts.append(text)

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(model.device)

            logits = model(**inputs).logits
            raw_scores = logits.squeeze(-1).tolist()

            if isinstance(raw_scores, float):
                raw_scores = [raw_scores]

            all_scores.extend(_normalize_scores(raw_scores))

        return all_scores

    return scorer


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/reward_model/final"
    print(f"Testing reward model scorer from: {model_path}")

    scorer = build_reward_model_scorer(model_path)

    test_cases = [
        ("Tell me a joke.", "Why did the chicken cross the road? To get to the other side."),
        ("Tell me a joke.", "I told my wife she was drawing her eyebrows too high. She looked surprised."),
        ("Tell me a joke.", "asdf asdf asdf"),
    ]

    for prompt, response in test_cases:
        score = scorer(prompt, response)
        print(f"  Score: {score:+.4f}  |  {response[:60]}...")
