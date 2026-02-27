"""
GRPO Training Script
====================

Train the SFT-finetuned Qwen3-8B model with Group Relative Policy
Optimization (GRPO), using TRL's GRPOTrainer.

GRPO Training Overview:
    Unlike SFT which learns from labeled (prompt, response) pairs, GRPO is
    an online RL algorithm: the model generates its own responses, which are
    then scored by a reward function. The training signal comes from the
    *relative quality* within a group of responses to the same prompt.

    For each training step:
    1. Sample a batch of prompts from the dataset.
    2. For each prompt, generate G completions (num_generations) using the
       current policy.
    3. Score all completions with the reward function.
    4. Compute group-relative advantages (normalize rewards within each
       group of G completions).
    5. Update the policy using a PPO-clip style objective with KL
       divergence regularization against the reference model.

Model Loading Strategy:
    Since the GRPO stage builds on top of the SFT model, we need to:
    1. Load the base Qwen3-8B model.
    2. Load the SFT LoRA adapter from checkpoints/sft/final.
    3. Merge the SFT adapter into the base weights (merge_and_unload).
       This produces a plain model with SFT knowledge baked in.
    4. Apply a NEW LoRA adapter (smaller rank) for GRPO training.
       GRPOTrainer uses the merged model as the reference (pi_ref),
       and trains the new LoRA as the policy (pi_theta).

    This "merge then re-LoRA" strategy ensures:
    - The reference model IS the SFT model (for KL constraint).
    - Only the new GRPO LoRA weights are trainable.
    - The SFT knowledge is preserved as a stable baseline.

Qwen3 Thinking Mode:
    Qwen3 defaults to a "thinking" mode that generates internal reasoning
    in <think>...</think> tags. For humor generation this is undesirable
    (jokes should be direct, not preceded by reasoning). We disable it via
    chat_template_kwargs={"enable_thinking": False} in GRPOConfig.

Usage:
    python -m rl.train_grpo
    python -m rl.train_grpo --model_name Qwen/Qwen3-8B --num_generations 8
    python -m rl.train_grpo --beta 0.04 --lr 5e-6
    python -m rl.train_grpo --use_reward_model

Dependencies:
    - transformers, trl (>= 0.27), peft, torch, datasets, accelerate
    - wandb (optional, for experiment tracking)
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from rl.humor_judge import build_batch_humor_scorer
from rl.reward_model import build_batch_reward_model_scorer
from rl.rewards import build_reward_fn


# ============================================================
# Path Constants
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRPO_PROMPTS_FILE = PROJECT_ROOT / "data" / "grpo" / "grpo_prompts.jsonl"
SFT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "sft" / "final"
GRPO_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "grpo"


# ============================================================
# Model Loading — Merge SFT Adapter into Base Model
# ============================================================

def load_sft_merged_model(
    base_model_name: str,
    sft_adapter_path: str | Path,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the base model, merge the SFT LoRA adapter, and return a plain model.

    This function implements the "merge then re-LoRA" strategy:
    1. Load the base Qwen3-8B model in bf16.
    2. Load the SFT LoRA adapter on top of it (PeftModel).
    3. Call merge_and_unload() to permanently fuse the LoRA weights
       into the base model's linear layers.
    4. Return the merged model (now a standard PreTrainedModel, not PeftModel).

    The returned model serves as both:
    - The starting point for the new GRPO LoRA adapter.
    - The reference model (pi_ref) for KL divergence computation.
      (GRPOTrainer creates its own reference copy internally.)

    Tokenizer notes:
    - padding_side="left" is required for generation-based training.
      Unlike SFT where we pad on the right (for label alignment),
      GRPO generates completions left-to-right and needs left padding
      so that the actual prompt tokens are right-aligned and contiguous
      with the generated tokens.

    Args:
        base_model_name: HuggingFace model name or local path.
            e.g., "Qwen/Qwen3-8B"
        sft_adapter_path: Path to the SFT LoRA adapter directory.
            e.g., "checkpoints/sft/final"

    Returns:
        tuple: (model, tokenizer)
            - model: A standard PreTrainedModel with SFT weights merged in.
            - tokenizer: Configured with padding_side="left".

    Raises:
        FileNotFoundError: If sft_adapter_path does not exist.
    """
    sft_adapter_path = Path(sft_adapter_path)
    if not sft_adapter_path.exists():
        raise FileNotFoundError(
            f"SFT adapter not found at {sft_adapter_path}. "
            f"Please run SFT training first: python -m sft.train_sft"
        )

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
        attn_implementation="flash_attention_2",
    )
    print(f"  Base model parameters: {model.num_parameters() / 1e9:.1f}B")

    # Load SFT LoRA adapter onto the base model
    print(f"Loading SFT adapter: {sft_adapter_path}")
    model = PeftModel.from_pretrained(model, str(sft_adapter_path))

    # Merge LoRA weights into base model and discard adapter structure.
    # After this call, model is a plain PreTrainedModel (not PeftModel)
    # with the SFT knowledge permanently fused into its weights.
    print("Merging SFT adapter into base weights...")
    model = model.merge_and_unload()
    print("  Merge complete. Model is now a standard PreTrainedModel.")

    return model, tokenizer


# ============================================================
# Dataset Loading
# ============================================================

def load_grpo_dataset(
    prompts_file: str | Path = GRPO_PROMPTS_FILE,
) -> Dataset:
    """Load the GRPO training prompt dataset.

    The GRPO dataset contains prompts only (no reference responses).
    The model generates its own responses during training, which are
    then scored by the reward function.

    Expected JSONL format (one JSON object per line):
        {
            "prompt": [{"role": "user", "content": "..."}],
            "headline": "...",
            "keywords": []
        }

    Columns:
        - prompt: Conversational format prompt (list of message dicts).
            This is the input that GRPOTrainer feeds to the model for
            generation.
        - headline: The news headline (used for reference, not by the
            current reward function).
        - keywords: List of required keywords (passed to the reward
            function via **kwargs).

    Args:
        prompts_file: Path to the GRPO prompts JSONL file.

    Returns:
        datasets.Dataset: The training prompt dataset.

    Raises:
        FileNotFoundError: If prompts_file does not exist.
    """
    prompts_file = Path(prompts_file)
    if not prompts_file.exists():
        raise FileNotFoundError(
            f"GRPO prompts file not found at {prompts_file}. "
            f"Please run the data pipeline first: "
            f"python -m data_preprocessing.pipeline --stage all"
        )

    dataset = load_dataset(
        "json",
        data_files=str(prompts_file),
        split="train",
    )

    # Inject /no_think system message to disable Qwen3 thinking mode.
    # This is more reliable than chat_template_kwargs={"enable_thinking": False},
    # which may not be correctly propagated by GRPOTrainer in all TRL versions.
    # The system message is prepended to each prompt's message list.
    NO_THINK_SYSTEM_MSG = {"role": "system", "content": "/no_think"}

    def _inject_no_think(example):
        prompt_messages = example["prompt"]
        if not prompt_messages or prompt_messages[0].get("role") != "system":
            example["prompt"] = [NO_THINK_SYSTEM_MSG] + prompt_messages
        return example

    dataset = dataset.map(_inject_no_think)

    print(f"Loaded GRPO dataset: {len(dataset)} prompts")
    print(f"  Columns: {dataset.column_names}")
    print(f"  /no_think system message injected into all prompts")

    return dataset


# ============================================================
# LoRA Configuration for GRPO Stage
# ============================================================

def build_grpo_lora_config(
    rank: int = 32,
    alpha: int = 64,
) -> LoraConfig:
    """Build LoRA configuration for the GRPO training stage.

    This LoRA is applied on top of the SFT-merged model. It uses a
    smaller rank than SFT (32 vs 64) because the GRPO stage is meant
    for fine-grained policy adjustment, not large-scale knowledge
    acquisition. The model already knows how to generate jokes from SFT;
    GRPO only needs to steer it toward higher-reward outputs.

    Using a smaller rank also:
    - Reduces VRAM usage (important since GRPO already holds
      num_generations * batch_size completions in memory).
    - Acts as implicit regularization against reward hacking.

    Args:
        rank: LoRA rank. Default 32.
        alpha: LoRA scaling factor. Usually 2 * rank. Default 64.

    Returns:
        LoraConfig: PEFT LoRA configuration for GRPO.
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"  GRPO LoRA: r={rank}, alpha={alpha}, dropout={config.lora_dropout}")
    return config


# ============================================================
# GRPO Training Configuration
# ============================================================

def build_grpo_config(
    num_generations: int = 16,
    max_completion_length: int = 256,
    batch_size: int = 8,
    grad_accum: int = 2,
    learning_rate: float = 5e-6,
    num_epochs: int = 2,
    beta: float = 0.04,
    temperature: float = 0.9,
    report_to: str = "wandb",
) -> GRPOConfig:
    """Build the GRPOConfig with hyperparameters tuned for our task.

    Key hyperparameter considerations for humor generation on A100 80GB:

    VRAM budget per step:
        Each step processes (batch_size * num_generations) completions.
        With batch_size=8, num_generations=16: 128 completions per step.
        The model (8B bf16 ~ 16GB) + activations for 128 completions
        + optimizer states fit within 80GB with gradient_checkpointing off.

    Loss type:
        We use "grpo" (classic GRPO loss) instead of the TRL default "dapo".
        GRPO normalizes loss per-sample then averages over the batch, which
        pairs well with the beta KL penalty. DAPO uses a different
        normalization scheme and is designed for beta=0 settings.

    Beta (KL penalty):
        Controls how far the policy can drift from the reference (SFT model).
        - beta=0: no constraint, risk of reward hacking.
        - beta=0.04: moderate constraint (recommended starting point).
        - beta>0.1: too conservative, policy barely updates.

    Temperature:
        Controls diversity of generated completions within each group.
        GRPO needs diversity to compute meaningful advantages.
        - temperature=0.9: good balance of diversity and quality.
        - temperature<0.7: too little diversity, advantages become noisy.

    Thinking mode:
        Disabled via chat_template_kwargs to prevent Qwen3 from generating
        <think>...</think> reasoning blocks before the joke.

    Args:
        num_generations: Number of completions per prompt (G). Default 16.
        max_completion_length: Max tokens per completion. Default 256.
        batch_size: Per-device prompt batch size. Default 8.
        grad_accum: Gradient accumulation steps. Default 2.
            Effective batch = batch_size * grad_accum = 16 prompts.
        learning_rate: RL-stage learning rate. Default 5e-6.
            Much smaller than SFT (2e-4) to prevent catastrophic updates.
        num_epochs: Number of training epochs. Default 2.
        beta: KL divergence penalty coefficient. Default 0.04.
        temperature: Sampling temperature for generation. Default 0.9.
        report_to: Experiment tracking backend. Default "wandb".

    Returns:
        GRPOConfig: Training configuration for GRPOTrainer.
    """
    effective_batch = batch_size * grad_accum
    print(f"  Effective prompt batch size: {batch_size} x {grad_accum} = {effective_batch}")
    print(f"  Completions per step: {batch_size} x {num_generations} = {batch_size * num_generations}")

    return GRPOConfig(
        output_dir=str(GRPO_CHECKPOINT_DIR),

        # --- GRPO Core ---
        num_generations=num_generations,
        max_completion_length=max_completion_length,
        temperature=temperature,
        top_p=0.95,
        beta=beta,
        loss_type="grpo",
        scale_rewards="group",

        # --- Training ---
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # --- Precision ---
        bf16=True,

        # --- Memory optimization ---
        # gradient_checkpointing=True saves VRAM by not storing intermediate
        # activations (recomputes them during backward pass), but makes
        # batch_size almost irrelevant to VRAM usage.
        # On A100 80GB with 8B model (~18GB static), there is plenty of
        # headroom. Disabling checkpointing lets activations scale with
        # batch_size, using more VRAM but speeding up training (~30% faster)
        # by avoiding recomputation.
        # If OOM occurs, switch back to True and reduce batch_size.
        gradient_checkpointing=False,

        # --- Qwen3 thinking mode ---
        # Thinking mode is primarily disabled via /no_think system message
        # injected in load_grpo_dataset(). This kwarg is kept as a secondary
        # safeguard, but may not be effective in all TRL versions.
        chat_template_kwargs={"enable_thinking": False},

        # --- Logging and Saving ---
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=5,
        log_completions=True,
        num_completions_to_print=0,  # log to wandb Table, suppress terminal output
        report_to=report_to,
        seed=42,
        run_name=f"grpo_vastai_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )


# ============================================================
# Main Training Flow
# ============================================================

def main():
    """GRPO training main entry point.

    Orchestrates the full GRPO training pipeline:
    1. Parse command-line arguments.
    2. Load base model + merge SFT adapter.
    3. Load GRPO prompt dataset.
    4. Build LoRA config (new adapter for GRPO).
    5. Build GRPOConfig (training hyperparameters).
    6. Build reward function (Phase 1: rule-based).
    7. Create GRPOTrainer and start training.
    8. Save final checkpoint.
    """
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-8B",
        help="Base model name or path (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--sft_adapter_path", type=str, default=str(SFT_ADAPTER_DIR),
        help="Path to the SFT LoRA adapter directory",
    )
    parser.add_argument(
        "--num_generations", type=int, default=16,
        help="Number of completions per prompt, G (default: 16)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Per-device prompt batch size (default: 8)",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=2,
        help="Gradient accumulation steps (default: 2)",
    )
    parser.add_argument(
        "--lr", type=float, default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2,
        help="Number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--beta", type=float, default=0.04,
        help="KL divergence penalty coefficient (default: 0.04)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.9,
        help="Sampling temperature for generation (default: 0.9)",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=32,
        help="LoRA rank for GRPO adapter (default: 32)",
    )
    parser.add_argument(
        "--max_completion_length", type=int, default=256,
        help="Max tokens per completion (default: 256)",
    )
    parser.add_argument(
        "--report_to", type=str, default="wandb",
        choices=["wandb", "tensorboard", "none"],
        help="Experiment tracking backend (default: wandb)",
    )
    parser.add_argument(
        "--use_humor_judge", action="store_true",
        help="Enable Phase 2a: use Gemini LLM-as-Judge for humor scoring. "
             "Requires GEMINI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--use_reward_model", action="store_true",
        help="Enable Phase 2b: use trained reward model for humor scoring. "
             "Mutually exclusive with --use_humor_judge.",
    )
    parser.add_argument(
        "--reward_model_path", type=str,
        default=str(PROJECT_ROOT / "checkpoints" / "reward_model" / "final"),
        help="Path to the trained reward model checkpoint "
             "(default: checkpoints/reward_model/final)",
    )
    args = parser.parse_args()

    if args.use_humor_judge and args.use_reward_model:
        parser.error("--use_humor_judge and --use_reward_model are mutually exclusive.")

    # ---- Step 1: Load Model (Base + Merge SFT Adapter) ----
    print("=" * 60)
    print("Step 1: Load model (base + merge SFT adapter)")
    print("=" * 60)
    model, tokenizer = load_sft_merged_model(
        base_model_name=args.model_name,
        sft_adapter_path=args.sft_adapter_path,
    )

    # ---- Step 2: Load Dataset ----
    print("\n" + "=" * 60)
    print("Step 2: Load GRPO prompt dataset")
    print("=" * 60)
    dataset = load_grpo_dataset()

    # ---- Step 3: Build LoRA Config for GRPO ----
    print("\n" + "=" * 60)
    print("Step 3: Configure GRPO LoRA")
    print("=" * 60)
    grpo_lora_config = build_grpo_lora_config(
        rank=args.lora_rank,
        alpha=args.lora_rank * 2,
    )

    # ---- Step 4: Build GRPOConfig ----
    print("\n" + "=" * 60)
    print("Step 4: Configure training hyperparameters")
    print("=" * 60)
    grpo_config = build_grpo_config(
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.num_epochs,
        beta=args.beta,
        temperature=args.temperature,
        report_to=args.report_to,
    )

    # ---- Step 5: Build Reward Function ----
    print("\n" + "=" * 60)
    if args.use_reward_model:
        print("Step 5: Build reward function (Phase 2b: rules + reward model)")
        print("=" * 60)
        batch_scorer = build_batch_reward_model_scorer(args.reward_model_path)
        reward_fn = build_reward_fn(batch_humor_scorer=batch_scorer)
        print("  Reward: format + keyword + relevance + humor (trained reward model)")
    elif args.use_humor_judge:
        print("Step 5: Build reward function (Phase 2a: rules + humor judge)")
        print("=" * 60)
        batch_scorer = build_batch_humor_scorer()
        reward_fn = build_reward_fn(batch_humor_scorer=batch_scorer)
        print("  Reward: format + keyword + relevance + humor (Gemini LLM-as-Judge)")
    else:
        print("Step 5: Build reward function (Phase 1: rule-based)")
        print("=" * 60)
        reward_fn = build_reward_fn()
        print("  Reward: format + keyword + relevance (no humor scorer)")

    # ---- Step 6: Create GRPOTrainer ----
    print("\n" + "=" * 60)
    print("Step 6: Create GRPOTrainer")
    print("=" * 60)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=grpo_lora_config,
    )
    print("  GRPOTrainer created successfully.")

    # ---- Step 7: Train ----
    print("\n" + "=" * 60)
    print("Step 7: Start GRPO training")
    print("=" * 60)
    trainer.train()

    # ---- Step 8: Save Final Model ----
    print("\n" + "=" * 60)
    print("Step 8: Save final model")
    print("=" * 60)
    final_dir = GRPO_CHECKPOINT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  Model saved to: {final_dir}")

    print("\nGRPO training complete.")


if __name__ == "__main__":
    main()
