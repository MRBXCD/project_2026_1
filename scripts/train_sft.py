"""
SFT 训练脚本
============

使用 TRL 的 SFTTrainer 对 Qwen3-8B 进行 LoRA 微调。

TRL 的 SFTTrainer 已经封装了 SFT 训练的标准流程:
    - 自动应用 chat template 将 messages 格式转为模型输入
    - 自动处理 padding / truncation
    - 支持直接传入 LoRA config (内部调用 PEFT)
    - 继承 HuggingFace Trainer 的全部功能 (logging, saving, eval, etc.)

因此本脚本只需要做三件事:
    1. 加载模型 + tokenizer
    2. 加载数据集
    3. 配置超参数并启动训练

用法:
    python -m scripts.train_sft
    python -m scripts.train_sft --model_name Qwen/Qwen3-8B --epochs 3 --batch_size 4

依赖:
    - transformers, trl, peft, torch, datasets, accelerate
    - wandb (可选，用于实验追踪)
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ============================================================
# 路径常量
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_DATA_DIR = PROJECT_ROOT / "data" / "sft"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sft"


# ============================================================
# 模型加载
# ============================================================

def load_model_and_tokenizer(model_name: str) -> tuple:
    """加载基座模型和 tokenizer。

    Args:
        model_name: HuggingFace 模型名或本地路径，如 "Qwen/Qwen3-8B"

    Returns:
        tuple: (model, tokenizer)

    注意:
        - 使用 bf16 精度加载 (80GB GPU 完全够用)
        - 启用 FlashAttention 2 加速训练
        - padding_side="right" 是 SFT 训练的标准设置
          (因为 SFT 需要在序列右侧 padding，使得 labels 的
           padding 部分可以被正确忽略)
    """
    print(f"加载模型: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
    )
    # 确保 pad_token 存在 (部分模型没有默认的 pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    print(f"  模型参数量: {model.num_parameters() / 1e9:.1f}B")
    print(f"  dtype: {model.dtype}")

    return model, tokenizer


# ============================================================
# 数据加载
# ============================================================

def load_sft_dataset() -> dict:
    """加载 SFT 训练/验证数据集。

    从 data/sft/sft_train.jsonl 和 sft_val.jsonl 加载。
    这些文件由 data_preprocessing pipeline 生成，格式为:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

    Returns:
        DatasetDict: 包含 "train" 和 "validation" split
    """
    train_file = SFT_DATA_DIR / "sft_train.jsonl"
    val_file = SFT_DATA_DIR / "sft_val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"{train_file} 不存在。请先运行数据处理 pipeline:\n"
            f"  python -m data_preprocessing.pipeline --stage all"
        )

    data_files = {"train": str(train_file)}
    if val_file.exists():
        data_files["validation"] = str(val_file)
    else:
        print(f"  警告: {val_file} 不存在，将不使用验证集")

    dataset = load_dataset("json", data_files=data_files)

    for split_name, ds in dataset.items():
        print(f"  {split_name}: {len(ds)} 条")

    return dataset


# ============================================================
# LoRA 配置
# ============================================================

def build_lora_config(rank: int = 64, alpha: int = 128) -> LoraConfig:
    """构建 LoRA 配置。

    Args:
        rank: LoRA 秩。越大表达能力越强但显存开销越大。
              8B 模型推荐 32-64。
        alpha: LoRA 缩放因子。通常设为 2 * rank。

    Returns:
        LoraConfig: PEFT LoRA 配置对象

    注意:
        target_modules 覆盖 Qwen3 的全部线性层 (attention + FFN)，
        这是 LoRA 微调的标准做法。不包含 lm_head (输出层) 和
        embedding 层 — 这些层参数量大，LoRA 化性价比低。
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    print(f"  LoRA: r={rank}, alpha={alpha}, dropout={config.lora_dropout}")
    return config


# ============================================================
# 训练配置
# ============================================================

def build_training_args(
    lora_config: LoraConfig,
    epochs: int = 3,
    batch_size: int = 4,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
    report_to: str = "wandb",
) -> SFTConfig:
    """构建 SFT 训练配置。

    Args:
        lora_config: LoRA 配置 (会传给 SFTConfig.peft_config)
        epochs: 训练轮数。笑话数据量不大 (~50K)，3 轮通常够用。
        batch_size: 每 GPU 的 batch size。
        grad_accum: 梯度累积步数。有效 batch = batch_size * grad_accum。
        learning_rate: LoRA 学习率。通常比全量微调大 (2e-4 vs 2e-5)。
        max_seq_length: 最大序列长度。笑话通常短，512 足够。
        report_to: 实验追踪工具 ("wandb" / "tensorboard" / "none")。

    Returns:
        SFTConfig: TRL 训练配置对象
    """
    effective_batch = batch_size * grad_accum
    print(f"  有效 batch size: {batch_size} x {grad_accum} = {effective_batch}")

    return SFTConfig(
        output_dir=str(CHECKPOINT_DIR),

        # --- 训练超参 ---
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,

        # --- 精度 ---
        bf16=True,

        # --- 序列长度 ---
        max_seq_length=max_seq_length,

        # --- 日志与保存 ---
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        # --- 优化 ---
        gradient_checkpointing=True,        # 用时间换显存，8B 模型建议开启

        # --- PEFT ---
        peft_config=lora_config,

        # --- 其他 ---
        report_to=report_to,
        seed=42,
    )


# ============================================================
# 主函数
# ============================================================

def main():
    """SFT 训练主流程。"""
    parser = argparse.ArgumentParser(description="SFT 训练脚本")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="基座模型名称或路径")
    parser.add_argument("--epochs", type=int, default=3,
                        help="训练轮数 (默认 3)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="每 GPU batch size (默认 4)")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="梯度累积步数 (默认 4)")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="学习率 (默认 2e-4)")
    parser.add_argument("--lora_rank", type=int, default=64,
                        help="LoRA 秩 (默认 64)")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="最大序列长度 (默认 512)")
    parser.add_argument("--report_to", type=str, default="wandb",
                        choices=["wandb", "tensorboard", "none"],
                        help="实验追踪工具 (默认 wandb)")
    args = parser.parse_args()

    # 1. 加载模型和 tokenizer
    print("=" * 60)
    print("Step 1: 加载模型")
    print("=" * 60)
    model, tokenizer = load_model_and_tokenizer(args.model_name)

    # 2. 加载数据集
    print("\n" + "=" * 60)
    print("Step 2: 加载数据集")
    print("=" * 60)
    dataset = load_sft_dataset()

    # 3. 构建 LoRA 配置
    print("\n" + "=" * 60)
    print("Step 3: 配置 LoRA")
    print("=" * 60)
    lora_config = build_lora_config(rank=args.lora_rank, alpha=args.lora_rank * 2)

    # 4. 构建训练配置
    print("\n" + "=" * 60)
    print("Step 4: 配置训练参数")
    print("=" * 60)
    training_args = build_training_args(
        lora_config=lora_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        max_seq_length=args.max_seq_length,
        report_to=args.report_to,
    )

    # 5. 创建 SFTTrainer 并启动训练
    print("\n" + "=" * 60)
    print("Step 5: 开始训练")
    print("=" * 60)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        processing_class=tokenizer,
    )

    trainer.train()

    # 6. 保存最终模型 (LoRA adapter + tokenizer)
    print("\n" + "=" * 60)
    print("Step 6: 保存模型")
    print("=" * 60)
    final_dir = CHECKPOINT_DIR / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"  模型已保存到: {final_dir}")

    print("\nSFT 训练完成。")


if __name__ == "__main__":
    main()
