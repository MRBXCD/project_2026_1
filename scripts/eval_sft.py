"""
SFT 模型评估脚本
================

评估 SFT 训练后的模型，包含两个部分：

1. 生成对比评估 (--mode generate)
   - base 模型 vs SFT 模型并排生成笑话
   - 自动化指标: 格式合规、长度、关键词包含率
   - 输出人眼可读的对比表格并保存为 JSON

2. 通用能力回归测试 (--mode benchmark)
   - 使用 lm-evaluation-harness 运行标准 benchmark (MMLU, ARC 等)
   - 对比 base 模型和 SFT 模型的分数

用法:
    python -m scripts.eval_sft --mode generate
    python -m scripts.eval_sft --mode benchmark
    python -m scripts.eval_sft --mode all

依赖:
    - transformers, peft, torch
    - lm-eval (pip install lm-eval)
"""

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 路径常量
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_ADAPTER_DIR = PROJECT_ROOT / "checkpoints" / "sft" / "final"
GRPO_PROMPTS_FILE = PROJECT_ROOT / "data" / "grpo" / "grpo_prompts.jsonl"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "evaluation"


# ============================================================
# 模型加载
# ============================================================

def load_base_model(model_name: str):
    """加载未经 SFT 的原始基座模型。"""
    print(f"  加载 base 模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def load_sft_model(model_name: str, adapter_path: str | Path):
    """加载 SFT 训练后的模型 (base + LoRA adapter)。"""
    print(f"  加载 SFT 模型: {model_name} + {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return model, tokenizer


def unload_model(model, tokenizer):
    """释放模型显存，为加载下一个模型腾出空间。"""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  模型已释放")


# ============================================================
# 推理
# ============================================================

def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> list[str]:
    """用模型对一组 prompt 生成回复。"""
    responses = []

    for i, prompt_text in enumerate(prompts):
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
            )

        response = tokenizer.decode(
            outputs[0][prompt_len:], skip_special_tokens=True
        ).strip()
        responses.append(response)

        if (i + 1) % 10 == 0:
            print(f"    生成进度: {i + 1}/{len(prompts)}")

    return responses


# ============================================================
# 评估 prompt 加载
# ============================================================

def load_eval_prompts(n_samples: int = 20) -> list[dict]:
    """从 GRPO prompt 文件中加载评估 prompt。"""
    if not GRPO_PROMPTS_FILE.exists():
        raise FileNotFoundError(
            f"{GRPO_PROMPTS_FILE} 不存在，请先运行 data pipeline"
        )

    with open(GRPO_PROMPTS_FILE, "r", encoding="utf-8") as f:
        all_prompts = [json.loads(line) for line in f if line.strip()]

    # 分离 headline 和 keyword 子任务
    headline_prompts = [p for p in all_prompts if not p.get("keywords")]
    keyword_prompts = [p for p in all_prompts if p.get("keywords")]

    # 每种取 n_samples 条
    selected = headline_prompts[:n_samples] + keyword_prompts[:n_samples]

    print(f"  加载评估 prompt: {len(selected)} 条 "
          f"(headline: {min(n_samples, len(headline_prompts))}, "
          f"keyword: {min(n_samples, len(keyword_prompts))})")

    return selected


# ============================================================
# 自动化指标
# ============================================================

def compute_auto_metrics(prompts: list[dict], responses: list[str]) -> dict:
    """计算自动化指标。"""
    n = len(responses)
    if n == 0:
        return {}

    # 格式合规: 非空且长度 10-500
    format_pass = sum(
        1 for r in responses
        if r.strip() and 10 <= len(r.strip()) <= 500
    )

    # 长度统计
    lengths = [len(r.strip()) for r in responses if r.strip()]

    # 关键词包含率 (仅 keyword 子任务)
    kw_total = 0
    kw_hit = 0
    for prompt_data, resp in zip(prompts, responses):
        keywords = prompt_data.get("keywords", [])
        if keywords:
            kw_total += 1
            resp_lower = resp.lower()
            if all(kw.lower() in resp_lower for kw in keywords):
                kw_hit += 1

    metrics = {
        "total": n,
        "format_pass_rate": format_pass / n,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }

    if kw_total > 0:
        metrics["keyword_satisfaction_rate"] = kw_hit / kw_total
        metrics["keyword_total"] = kw_total

    return metrics


# ============================================================
# 评估 1: 生成对比
# ============================================================

def run_generation_eval(model_name: str, adapter_path: str, n_samples: int = 20):
    """运行生成对比评估：base 模型 vs SFT 模型。"""
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 加载评估 prompt
    eval_prompts = load_eval_prompts(n_samples)
    prompt_texts = [p["prompt"][0]["content"] for p in eval_prompts]

    # 2. Base 模型生成
    print("\n--- Base 模型生成 ---")
    base_model, base_tokenizer = load_base_model(model_name)
    base_responses = generate_responses(base_model, base_tokenizer, prompt_texts)
    unload_model(base_model, base_tokenizer)

    # 3. SFT 模型生成
    print("\n--- SFT 模型生成 ---")
    sft_model, sft_tokenizer = load_sft_model(model_name, adapter_path)
    sft_responses = generate_responses(sft_model, sft_tokenizer, prompt_texts)
    unload_model(sft_model, sft_tokenizer)

    # 4. 计算自动化指标
    base_metrics = compute_auto_metrics(eval_prompts, base_responses)
    sft_metrics = compute_auto_metrics(eval_prompts, sft_responses)

    # 5. 打印对比结果
    print("\n" + "=" * 70)
    print("自动化指标对比")
    print("=" * 70)
    print(f"{'指标':<30} {'Base':>15} {'SFT':>15}")
    print("-" * 60)
    for key in base_metrics:
        bv = base_metrics[key]
        sv = sft_metrics[key]
        if isinstance(bv, float):
            print(f"{key:<30} {bv:>15.3f} {sv:>15.3f}")
        else:
            print(f"{key:<30} {bv:>15} {sv:>15}")

    # 6. 打印生成样本对比 (前 5 条)
    print("\n" + "=" * 70)
    print("生成样本对比 (前 5 条)")
    print("=" * 70)
    for i in range(min(5, len(prompt_texts))):
        prompt_short = prompt_texts[i][:80] + "..." if len(prompt_texts[i]) > 80 else prompt_texts[i]
        print(f"\n[{i + 1}] Prompt: {prompt_short}")
        print(f"  Base: {base_responses[i][:200]}")
        print(f"  SFT:  {sft_responses[i][:200]}")

    # 7. 保存完整结果
    results = {
        "metrics": {"base": base_metrics, "sft": sft_metrics},
        "samples": [
            {
                "prompt": prompt_texts[i],
                "headline": eval_prompts[i].get("headline", ""),
                "keywords": eval_prompts[i].get("keywords", []),
                "base_response": base_responses[i],
                "sft_response": sft_responses[i],
            }
            for i in range(len(prompt_texts))
        ],
    }

    output_file = EVAL_OUTPUT_DIR / "generation_comparison.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存到: {output_file}")


# ============================================================
# 评估 2: 通用能力回归测试 (lm-evaluation-harness)
# ============================================================

def run_lm_eval(
    model_args: str,
    tasks: str,
    output_dir: str,
    num_fewshot: int,
    batch_size: str,
    device: str,
    limit: str | None = None,
):
    """Run lm-evaluation-harness CLI with shared arguments."""
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

    print(f"  命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_benchmark_eval(
    model_name: str,
    adapter_path: str,
    tasks: str = "mmlu",
    num_fewshot: int = 5,
    batch_size: str = "4",
    device: str = "cuda:0",
    sft_eval_mode: str = "peft",
    limit: str | None = None,
):
    """使用 lm-evaluation-harness 运行标准 benchmark。

    分别评估 base 模型和 SFT 模型，SFT 默认使用 PEFT adapter 直评。

    Args:
        model_name: 基座模型名
        adapter_path: SFT LoRA adapter 路径
        tasks: lm-eval 任务名，逗号分隔，例如 "mmlu,arc_challenge"
        num_fewshot: few-shot 样本数
        batch_size: lm-eval batch size，支持数字或 "auto"/"auto:N"
        device: 设备，例如 "cuda:0"
        sft_eval_mode: "peft" 或 "merge"
        limit: 可选，限制评估样本数（用于冒烟测试）
    """
    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 评估 Base 模型 ---
    print("\n--- 评估 Base 模型 ---")
    base_output_dir = str(EVAL_OUTPUT_DIR / "benchmark_base")
    base_model_args = (
        f"pretrained={model_name},dtype=bfloat16,trust_remote_code=True"
    )
    run_lm_eval(
        model_args=base_model_args,
        tasks=tasks,
        output_dir=base_output_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )

    # --- 评估 SFT 模型 ---
    print("\n--- 评估 SFT 模型 ---")
    sft_output_dir = str(EVAL_OUTPUT_DIR / "benchmark_sft")
    if sft_eval_mode == "peft":
        sft_model_args = (
            f"pretrained={model_name},peft={adapter_path},"
            "dtype=bfloat16,trust_remote_code=True"
        )
    elif sft_eval_mode == "merge":
        print("\n--- 合并 SFT adapter ---")
        merged_dir = EVAL_OUTPUT_DIR / "merged_sft_model"
        if not merged_dir.exists():
            print(f"  合并 adapter 到: {merged_dir}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="cpu"
            )
            model = PeftModel.from_pretrained(model, str(adapter_path))
            model = model.merge_and_unload()

            model.save_pretrained(str(merged_dir))
            tokenizer.save_pretrained(str(merged_dir))
            del model, tokenizer
            gc.collect()
            print("  合并完成")
        else:
            print(f"  已存在合并模型: {merged_dir}，跳过合并")
        sft_model_args = (
            f"pretrained={merged_dir},dtype=bfloat16,trust_remote_code=True"
        )
    else:
        raise ValueError(f"不支持的 sft_eval_mode: {sft_eval_mode}")

    run_lm_eval(
        model_args=sft_model_args,
        tasks=tasks,
        output_dir=sft_output_dir,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
    )

    print(f"\nBenchmark 结果已保存到:")
    print(f"  Base: {base_output_dir}")
    print(f"  SFT:  {sft_output_dir}")
    print("请手动对比两个目录下的结果文件。")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SFT 模型评估")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["generate", "benchmark", "all"],
                        help="评估模式")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-8B",
                        help="基座模型名")
    parser.add_argument("--adapter_path", type=str, default=str(SFT_ADAPTER_DIR),
                        help="SFT LoRA adapter 路径")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="生成对比的样本数 (默认 20)")
    parser.add_argument("--benchmark_tasks", type=str,
                        default="mmlu",
                        help="lm-eval 任务名，逗号分隔 (默认 mmlu)")
    parser.add_argument("--num_fewshot", type=int, default=5,
                        help="benchmark few-shot 数 (默认 5)")
    parser.add_argument("--benchmark_batch_size", type=str, default="4",
                        help='benchmark batch size，支持如 "4", "auto", "auto:4"')
    parser.add_argument("--benchmark_device", type=str, default="cuda:0",
                        help='benchmark 设备 (默认 "cuda:0")')
    parser.add_argument("--sft_eval_mode", type=str, default="peft",
                        choices=["peft", "merge"],
                        help="SFT benchmark 评估方式: peft(默认) 或 merge")
    parser.add_argument("--benchmark_limit", type=str, default=None,
                        help="可选，限制 benchmark 样本数用于冒烟测试，例如 20 或 0.1")
    args = parser.parse_args()

    if args.mode in ("generate", "all"):
        print("=" * 60)
        print("评估: 生成对比 (Base vs SFT)")
        print("=" * 60)
        run_generation_eval(args.model_name, args.adapter_path, args.n_samples)

    if args.mode in ("benchmark", "all"):
        print("\n" + "=" * 60)
        print("评估: 通用能力回归测试 (lm-evaluation-harness)")
        print("=" * 60)
        run_benchmark_eval(
            model_name=args.model_name,
            adapter_path=args.adapter_path,
            tasks=args.benchmark_tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.benchmark_batch_size,
            device=args.benchmark_device,
            sft_eval_mode=args.sft_eval_mode,
            limit=args.benchmark_limit,
        )


if __name__ == "__main__":
    main()
