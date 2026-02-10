"""
上传 SFT adapter 到 HuggingFace Hub。

用法:
    # 首次使用需要登录
    huggingface-cli login

    # 上传 SFT adapter
    python -m scripts.upload_model --adapter_path checkpoints/sft/final --repo_id YOUR_USERNAME/humor-sft-qwen3-8b

依赖:
    pip install huggingface_hub
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="上传模型到 HuggingFace Hub")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="LoRA adapter 目录路径")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="HuggingFace repo ID，如 your-username/humor-sft-qwen3-8b")
    parser.add_argument("--private", action="store_true",
                        help="设为私有仓库")
    args = parser.parse_args()

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter 路径不存在: {adapter_path}")

    api = HfApi()

    # 创建 repo (如果不存在)
    api.create_repo(repo_id=args.repo_id, exist_ok=True, private=args.private)

    # 上传整个目录
    print(f"上传 {adapter_path} → {args.repo_id}")
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=args.repo_id,
        commit_message="Upload SFT LoRA adapter",
    )
    print(f"完成: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
