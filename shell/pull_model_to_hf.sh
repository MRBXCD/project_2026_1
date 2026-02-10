parent_dir=$(cd "$(dirname "$0")/.."; pwd)
cd parent_dir
python -m scripts.upload_model \
    --adapter_path checkpoints/sft/final \
    --repo_id MRBSTUDIO/humor-sft-qwen3-8b \
    --private