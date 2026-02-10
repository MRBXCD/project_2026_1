#!/bin/bash
# ============================================================
# 远程服务器部署脚本
# ============================================================
#
# 在远程服务器上运行此脚本，完成全部部署:
#   1. 构建 Docker 镜像
#   2. 启动容器
#   3. 在容器内下载模型和数据
#
# 前置条件:
#   - 已安装 Docker + NVIDIA Container Toolkit
#   - 已 git clone 本仓库
#   - 已设置环境变量 (见下方)
#
# 用法:
#   # 设置环境变量
#   export HF_USERNAME="your-huggingface-username"
#   export HF_TOKEN="your-huggingface-token"          # 可选，私有 repo 需要
#   export GEMINI_API_KEY="your-gemini-api-key"        # 可选，合成数据需要
#
#   # 运行部署
#   bash deploy.sh
# ============================================================

set -e  # 遇到错误立即退出

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="humor_llm"
IMAGE_NAME="humor-llm-env:latest"

echo "============================================================"
echo "Step 1: 构建 Docker 镜像"
echo "============================================================"
cd "$PROJECT_DIR"
docker build -t "$IMAGE_NAME" .

echo ""
echo "============================================================"
echo "Step 2: 启动容器"
echo "============================================================"

# 如果同名容器已存在，先停止并删除
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "  发现已存在的容器 ${CONTAINER_NAME}，正在移除..."
    docker rm -f "$CONTAINER_NAME"
fi

docker run -it -d \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$PROJECT_DIR":/workspace \
    -p 8888:8888 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e GEMINI_API_KEY="${GEMINI_API_KEY:-}" \
    "$IMAGE_NAME"

echo "  容器已启动: $CONTAINER_NAME"

echo ""
echo "============================================================"
echo "Step 3: 在容器内下载模型"
echo "============================================================"

# 下载 base 模型 (Qwen3-8B)
docker exec "$CONTAINER_NAME" bash -c '
    echo "  下载 Qwen3-8B 基座模型..."
    huggingface-cli download Qwen/Qwen3-8B --local-dir /workspace/models/Qwen3-8B
'

# 下载 SFT adapter (如果设置了 HF_USERNAME)
if [ -n "$HF_USERNAME" ]; then
    docker exec "$CONTAINER_NAME" bash -c "
        echo '  下载 SFT adapter...'
        mkdir -p /workspace/checkpoints/sft/final
        huggingface-cli download ${HF_USERNAME}/humor-sft-qwen3-8b \
            --local-dir /workspace/checkpoints/sft/final
    "
else
    echo "  跳过 SFT adapter 下载 (未设置 HF_USERNAME)"
fi

echo ""
echo "============================================================"
echo "部署完成!"
echo "============================================================"
echo ""
echo "进入容器:"
echo "  docker exec -it $CONTAINER_NAME bash"
echo ""
echo "在容器内运行:"
echo "  # 数据处理"
echo "  python -m data_preprocessing.pipeline --stage all"
echo ""
echo "  # SFT 训练"
echo "  python -m scripts.train_sft --model_name /workspace/models/Qwen3-8B"
echo ""
echo "  # 评估"
echo "  python -m scripts.eval_sft --mode generate --model_name /workspace/models/Qwen3-8B"
