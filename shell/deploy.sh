# 下载 SFT adapter (如果设置了 HF_USERNAME)
project_path = /workspace/proj_2026_1
HF_USERNAME=MRBSTUDIO
if [ -n "$HF_USERNAME" ]; then
    cd $project_path
    echo '  下载 SFT adapter...'
    mkdir -p $project_path/checkpoints/sft/final
    huggingface-cli download ${HF_USERNAME}/humor-sft-qwen3-8b \
        --local-dir $project_path/checkpoints/sft/final

else
    echo "  跳过 SFT adapter 下载 (未设置 HF_USERNAME)"
fi

echo "在容器内运行:"
echo "  # 数据处理"
echo "  python -m data_preprocessing.pipeline --stage all"
echo ""
echo "  # SFT 训练"
echo "  python -m scripts.train_sft --model_name /workspace/models/Qwen3-8B"
echo ""
echo "  # 评估"
echo "  python -m scripts.eval_sft --mode generate --model_name /workspace/models/Qwen3-8B"
