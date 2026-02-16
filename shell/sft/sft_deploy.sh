# download sft adapter to local from huggingface
project_path=$(cd "$(dirname "$0")/../.."; pwd)
echo $project_path
HF_USERNAME=MRBSTUDIO
if [ -n "$HF_USERNAME" ]; then
    cd $project_path
    echo '  下载 SFT adapter...'
    mkdir -p $project_path/checkpoints/sft/final
    hf download ${HF_USERNAME}/humor-qwen3-8b \
        --local-dir $project_path/checkpoints/

else
    echo "  跳过 SFT adapter 下载 (未设置 HF_USERNAME)"
fi

echo "在容器内运行:"
echo "  # 数据处理"
echo "  python -m data_preprocessing.pipeline --stage all"
echo ""
echo "  # SFT 训练"
echo "  python -m sft.train_sft"
echo ""
echo "  # 评估"
echo "  python -m sft.eval_sft --mode generate"
echo "  python -m sft.eval_sft --mode benchmark --benchmark_tasks mmlu,arc_challenge --num_fewshot 5 --sft_eval_mode peft"
