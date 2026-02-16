# download sft adapter to local from huggingface
project_path=$(cd "$(dirname "$0")/.."; pwd)
HF_USERNAME=MRBSTUDIO
if [ -n "$HF_USERNAME" ]; then
    cd $project_path
    echo '  下载 SFT adapter...'
    mkdir -p $project_path/checkpoints/sft/final
    hf download ${HF_USERNAME}/humor-qwen3-8b/sft \
        --local-dir $project_path/checkpoints/sft/final

else
    echo "  跳过 SFT adapter 下载 (未设置 HF_USERNAME)"
fi

# download grpo adapter to local from huggingface
project_path=$(cd "$(dirname "$0")/.."; pwd)
HF_USERNAME=MRBSTUDIO
if [ -n "$HF_USERNAME" ]; then
    cd $project_path
    echo '  下载 SFT adapter...'
    mkdir -p $project_path/checkpoints/grpo/final
    hf download ${HF_USERNAME}/humor-qwen3-8b/grpo \
        --local-dir $project_path/checkpoints/grpo/final

else
    echo "  跳过 GRPO adapter 下载 (未设置 HF_USERNAME)"
fi

