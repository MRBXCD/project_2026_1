# this is for sft evaluation on colab

## install dependencies
pip install -r requirements.txt

## download data
cp /content/drive/MyDrive/A-Research/project_2026_1/data/data_260311.zip ./data_260311.zip
unzip ./data_260311.zip
rm ./data_260311.zip

# pull sft and grpo adapter
sh ./shell/rl/rl_deploy.sh

uv run python -m evaluation.pipeline --steps all \
    --models base,grpo \
    --skip_sft_for_grpo