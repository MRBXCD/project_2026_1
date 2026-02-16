# this is for sft evaluation on colab

## install dependencies
pip install -r requirements.txt

## download data
cp /content/drive/MyDrive/A-Research/project_2026_1/data.zip ./data.zip
unzip ./data.zip
rm ./data.zip

# pull sft and grpo adapter
sh ./shell/sft/sft_deploy.sh