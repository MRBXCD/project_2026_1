# this is for sft evaluation on colab

## install dependencies
pip install -r requirements.txt

## download data
cp /content/drive/MyDrive/A-Research/project_2026_1/data_amended_03-03-26.zip ./data_amended_03-03-26.zip
unzip ./data_amended_03-03-26.zip
rm ./data_amended_03-03-26.zip

# pull sft and grpo adapter
sh ./shell/rl/rl_deploy.sh