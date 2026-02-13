## pull github repo to colab
git clone https://github.com/MRBXCD/project_2026_1.git
cd project_2026_1

## install dependencies
pip install -r requirements.txt

## download data
cp /content/drive/MyDrive/A-Research/project_2026_1/data.zip ./project_2026_1
unzip ./project_2026_1/data.zip

# pull sft adapter
sh deploy.sh