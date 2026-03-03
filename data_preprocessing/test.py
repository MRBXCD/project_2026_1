import pandas as pd

file_path = '/home/mrb/projects/proj_2026_1/data/cfun/pure_jokes.csv'

try:
    # 加载 CSV 文件
    # 如果文件编码不是 UTF-8，可以尝试 encoding='utf-16' 或 'gbk'
    df = pd.read_csv(file_path)

    # 打印第 10 行内容
    # 注意：Python 索引从 0 开始，所以第 10 行的索引是 9
    if len(df) >= 10:
        print("第 10 行的数据如下：")
        print(df.iloc[9])
    else:
        print(f"文件行数不足，当前文件共有 {len(df)} 行数据。")

except FileNotFoundError:
    print("错误：未找到指定文件，请检查文件路径是否正确。")
except Exception as e:
    print(f"读取出错：{e}")