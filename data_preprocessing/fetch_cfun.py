# -------------------------------------------------
# 依赖
# -------------------------------------------------
# pip install datasets tqdm   # 只需运行一次
import csv
from datasets import load_dataset

# -------------------------------------------------
# 加载数据集（只下载一次，后续会缓存到 ~/.cache/huggingface/datasets）
# -------------------------------------------------
ds = load_dataset("ZhenghanYU/CFunSet", split="train")

# -------------------------------------------------
# 定义抽取完整笑话的函数
# -------------------------------------------------
# 这几条指令对应的记录里，笑话要么在 `output`（生成式），要么在 `input`（判定/解释式）。
INSTR_HUMOR_DETECT = (
    "以下是一段文本，请分析它是否具有幽默性。幽默性指该文本是否可能引起读者发笑，"
    "或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外）营造幽默效果。只需要输出“幽默”或“不幽默”。"
)
INSTR_HUMOR_REASON = (
    "请阅读以下文字，分析其幽默的原因。幽默性指该文本是否可能引起读者发笑，"
    "或通过语言技巧（如双关语、讽刺、夸张、荒诞或逻辑上的意外等等方式）营造幽默效果。请你写出以下文字幽默的原因："
)

def extract_joke(record):
    """返回该条记录中的完整笑话（若不存在则返回 None）。"""
    instr = record["instruction"]

    # ① 生成式：笑话在 output
    if (instr.startswith("生成一个主题为") or
        instr.startswith("生成一个关键词为") or
        instr.startswith("我将给你笑话的第一句话，请你生成整个笑话")):
        return record["output"].strip()

    # ② 判定/解释式：笑话在 input
    if instr == INSTR_HUMOR_DETECT or instr == INSTR_HUMOR_REASON:
        return record["input"].strip()

    return None   # 这条记录不符合 “有完整笑话” 的条件

# -------------------------------------------------
# 抽取所有笑话
# -------------------------------------------------
jokes = []
for rec in ds:
    j = extract_joke(rec)
    if j:                     # 只保留非空的笑话
        jokes.append(j)

print(f"共提取到 {len(jokes)} 条完整笑话")

# -------------------------------------------------
# 保存为 CSV（单列 `joke`）
# -------------------------------------------------
csv_path = "/home/mrb/projects/proj_2026_1/data/cfun/pure_jokes.csv"
with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["joke"])          # 表头
    for j in jokes:
        writer.writerow([j])

print(f"已写入 {csv_path}")