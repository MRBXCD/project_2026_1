import json
from lm_eval.utils import make_table
from pathlib import Path    

PROJECT_ROOT = Path(__file__).resolve().parent.parent

json_file_path = PROJECT_ROOT / "evaluation" / "benchmark_sft" / "__content__project_2026_1__checkpoints__sft__final/results_2026-02-13T03-47-42.180523.json"

with open(json_file_path, "r", encoding="utf-8") as f:
    results_dict = json.load(f)

table_string = make_table(results_dict)

print(table_string)

with open(PROJECT_ROOT / "evaluation" / "benchmark_sft" / "summary_table.md", "w", encoding="utf-8") as f:
    f.write(table_string)
    
print("表格已成功保存为 summary_table.md")