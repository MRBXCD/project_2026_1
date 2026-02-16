# SFT vs Base Evaluation Comparison

- Base JSON: `/home/mrb/projects/proj_2026_1/evaluation/benchmark_base/Qwen__Qwen3-8B/results_2026-02-13T03-13-43.283297.json`
- SFT JSON: `/home/mrb/projects/proj_2026_1/evaluation/benchmark_sft/__content__project_2026_1__checkpoints__sft__final/results_2026-02-13T03-47-42.180523.json`

## Overall Summary

- Compared tasks: **62**
- SFT better on: **15** tasks
- Base better on: **43** tasks
- Tie: **4** tasks
- Mean delta (SFT-Base): **-1.97pp**
- Best gain: **  - medical_genetics (+4.00pp)**
- Largest drop: **  - moral_scenarios (-17.65pp)**

## Overall Tasks

| Task | Metric | Base | SFT | Delta (SFT-Base) | Better |
|---|---|---:|---:|---:|---|
| mmlu | `acc,none` | 74.84% | 72.11% | -2.73pp | Base |

## Task Groups

| Task | Metric | Base | SFT | Delta (SFT-Base) | Better |
|---|---|---:|---:|---:|---|
|  - humanities | `acc,none` | 66.16% | 61.83% | -4.34pp | Base |
|  - other | `acc,none` | 78.40% | 75.99% | -2.41pp | Base |
|  - social sciences | `acc,none` | 83.88% | 82.00% | -1.88pp | Base |
|  - stem | `acc,none` | 75.45% | 73.96% | -1.49pp | Base |

## Top 10 Improved Subtasks (SFT > Base)

| Task | Metric | Base | SFT | Delta (SFT-Base) | Better |
|---|---|---:|---:|---:|---|
|   - medical_genetics | `acc,none` | 83.00% | 87.00% | +4.00pp | SFT |
|   - econometrics | `acc,none` | 66.67% | 70.18% | +3.51pp | SFT |
|   - abstract_algebra | `acc,none` | 57.00% | 59.00% | +2.00pp | SFT |
|   - us_foreign_policy | `acc,none` | 84.00% | 86.00% | +2.00pp | SFT |
|   - college_chemistry | `acc,none` | 56.00% | 58.00% | +2.00pp | SFT |
|   - computer_security | `acc,none` | 81.00% | 83.00% | +2.00pp | SFT |
|   - management | `acc,none` | 87.38% | 89.32% | +1.94pp | SFT |
|   - jurisprudence | `acc,none` | 81.48% | 83.33% | +1.85pp | SFT |
|   - high_school_mathematics | `acc,none` | 54.07% | 55.56% | +1.48pp | SFT |
|   - professional_accounting | `acc,none` | 58.87% | 60.28% | +1.42pp | SFT |

## Top 10 Regressed Subtasks (Base > SFT)

| Task | Metric | Base | SFT | Delta (SFT-Base) | Better |
|---|---|---:|---:|---:|---|
|   - moral_scenarios | `acc,none` | 54.86% | 37.21% | -17.65pp | Base |
|   - global_facts | `acc,none` | 52.00% | 40.00% | -12.00pp | Base |
|   - machine_learning | `acc,none` | 61.61% | 51.79% | -9.82pp | Base |
|   - high_school_physics | `acc,none` | 72.85% | 64.24% | -8.61pp | Base |
|   - college_physics | `acc,none` | 64.71% | 58.82% | -5.88pp | Base |
|   - world_religions | `acc,none` | 87.72% | 81.87% | -5.85pp | Base |
|   - security_studies | `acc,none` | 79.18% | 73.47% | -5.71pp | Base |
|   - human_sexuality | `acc,none` | 83.21% | 77.86% | -5.34pp | Base |
|   - nutrition | `acc,none` | 81.05% | 75.82% | -5.23pp | Base |
|   - human_aging | `acc,none` | 75.78% | 71.30% | -4.48pp | Base |
