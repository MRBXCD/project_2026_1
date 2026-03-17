# Humor Generation — Evaluation Report

Generated: 2026-03-16 00:45

---

## 1. General Capability Benchmark

- base: `/content/project_2026_1/evaluation/benchmark_base/Qwen__Qwen3-8B/results_2026-03-15T22-59-33.053560.json`
- sft: `/content/project_2026_1/evaluation/benchmark_sft/MRBSTUDIO__Humor-Qwen3-8B-SFT/results_2026-03-15T23-31-37.500216.json`
- grpo: `/content/project_2026_1/evaluation/benchmark_grpo/__content__project_2026_1__evaluation__merged_grpo_model/results_2026-03-15T23-56-12.245479.json`

### Summary

- Reference model: **base**
- Compared tasks: **62**
- **sft** vs base: 37 improved, 21 degraded, mean delta +0.48pp
- **grpo** vs base: 39 improved, 17 degraded, mean delta +0.63pp

### Overall Tasks


| Task | Metric     | base   | sft    | Delta vs base | grpo   | Delta vs base |
| ---- | ---------- | ------ | ------ | ------------- | ------ | ------------- |
| mmlu | `acc,none` | 74.84% | 75.33% | +0.49pp       | 75.45% | +0.61pp       |


### Task Groups


| Task              | Metric     | base   | sft    | Delta vs base | grpo   | Delta vs base |
| ----------------- | ---------- | ------ | ------ | ------------- | ------ | ------------- |
| - humanities      | `acc,none` | 66.16% | 67.29% | +1.13pp       | 67.21% | +1.04pp       |
| - other           | `acc,none` | 78.40% | 77.95% | -0.45pp       | 78.31% | -0.10pp       |
| - social sciences | `acc,none` | 83.88% | 84.14% | +0.26pp       | 84.21% | +0.32pp       |
| - stem            | `acc,none` | 75.45% | 76.15% | +0.70pp       | 76.40% | +0.95pp       |


### Top 10 Improved Subtasks (grpo vs base)


| Task                      | Metric     | base   | sft    | Delta vs base | grpo   | Delta vs base |
| ------------------------- | ---------- | ------ | ------ | ------------- | ------ | ------------- |
| - international_law       | `acc,none` | 73.55% | 79.34% | +5.79pp       | 79.34% | +5.79pp       |
| - machine_learning        | `acc,none` | 61.61% | 66.07% | +4.46pp       | 66.96% | +5.36pp       |
| - logical_fallacies       | `acc,none` | 80.37% | 85.28% | +4.91pp       | 85.28% | +4.91pp       |
| - business_ethics         | `acc,none` | 75.00% | 79.00% | +4.00pp       | 79.00% | +4.00pp       |
| - astronomy               | `acc,none` | 90.13% | 92.11% | +1.97pp       | 94.08% | +3.95pp       |
| - elementary_mathematics  | `acc,none` | 78.57% | 82.28% | +3.70pp       | 81.75% | +3.17pp       |
| - econometrics            | `acc,none` | 66.67% | 68.42% | +1.75pp       | 69.30% | +2.63pp       |
| - philosophy              | `acc,none` | 78.14% | 80.39% | +2.25pp       | 80.71% | +2.57pp       |
| - clinical_knowledge      | `acc,none` | 78.49% | 80.00% | +1.51pp       | 80.75% | +2.26pp       |
| - professional_accounting | `acc,none` | 58.87% | 59.22% | +0.35pp       | 60.99% | +2.13pp       |


### Top 10 Regressed Subtasks (base vs grpo)


| Task                           | Metric     | base   | sft    | Delta vs base | grpo   | Delta vs base |
| ------------------------------ | ---------- | ------ | ------ | ------------- | ------ | ------------- |
| - global_facts                 | `acc,none` | 52.00% | 46.00% | -6.00pp       | 46.00% | -6.00pp       |
| - marketing                    | `acc,none` | 94.02% | 91.88% | -2.14pp       | 91.88% | -2.14pp       |
| - us_foreign_policy            | `acc,none` | 84.00% | 82.00% | -2.00pp       | 82.00% | -2.00pp       |
| - college_physics              | `acc,none` | 64.71% | 61.76% | -2.94pp       | 62.75% | -1.96pp       |
| - public_relations             | `acc,none` | 70.00% | 68.18% | -1.82pp       | 68.18% | -1.82pp       |
| - high_school_european_history | `acc,none` | 86.06% | 84.85% | -1.21pp       | 84.24% | -1.82pp       |
| - high_school_statistics       | `acc,none` | 77.78% | 75.93% | -1.85pp       | 76.39% | -1.39pp       |
| - miscellaneous                | `acc,none` | 85.82% | 84.29% | -1.53pp       | 84.55% | -1.28pp       |
| - virology                     | `acc,none` | 56.02% | 54.82% | -1.20pp       | 54.82% | -1.20pp       |
| - world_religions              | `acc,none` | 87.72% | 87.13% | -0.58pp       | 86.55% | -1.17pp       |


---

## 2. Task-Specific Automated Metrics


| Metric               | base   | sft    | grpo   |
| -------------------- | ------ | ------ | ------ |
| Samples              | 180    | 180    | 180    |
| Format Compliance    | 100.0% | 100.0% | 100.0% |
| Degeneracy Rate      | 0.0%   | 0.0%   | 0.0%   |
| Distinct-1           | 45.6%  | 44.7%  | 43.7%  |
| Distinct-2           | 82.0%  | 84.2%  | 82.9%  |
| Keyword Satisfaction | 100.0% | 100.0% | 100.0% |
| Avg Length           | 109.5  | 106.0  | 106.4  |
| Median Length        | 109.0  | 115.0  | 115.0  |


### Language: en


| Metric               | base   | sft    | grpo   |
| -------------------- | ------ | ------ | ------ |
| Samples              | 60     | 60     | 60     |
| Format Compliance    | 100.0% | 100.0% | 100.0% |
| Degeneracy Rate      | 0.0%   | 0.0%   | 0.0%   |
| Distinct-1           | 49.5%  | 45.4%  | 45.7%  |
| Distinct-2           | 85.1%  | 85.7%  | 85.3%  |
| Keyword Satisfaction | 100.0% | 100.0% | 100.0% |
| Avg Length           | 126.3  | 131.8  | 133.2  |
| Median Length        | 121.0  | 132.5  | 132.5  |


### Language: zh


| Metric               | base   | sft    | grpo   |
| -------------------- | ------ | ------ | ------ |
| Samples              | 60     | 60     | 60     |
| Format Compliance    | 100.0% | 100.0% | 100.0% |
| Degeneracy Rate      | 0.0%   | 0.0%   | 0.0%   |
| Distinct-1           | 35.5%  | 34.4%  | 34.4%  |
| Distinct-2           | 52.0%  | 50.8%  | 50.8%  |
| Keyword Satisfaction | 100.0% | 100.0% | 100.0% |
| Avg Length           | 62.5   | 59.7   | 57.6   |
| Median Length        | 63.0   | 59.0   | 57.5   |


### Language: es


| Metric               | base   | sft    | grpo   |
| -------------------- | ------ | ------ | ------ |
| Samples              | 60     | 60     | 60     |
| Format Compliance    | 100.0% | 100.0% | 100.0% |
| Degeneracy Rate      | 0.0%   | 0.0%   | 0.0%   |
| Distinct-1           | 43.8%  | 46.2%  | 43.5%  |
| Distinct-2           | 82.0%  | 86.0%  | 83.5%  |
| Keyword Satisfaction | 100.0% | 100.0% | 100.0% |
| Avg Length           | 139.7  | 126.5  | 128.3  |
| Median Length        | 132.5  | 127.0  | 127.0  |


---

## 3. LLM-as-Judge Pairwise Comparison

*LLM judge results not available. Run the llm_judge step first.*

---

## 4. Human Evaluation

Blind A/B samples exported (36 pairs). **Pending**: awaiting evaluator responses.

- CSV file: `/content/project_2026_1/evaluation/results/human_eval_samples.csv`
- Fill in the `your_verdict` column with A, B, or TIE
- Then re-run: `python -m evaluation.pipeline --steps report`

---

## Appendix

- Report generated: 2026-03-16 00:45:03
- Position bias mitigation: each pairwise comparison is run twice with A/B order swapped; only consistent verdicts are counted as wins.
- Keyword Satisfaction only applies to prompts with keyword constraints.
- Distinct-N is computed across all responses in the evaluation set.

