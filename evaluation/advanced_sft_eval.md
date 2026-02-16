# Advanced SFT Evaluation Framework (Project Notes)

## 1. Background and Current Status

### 1.1 Project Goal
The SFT model is tuned for **humor generation**.
This implies a deliberate specialization tradeoff: improved in-domain behavior may cause some out-of-domain regression.

### 1.2 Current Benchmark Backend
We have migrated benchmark evaluation to **lm-evaluation-harness** (`lm-eval`) and currently run with `hf` backend in the project script.

### 1.3 Current Observations (Base vs SFT)
From current comparison results:
- MMLU overall regressed by around `-2.73pp`
- SFT better on some subtasks, but base better on more subtasks
- A few subtasks show large regression

Interpretation:
- This is **noticeable capability regression**, but not enough evidence to claim full catastrophic forgetting.
- More accurate label: **specialization with non-trivial general capability drift**.

---

## 2. Core Conclusions from Discussions

### 2.1 Is single-score evaluation suitable for humor?
Not sufficient.

Reason:
- Humor is highly subjective and culturally dependent.
- Single absolute score has high evaluator variance and poor calibration.

Conclusion:
- Score-based evaluation can still be used, but only as part of a **multi-signal evaluation framework**.

### 2.2 Does lm-eval have built-in humor benchmark tasks?
No clear built-in, dedicated "humor quality" benchmark found in current official task families.

But lm-eval supports:
- custom tasks (YAML/Python class)
- external task loading via `--include_path`
- validation/listing tools (`lm-eval validate`, `lm-eval ls`)

So the practical path is:
- use built-in tasks for safety/truthfulness/general regression
- build custom tasks for humor-specific quality

### 2.3 Do we need humans?
Humans are ideal but not strictly required at start.

If human eval is unavailable:
- run fully automated proxy evaluation first
- treat it as proxy signal, not final ground truth

---

## 3. Recommended Evaluation Architecture (Automation-First)

### 3.1 Layer A: Domain Target (Humor) - Custom/Proxy
Focus on humor-oriented metrics:
- novelty/diversity proxies
- coherence/consistency proxies
- pairwise preference proxy (model-as-judge)

### 3.2 Layer B: Safety and Risk Guardrails
Use existing lm-eval tasks where possible:
- `toxigen`
- `realtoxicityprompts`
- (optional) `truthfulqa` for factuality tendencies

### 3.3 Layer C: General Capability Regression
Keep tracking:
- `mmlu` (already in use)
- optional additional tasks (e.g., ARC variants)

---

## 4. Metric Design Principles

### 4.1 Novelty (do not optimize alone)
Possible automatic signals:
- `distinct-n` (higher preferred)
- `self-BLEU` (lower preferred)
- embedding similarity vs training outputs (lower preferred)

Risk:
- novelty-only optimization can reward nonsense.

### 4.2 Coherence
Operational definition:
- local consistency (sentence-to-sentence logic)
- global consistency (setup-punchline compatibility)
- contradiction control (NLI-assisted checks)

### 4.3 Safety
Track:
- harmful content rate
- unsafe instruction compliance rate
- refusal/redirect quality for risky prompts

---

## 5. Practical Auto-Eval Decision Rule (Suggested)

A model version is acceptable only if all conditions pass:

1. Humor preference win-rate (SFT vs Base) >= target threshold
2. Novelty improves without severe coherence collapse
3. Safety metrics do not degrade beyond tolerance
4. General benchmark regression remains within budget (e.g., MMLU drop <= 2~3pp)

This forms a release gate for iterative training.

---

## 6. Why Build Full Framework First (Before Fine Optimization)

This strategy is beneficial because:
- avoids blind hyperparameter tuning
- allows fast regression diagnosis
- creates reusable infrastructure for RL stage
- enables stable reward model / policy optimization validation later

---

## 7. Next-Stage Implementation Roadmap (High Level)

1. Standardize benchmark runner outputs (json + markdown summaries)
2. Add automated humor-proxy module (novelty/coherence/preference)
3. Integrate safety suite into regular pipeline
4. Define hard acceptance thresholds
5. Run end-to-end CI-like evaluation per model checkpoint
6. Enter targeted optimization loop (data, loss, prompt, reward)

---

## 8. Notes for Future RL Integration

The same framework can be reused for RL by:
- converting multi-metric outputs into reward components
- monitoring reward hacking risk (single-metric over-optimization)
- validating policy improvements against safety/generalization constraints

Goal in RL stage:
- improve in-domain humor quality while maintaining guardrails and acceptable general capability floor.
