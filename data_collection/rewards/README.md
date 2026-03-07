# Reward Data

## Current (use these)

| File | Description |
|------|-------------|
| `k4_rewards_v3.jsonl` | **Primary.** 16,532 records re-judged with v3 all-continuous rubric (Reasoning Quality / Instruction Following / Communication Quality) by a fixed panel of Mistral-Large, DeepSeek-R1, Qwen-2.5-72B. No family overlap with k4 candidate models. |
| `k4_train_rewards.jsonl.gz` | Train split (from v2 data, to be regenerated from v3). |
| `k4_dev_rewards.jsonl.gz` | Dev split (from v2 data, to be regenerated from v3). |
| `k4_val_rewards.jsonl.gz` | Val split (from v2 data, to be regenerated from v3). |
| `k4_holdout_rewards.jsonl.gz` | Holdout split (from v2 data, to be regenerated from v3). |
| `calibration/` | Judge calibration data (200 samples). |

## Archive (do not use for new experiments)

| Folder | Description |
|--------|-------------|
| `archive/v2_binary_rubric/` | k4 rewards from v2 rubric (binary Logic/Constraint + continuous Utility). Biased judge panels (family-excluded rotation). |
| `archive/legacy_k10/` | Original k=10 model rewards from early experiments. |

## Rubric versions

- **v1** — `vote × confidence` per judge.
- **v2** — Binary Logic (0/1, w=0.5) + Binary Constraint (0/1, w=0.3) + Continuous Utility (0–1, w=0.2). 48% of rewards were exactly 1.0.
- **v3** — All-continuous: Reasoning Quality (0–1, w=0.4) + Instruction Following (0–1, w=0.3) + Communication Quality (0–1, w=0.3). Fixed unbiased judge panel.
