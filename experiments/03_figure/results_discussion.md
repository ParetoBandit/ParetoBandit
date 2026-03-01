# Results Discussion: BanditGPT vs RouteLLM Evaluation

## Experimental Protocol

### 1. Ground-Truth Multi-Judge Rewards

Rewards are computed via `extract_reward()`: mean of (vote x confidence) across
the multi-judge panel.  This replaces the circular BT-proxy evaluation from
earlier drafts.  Under binary `raw_score`, 58-66% of prompts give identical
rewards for all K models; under mean(vote x confidence), this drops to < 1%,
providing much richer signal for routing evaluation.

### 2. Train-then-Freeze Protocol

BanditGPT trains on the dev set with oracle rewards from `extract_reward()`,
then is frozen for holdout evaluation.  RouteLLM is static (pre-trained on
100k LMSYS pairs) and evaluated on the same holdout.  Both routers are
frozen during evaluation, making the comparison directly interpretable.

### 3. Fair RouteLLM Comparison (K=2)

The K=2 portfolio uses Mixtral-8x7B + GPT-4-Turbo, which are models that
RouteLLM's MF router was trained on.  This is an **in-distribution** comparison,
unlike earlier drafts that used OOD model pairs (Llama-3.3-70B + o3).

RouteLLM's threshold tau is selected on a val subset of the dev set (85/15 split)
using the same cost-penalised objective as BanditGPT:
`argmax_tau E[reward - cost_penalty * cost]`.

### 4. Val / Holdout Separation

RouteLLM's threshold tau is tuned on val and frozen before holdout evaluation.
BanditGPT's hyperparameters (alpha, Corralling LR, n_effective) were fixed from
prior ablation studies and are not tuned on any evaluation data.  BanditGPT
trains on the full dev set.

### 5. Statistical Reporting

**Primary test:** Across-seeds paired t-test (df = N_SEEDS - 1 = 4).
The unit of observation is the per-seed final average reward.  With 5 seeds
the test requires Cohen's |d| >> 1 for significance at alpha = 0.05.

**Secondary (exploratory):** Paired bootstrap (n = holdout size).
Flagged as over-powered due to sequential state dependence.

---

## 1. K=2 BanditGPT vs RouteLLM (Figure 3)

### Methods compared
- **BanditGPT (warmup + Corralling)**: warmup priors from 80k RouteLLM battles,
  Corralling meta-learner, trained on dev set.
- **BanditGPT (tabula rasa)**: no priors, no Corralling, cold-start ablation.
- **RouteLLM (MF, tau from val, in-distribution)**: threshold selected on aligned
  objective.  The MF router was trained on data including Mixtral and GPT-4-Turbo.
- **Static routing**: always route to one model.
- **Random routing**.

### What the comparison tests
RouteLLM brings 100k pre-trained pairs (massive offline data advantage).
BanditGPT has ~1,121 dev prompts for online learning + warmup priors.
The comparison tests whether online adaptation with far less data can match
or beat a large pre-trained router on in-distribution models.

---

## 2. K=10 Multi-Model Pareto (Figure 4)

### Baselines
- **Oracle**: per-prompt argmax of ground-truth reward.
- **Tabula rasa**: BanditGPT without priors or Corralling (cold-start ablation).
- **Epsilon-greedy**: exploit empirical best model from training set.
- **Random, best-static**: standard reference points.

RouteLLM does not natively support K > 2 and is not included in this evaluation.

---

## 3. Key Improvements Over Earlier Drafts

- **No circular evaluation**: rewards are ground-truth multi-judge scores,
  not predictions from a BT-tracker that was also used as the learning signal.
- **In-distribution comparison**: RouteLLM evaluated on models it was trained on.
- **Cleaner protocol**: train-then-freeze eliminates the confound of online
  learning during evaluation (prequential protocol).
- **True cold-start ablation**: `prior_n_effective=0` with no Corralling.
