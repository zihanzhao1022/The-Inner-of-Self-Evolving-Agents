# Paired analysis — projection × refusal behaviour

_Generated: 2026-05-11 15:10_

For each model we take the per-prompt raw residual at the best (pos, layer) — saved during the refusal-direction extraction — and project it onto the unit refusal direction (`harm_mean − safe_mean`). We then compare to whether the model's generation actually refused (regex prefix match).

A high AUC means the projection cleanly separates refusal vs compliance — i.e. the refusal direction is causally / behaviourally predictive of what the model will do.

## 3B

| model | L_best | n | refuse-rate | harmful r-rate | harmless r-rate | AUC | Pearson r | Welch p | mean(proj | refuse) | mean(proj | comply) |
|---|---|---|---|---|---|---|---|---|---|---|
| AZR-Coder-3B | L27 | 32 | 0.03 | 0.06 | 0.00 | 0.774 | 0.171 | nan | 8.19 | 4.85 |
| Qwen2.5-3B | L27 | 124 | 0.39 | 0.46 | 0.23 | 0.633 | 0.199 | 0.025 | 19.90 | 17.73 |
| Qwen2.5-3B-Instruct | L20 | 148 | 0.20 | 0.27 | 0.04 | 0.589 | 0.126 | 0.050 | -6.21 | -6.62 |
| Qwen2.5-Coder-3B | L27 | 35 | 0.23 | 0.25 | 0.18 | 0.694 | 0.268 | 0.124 | 15.54 | 11.22 |


## 7B

| model | L_best | n | refuse-rate | harmful r-rate | harmless r-rate | AUC | Pearson r | Welch p | mean(proj | refuse) | mean(proj | comply) |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | L21 | 31 | 0.03 | 0.00 | 0.09 | 0.000 | -0.363 | nan | -14.28 | 3.04 |

> 7B only has Qwen2.5-7B generations (n_refused=1) — AUC is not interpretable. Need to generate for Instruct / Coder / AZR-Base / AZR-Coder (~18 h GPU) before drawing 7B conclusions.

## Interpretation

### 1. Refusal direction is a prompt-type detector, NOT a behaviour predictor

Look at the boxplot (`fig_box_3B.png`). In every 3B model:

- **harmful vs harmless projections are clearly separated** (no surprise — that's what the probe was selected for, acc 0.92–0.95).
- **refuse vs comply within the same class barely differ** — boxes for harm/refuse and harm/comply overlap heavily, and so do harmless/refuse and harmless/comply.

This means the AUC 0.59–0.77 is almost entirely a base-rate artefact: harmful prompts are both more likely to have high projection AND more likely to be refused, so projection appears to "predict" refusal. But conditional on prompt class, projection carries essentially no behavioural information.

**Implication**: the cosine of `refusal direction × refusal direction` between models (which we report as our main "alignment is preserved" evidence) really measures **shared harm-detection geometry**, not "shared refusal mechanism". The behaviour layer is decoupled from this direction.

A proper behaviour-predicting direction would need a within-class contrast (refused vs complied on harmful only), not the across-class DiM Arditi-style. Worth a future experiment.

### 2. Self-evolving DOES change behaviour, even though it doesn't change representation

Comparing AZR-Coder-3B to its actual base Qwen2.5-Coder-3B on the same 50 prompts:

| metric | Qwen2.5-Coder-3B | AZR-Coder-3B | Δ |
|---|---|---|---|
| harmful refusal-rate (clean) | 25 % (6/24) | 5.6 % (1/18) | **-19.4 pp** |
| harmless refusal-rate (clean) | 18 % (2/11) | 0 % (0/14) | -18.2 pp |
| degenerate-loop rate | 30 % (15/50) | 36 % (18/50) | +6 pp |
| Procrustes disparity @ L35 | — | **0.0005** | — |
| Displacement angle L33→L35 | — | **1.0°** | — |

The representation is essentially unchanged (1.0° displacement, 0.0005 Procrustes), but the model **refuses 4× less often** on harmful prompts after self-evolving. Self-evolving RL on AZR's coding tasks evidently teaches the model to be more compliant / less helpful-refusal-y — but does so through a mechanism that **doesn't perturb the harmful-vs-harmless representational geometry**.

This is a small-sample observation (n_harmful_clean=18 vs 24), so it should be confirmed at 7B (currently blocked by missing AZR-Coder-7B / AZR-Base-7B generations). But it's the first concrete behavioural divergence we've documented; previous deltas were all representational and all near-zero.

### 3. Caveats

- Refusal regex is prefix-match on first 120 chars; soft refusals like "I understand your request, however I need more context" are counted as compliance. A few % misclassification likely.
- AZR-Coder-3B and Qwen2.5-Coder-3B both have ~30–36 % degenerate-loop rate; we exclude these from stats. Coder family clearly less stable at 512 max-tokens than the chat-tuned Instruct (0 % degen).
- The Welch p-values on Qwen2.5-3B (0.025) and 3B-Instruct (0.050) are technically significant for projection-distinguishes-refusal-from-compliance, but driven mostly by harmful-prompt over-representation in the refusal bucket. Not a within-class effect.

### Bottom line

The paired-analysis tightens the thesis as follows:

> Self-evolving RL is silent at the **representation level** (1.0° / 0.0005 Procrustes) but **not silent at the behaviour level** — on AZR-Coder-3B refusal rate on harmful prompts drops from 25 % to ≈6 %. The mechanism that drives this behavioural shift evidently routes around the harmful-vs-harmless geometry we measured. Future work: identify what direction(s) DO predict refuse vs comply within prompt class.
