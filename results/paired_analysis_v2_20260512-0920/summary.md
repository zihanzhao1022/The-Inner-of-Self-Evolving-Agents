# Paired analysis v2 — within-class baselines

_Generated: 2026-05-12 09:20_

Tightens v1's claim that the Arditi DiM refusal direction is a **prompt-type detector** (separating harmful from harmless) rather than a **behaviour predictor** (separating refuse from comply on a fixed prompt class).

Key v2 metrics:
- `AUC_within`: AUC of projection on harmful prompts only. If this ≈ 0.5, direction is informationally null at predicting behaviour.
- `null_AUC`: mean ± std of AUC for 20 random unit directions at the same layer. Sets the noise floor.
- `z_within_harm`: z-score of the Arditi direction's within-harmful AUC vs the null distribution.
- `cos(Arditi, behaviour-dir)`: cosine between Arditi DiM and the within-class refused-minus-comply direction (when ≥3 of each are available). A small cosine = behaviour mediator is NOT the Arditi direction.

## 7B

| model | L_best | n_clean | refuse-rate | harm-refuse | AUC_across | AUC_within | null AUC (mean±std) | z | cos(Arditi, beh-dir) | AUC_beh insample |
|---|---|---|---|---|---|---|---|---|---|---|
| AZR-Base-7B | L15 | 30 | 0.00 | 0.00 | — | — | — | — | — | — |
| AZR-Coder-7B | L8 | 48 | 0.17 | 0.21 | 0.359 | 0.192 | 0.52±0.18 | -1.8 | -0.288 | 0.951 |
| Qwen2.5-7B | L21 | 31 | 0.03 | 0.00 | 0.000 | — | — | — | — | — |
| Qwen2.5-Coder-7B | L9 | 47 | 0.19 | 0.18 | 0.450 | 0.457 | 0.54±0.16 | -0.5 | -0.015 | 0.969 |


## Reading the table

- If `AUC_within` is near `null AUC mean ± std` (low z), the Arditi refusal direction has no within-class predictive power — it's just a harmful-vs-harmless detector. This confirms v1's Finding 1.
- If `AUC_within` is significantly above the null (z > 2), the direction does carry some behavioural signal beyond prompt-type.
- `cos(Arditi, behaviour-dir)` is the cleanest test: if the within-class behaviour-driven direction is nearly orthogonal to the Arditi direction (|cos| < 0.3), they encode different things.
- `AUC_beh insample` is the upper bound: the behaviour direction was fit on these same prompts, so AUC here is inflated. Reported only as a sanity check that a behaviour direction exists at all.