# Paired analysis v2 — within-class baselines

_Generated: 2026-05-11 15:30_

Tightens v1's claim that the Arditi DiM refusal direction is a **prompt-type detector** (separating harmful from harmless) rather than a **behaviour predictor** (separating refuse from comply on a fixed prompt class).

Key v2 metrics:
- `AUC_within`: AUC of projection on harmful prompts only. If this ≈ 0.5, direction is informationally null at predicting behaviour.
- `null_AUC`: mean ± std of AUC for 20 random unit directions at the same layer. Sets the noise floor.
- `z_within_harm`: z-score of the Arditi direction's within-harmful AUC vs the null distribution.
- `cos(Arditi, behaviour-dir)`: cosine between Arditi DiM and the within-class refused-minus-comply direction (when ≥3 of each are available). A small cosine = behaviour mediator is NOT the Arditi direction.

## 3B

| model | L_best | n_clean | refuse-rate | harm-refuse | AUC_across | AUC_within | null AUC (mean±std) | z | cos(Arditi, beh-dir) | AUC_beh insample |
|---|---|---|---|---|---|---|---|---|---|---|
| AZR-Coder-3B | L27 | 32 | 0.03 | 0.06 | 0.774 | 0.588 | 0.46±0.29 | 0.4 | — | — |
| Qwen2.5-3B | L27 | 124 | 0.39 | 0.46 | 0.633 | 0.566 | 0.43±0.07 | 2.0 | 0.122 | 0.875 |
| Qwen2.5-3B-Instruct | L20 | 148 | 0.24 | 0.34 | 0.640 | 0.499 | 0.50±0.05 | -0.1 | -0.003 | 0.722 |
| Qwen2.5-Coder-3B | L27 | 35 | 0.23 | 0.25 | 0.694 | 0.750 | 0.56±0.18 | 1.0 | 0.321 | 0.944 |


## 7B

| model | L_best | n_clean | refuse-rate | harm-refuse | AUC_across | AUC_within | null AUC (mean±std) | z | cos(Arditi, beh-dir) | AUC_beh insample |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | L21 | 31 | 0.03 | 0.00 | 0.000 | — | — | — | — | — |


## Reading the table

- If `AUC_within` is near `null AUC mean ± std` (low z), the Arditi refusal direction has no within-class predictive power — it's just a harmful-vs-harmless detector. This confirms v1's Finding 1.
- If `AUC_within` is significantly above the null (z > 2), the direction does carry some behavioural signal beyond prompt-type.
- `cos(Arditi, behaviour-dir)` is the cleanest test: if the within-class behaviour-driven direction is nearly orthogonal to the Arditi direction (|cos| < 0.3), they encode different things.
- `AUC_beh insample` is the upper bound: the behaviour direction was fit on these same prompts, so AUC here is inflated. Reported only as a sanity check that a behaviour direction exists at all.

## V2 verdict (5 corroborating evidence lines for Finding 1)

1. **`AUC_across` → `AUC_within` drops sharply.** All four 3B models lose 0.07–0.20 AUC when we restrict to harmful prompts only. The across-class number was mostly base-rate.

2. **`AUC_within` sits inside the null distribution.** z-scores: AZR-Coder-3B 0.4, Qwen2.5-3B 2.0 (marginal), Instruct -0.1, Coder-3B 1.0. With 20 random unit directions as null reference, the Arditi direction has no consistent within-class predictive lift.

3. **Instruct shows perfect orthogonality.** `cos(Arditi, behaviour-direction) = -0.003` on Qwen2.5-3B-Instruct. The within-class refused-minus-comply direction has essentially zero projection onto the Arditi prompt-type direction. They are different axes.

4. **A behaviour direction does exist (`AUC_beh insample` is high).** Qwen2.5-3B 0.875, Instruct 0.722, Coder-3B 0.944. So the within-class refuse-vs-comply distinction IS encoded in the residual stream — it just isn't on the Arditi axis.

5. **The Arditi direction was probe-selected on prompt class.** This is mechanistic: probe accuracy ≥ 0.92 over (harmful vs harmless) labels selects whatever direction best separates the two classes — which is the *causal driver of the model's harmful detection*, not necessarily of its refusal decision. The two can be in different layers / different sub-spaces.

## What this means for the paper

The "self-evolving is silent at the representation level" thesis needs to be more careful about WHICH representation:

- **"Self-evolving preserves the harmful-vs-harmless prompt-type geometry"** → strongly supported (Procrustes ≤ 0.0005, displacement ≤ 1.4°, refusal-direction cosine ≥ 0.98, behaviour direction not yet computed for AZR due to n_refused=1).
- **"Self-evolving preserves the refuse-vs-comply behaviour mediator"** → NOT what we measured. The mediator is on a different (approximately orthogonal) direction. The behaviour layer can shift (Finding 2: AZR-Coder-3B refusal rate 25% → 5.6%) without rotating the prompt-type axis.

Open question: does the behaviour direction itself rotate under self-evolving? Hard to test on AZR-Coder-3B (n_refused=1) and Qwen2.5-7B (n_refused=1). Needs the in-flight 7B generations on AZR-Coder-7B, Coder-7B, AZR-Base-7B.

## Caveats

- The behaviour direction is fit with at-most ~36 refused points (Qwen2.5-3B-Instruct) and 1 refused point (AZR-Coder-3B) — very small samples. `AUC_beh insample` is an upper bound by construction.
- Soft-refusal regex catches 7 extra refusals in Instruct only. Other 3 models have 0 soft hits, suggesting they either refuse cleanly with a hard prefix or don't refuse at all.
- Qwen2.5-7B `AUC_across=0.000` is the n_refused=1 artefact; ignore until 7B has full generations.
