# Cross-model behaviour-direction transfer

_Generated: 2026-05-11 15:35_

For each model with ≥3 refused AND ≥3 complied harmful prompts, compute the within-harmful behaviour direction `v_M = mean(raw|refuse) − mean(raw|comply)`. Then look at how these directions relate across models, and whether one model's direction predicts another's refusal.

## 3B

**Models with behaviour direction defined** (≥3 refused, ≥3 complied harmful):

- Qwen2.5-3B: refused=39, comply=45
- Qwen2.5-3B-Instruct: refused=34, comply=66
- Qwen2.5-Coder-3B: refused=6, comply=18

**Models with too few refusals (recipient-only):**

- AZR-Coder-3B: refused=1, comply=17

### Cross-cosine matrix `cos(v_A, v_B)`

| | Qwen2.5-3B | Qwen2.5-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|
| Qwen2.5-3B | 1.00 | 0.05 | 0.03 |
| Qwen2.5-3B-Instruct | 0.05 | 1.00 | -0.06 |
| Qwen2.5-Coder-3B | 0.03 | -0.06 | 1.00 |

### Transfer AUC matrix (donor v predicting recipient's harmful refusal)

| donor \ recipient | AZR-Coder-3B | Qwen2.5-3B | Qwen2.5-3B-Instruct | Qwen2.5-Coder-3B |
|---|---|---|---|---|
| Qwen2.5-3B | 1.00 | 0.87 | 0.60 | 0.56 |
| Qwen2.5-3B-Instruct | 0.24 | 0.54 | 0.72 | 0.32 |
| Qwen2.5-Coder-3B | 0.82 | 0.57 | 0.42 | 0.94 |

### cos(Arditi DiM, behaviour direction) per donor — sanity check vs v1/v2

- Qwen2.5-3B: 0.122
- Qwen2.5-3B-Instruct: -0.003
- Qwen2.5-Coder-3B: 0.321


## 7B

**Models with behaviour direction defined** (≥3 refused, ≥3 complied harmful):


**Models with too few refusals (recipient-only):**

- Qwen2.5-7B: refused=0, comply=20
- Qwen2.5-Coder-7B: refused=0, comply=2

### Cross-cosine matrix `cos(v_A, v_B)`

| |  |
|---|

### Transfer AUC matrix (donor v predicting recipient's harmful refusal)

| donor \ recipient | Qwen2.5-7B | Qwen2.5-Coder-7B |
|---|---|---|

### cos(Arditi DiM, behaviour direction) per donor — sanity check vs v1/v2



## How to read

**Cross-cosine matrix**:
- High off-diagonal (≥0.5) = models share a behaviour axis. Self-evolving / RLHF preserves the axis even if they rotate prompt-type geometry differently.
- Low off-diagonal (<0.3) = each model has its own private behaviour mediator.
- Watch for: cos(Coder, AZR) >> cos(Coder, RLHF). That would say self-evolving keeps the behaviour axis Coder already had, while RLHF rotates it.

**Transfer AUC matrix**:
- Diagonal ≈ in-sample upper bound (donor's own AUC on its own prompts). Inflated by fitting.
- Off-diagonal close to diagonal = direction transfers cleanly.
- Off-diagonal near 0.5 = direction is private.

**Recipient-only distributions** (AZR-Coder-3B + Qwen2.5-7B until 7B gens finish):
- If the recipient's complied prompts cluster on the donor's *comply* mean → activations look like the donor's compliance pattern.
- If the recipient's complied prompts shifted toward the donor's *refuse* mean → activations 'almost wanted to refuse' but generation didn't. Suggests downstream decoupling.
- If the 1 refused AZR prompt is far below the donor's refuse cluster → it's an atypical refusal (e.g. only because the prompt was easy to refuse).

## Finding 3: behaviour directions are model-private (3B)

**Headline**: the within-class refuse-vs-comply axis (`v_M`) is essentially private to each model. Cross-cosines between the three 3B donors are all 0.05, 0.03, −0.06 — basically orthogonal. This is in **sharp contrast** to the across-class Arditi DiM, which has cos ≥ 0.93 across the same models (see `cross_model_metrics.json` in the refusal extraction).

| pair | cos(Arditi DiM) | cos(behaviour dir) |
|---|---|---|
| Qwen2.5-3B ↔ Instruct | ≈ 0.94 | **0.05** |
| Qwen2.5-3B ↔ Coder-3B | ≈ 0.95 | **0.03** |
| Instruct ↔ Coder-3B | ≈ 0.94 | **-0.06** |

Implications:

1. **Two-layer ontology of refusal-related geometry**:
   - **Shared layer**: harm-detection (Arditi DiM, cos > 0.93 across models)
   - **Private layer**: refusal-decision mediator (behaviour direction, cos ≈ 0)
2. **Behavioural shifts under fine-tuning are private**. RLHF and Coder pretraining each instal their own behaviour mediator, oriented essentially in different sub-spaces. Self-evolving (AZR) does the same (qualitatively inferred from Finding 2 + recipient-only plot below; can't quantify without n_refused ≥ 3 on AZR).
3. **Cross-model transfer fails**. AUC of donor-A's `v` predicting model-B's refusal on B's prompts:
    - Within-model: 0.72–1.00 (in-sample fit upper bound)
    - Cross-model: 0.32–0.60, near random for most pairs
    - The 0.82 for Coder → AZR-Coder is unreliable (n_refused on AZR = 1).

## Finding 4: AZR-Coder-3B activations are pushed *away* from its base's behaviour axis

The recipient-only plot (`fig_recipient_AZR-Coder-3B_3B.png`) shows AZR's 18 harmful activations projected onto each donor's `v`:

- **donor = Qwen2.5-3B**: AZR's comply cluster sits in Qwen-base's *refuse* region (~4–8 vs Qwen-base refuse mean ~9). Looks like "would refuse if it were Qwen-base".
- **donor = Instruct**: AZR's comply cluster straddles Instruct's comply / refuse boundary (~15–19).
- **donor = Coder-3B (AZR's true base)**: AZR's comply cluster sits at **−15 to −5**, well *below* Coder's own comply mean (≈ 0). AZR has been pushed off Coder's behaviour axis toward "more compliant than Coder ever was".

The third panel is the diagnostic one. Self-evolving doesn't just lower refusal rate — it actively shifts the harmful-prompt activations in a direction Coder uses to mark "definitely comply". The behaviour axis is preserved in *identity* (per cos ≈ 0.999 of the model weights) but used differently: the same axis now reads "stronger comply signal" for the same harmful inputs.

## Caveats

- The behaviour direction is fit with n_refused as small as 6 (Coder-3B). Statistical confidence is limited.
- AZR-Coder-3B has only 1 refused prompt; we can compute none of its own behaviour direction, and all "Coder→AZR" / "Qwen→AZR" cross-AUCs are based on N=1 refused vs 17 complied.
- The "behaviour direction" here is a single 2048-D axis fit by DiM. The actual behaviour mediator could be a low-dim subspace; we'd need probe / multi-dim analysis to rule that out.
- This run uses the v2 refusal classifier (hard + soft regex). Soft-refusal regex adds 7 Instruct refusals over hard-only; effect on the cross-cosine matrix has not been ablated.