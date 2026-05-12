# Cross-model behaviour-direction transfer

_Generated: 2026-05-12 09:21_

For each model with ≥3 refused AND ≥3 complied harmful prompts, compute the within-harmful behaviour direction `v_M = mean(raw|refuse) − mean(raw|comply)`. Then look at how these directions relate across models, and whether one model's direction predicts another's refusal.

## 7B

**Models with behaviour direction defined** (≥3 refused, ≥3 complied harmful):

- AZR-Coder-7B: refused=7, comply=26
- Qwen2.5-Coder-7B: refused=6, comply=27

**Models with too few refusals (recipient-only):**

- AZR-Base-7B: refused=0, comply=20
- Qwen2.5-7B: refused=0, comply=20

### Cross-cosine matrix `cos(v_A, v_B)`

| | AZR-Coder-7B | Qwen2.5-Coder-7B |
|---|---|---|
| AZR-Coder-7B | 1.00 | 0.66 |
| Qwen2.5-Coder-7B | 0.66 | 1.00 |

### Transfer AUC matrix (donor v predicting recipient's harmful refusal)

| donor \ recipient | AZR-Base-7B | AZR-Coder-7B | Qwen2.5-7B | Qwen2.5-Coder-7B |
|---|---|---|---|---|
| AZR-Coder-7B | — | 0.95 | — | 0.94 |
| Qwen2.5-Coder-7B | — | 0.96 | — | 0.97 |

### cos(Arditi DiM, behaviour direction) per donor — sanity check vs v1/v2

- AZR-Coder-7B: -0.288
- Qwen2.5-Coder-7B: -0.015


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