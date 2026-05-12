# Bootstrap cosine confidence intervals

_Generated: 2026-05-12 09:21_

Stratified bootstrap on each donor's refused/complied harmful records (sample-with-replacement, preserving counts). 2000 iterations per pair. We report median + 95% CI.

**Interpretation guide**:
- `P(|cos|<0.05)`: probability that the bootstrap cos is near-zero (within ±0.05). High = behaviour directions are tight-clustered around orthogonal.
- `P(|cos|<0.2)`: probability of weak alignment (|cos|<0.2).
- `P(cos>0.5)`: probability of substantial positive alignment (red flag for our Finding 3 if non-trivial).

## 7B  (n_iter = 2000)

### Cross-pair `cos(v_A, v_B)` (the Finding 3 metric)

| pair | median | 95% CI | P(|cos|<0.05) | P(|cos|<0.2) | P(cos>0.5) |
|---|---|---|---|---|---|
| AZR-Coder-7B × Qwen2.5-Coder-7B | +0.502 | [+0.298, +0.627] | 0.0% | 0.2% | 50.7% |

### Positive control: `cos(v_M, Arditi_M)`

| model | median | 95% CI |
|---|---|---|
| AZR-Coder-7B | -0.253 | [-0.379, -0.084] |
| Qwen2.5-Coder-7B | -0.017 | [-0.154, +0.142] |

