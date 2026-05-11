# Bootstrap cosine confidence intervals

_Generated: 2026-05-11 15:48_

Stratified bootstrap on each donor's refused/complied harmful records (sample-with-replacement, preserving counts). 2000 iterations per pair. We report median + 95% CI.

**Interpretation guide**:
- `P(|cos|<0.05)`: probability that the bootstrap cos is near-zero (within ±0.05). High = behaviour directions are tight-clustered around orthogonal.
- `P(|cos|<0.2)`: probability of weak alignment (|cos|<0.2).
- `P(cos>0.5)`: probability of substantial positive alignment (red flag for our Finding 3 if non-trivial).

## 3B  (n_iter = 2000)

### Cross-pair `cos(v_A, v_B)` (the Finding 3 metric)

| pair | median | 95% CI | P(|cos|<0.05) | P(|cos|<0.2) | P(cos>0.5) |
|---|---|---|---|---|---|
| Qwen2.5-3B × Qwen2.5-3B-Instruct | +0.037 | [-0.071, +0.153] | 52.5% | 100.0% | 0.0% |
| Qwen2.5-3B × Qwen2.5-Coder-3B | +0.025 | [-0.113, +0.154] | 50.6% | 99.9% | 0.0% |
| Qwen2.5-3B-Instruct × Qwen2.5-Coder-3B | -0.043 | [-0.103, +0.031] | 58.2% | 100.0% | 0.0% |

### Positive control: `cos(v_M, Arditi_M)`

| model | median | 95% CI |
|---|---|---|
| Qwen2.5-3B | +0.099 | [-0.108, +0.304] |
| Qwen2.5-3B-Instruct | -0.005 | [-0.148, +0.155] |
| Qwen2.5-Coder-3B | +0.276 | [+0.077, +0.419] |

