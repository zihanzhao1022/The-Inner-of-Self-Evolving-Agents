## Value-benchmark comparison

_Sample sizes — truthfulqa_mc1: n=5, bbq_ambig: n=5, advbench_compliance: n=5._

| Model | TruthfulQA MC1 ↑ | BBQ ambig acc ↑ | AdvBench logP(comply) ↓ (less = more aligned) |
|---|---|---|---|
| Qwen2.5-3B (base) | 0.400 | 0.000 | -1.490 |
| Qwen2.5-3B-Instruct (RLHF) | 0.600 (+0.200) | 0.000 (+0.000) | -2.096 (-0.607) |
| AZR-Coder-3B (self-evolved) | 0.600 (+0.200) | 0.000 (+0.000) | -1.573 (-0.083) |

### BBQ by category (accuracy on ambiguous = picks 'unknown' option)

| Model | Age |
|---|---|
| Qwen2.5-3B (base) | 0.000 |
| Qwen2.5-3B-Instruct (RLHF) | 0.000 (+0.000) |
| AZR-Coder-3B (self-evolved) | 0.000 (+0.000) |

### How to read this

* **Δ in parentheses = vs Base.** Positive Δ on TruthfulQA / BBQ means improvement; negative Δ on AdvBench logP means *less* willing to comply with harmful requests (i.e., better alignment).
* **TruthfulQA MC1**: random baseline ≈ 0.25 (4-way), human ≈ 0.94. Higher = more honest / fewer common-misconception answers.
* **BBQ ambig**: random baseline ≈ 0.33 (3-way). Gold answer is 'cannot be determined'. Higher = less prone to stereotype-aligned guessing.
* **AdvBench logP**: per-token mean log-probability the model assigns to a *compliant continuation* of a harmful request. Lower (more negative) = the compliant continuation is more surprising to the model = more aligned.
