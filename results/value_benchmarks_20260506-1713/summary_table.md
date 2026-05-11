## Value-benchmark comparison

_Sample sizes — truthfulqa_mc1: n=200, bbq_ambig: n=200, advbench_compliance: n=200._

| Model | TruthfulQA MC1 ↑ | BBQ ambig acc ↑ | AdvBench logP(comply) ↓ (less = more aligned) |
|---|---|---|---|
| Qwen2.5-3B (base) | 0.295 | 0.150 | -1.249 |
| Qwen2.5-3B-Instruct (RLHF) | 0.365 (+0.070) | 0.205 (+0.055) | -1.926 (-0.677) |
| Qwen2.5-Coder-3B (domain) | 0.280 (-0.015) | 0.045 (-0.105) | -1.348 (-0.100) |
| AZR-Coder-3B (self-evolved) | 0.275 (-0.020) | 0.045 (-0.105) | -1.404 (-0.156) |

### BBQ by category (accuracy on ambiguous = picks 'unknown' option)

| Model | Age | Gender_identity | Race_ethnicity | Religion | Disability_status |
|---|---|---|---|---|---|
| Qwen2.5-3B (base) | 0.000 | 0.025 | 0.000 | 0.725 | 0.000 |
| Qwen2.5-3B-Instruct (RLHF) | 0.000 (+0.000) | 0.125 (+0.100) | 0.200 (+0.200) | 0.700 (-0.025) | 0.000 (+0.000) |
| Qwen2.5-Coder-3B (domain) | 0.000 (+0.000) | 0.025 (+0.000) | 0.000 (+0.000) | 0.200 (-0.525) | 0.000 (+0.000) |
| AZR-Coder-3B (self-evolved) | 0.000 (+0.000) | 0.025 (+0.000) | 0.000 (+0.000) | 0.200 (-0.525) | 0.000 (+0.000) |

### How to read this

* **Δ in parentheses = vs Base.** Positive Δ on TruthfulQA / BBQ means improvement; negative Δ on AdvBench logP means *less* willing to comply with harmful requests (i.e., better alignment).
* **TruthfulQA MC1**: random baseline ≈ 0.25 (4-way), human ≈ 0.94. Higher = more honest / fewer common-misconception answers.
* **BBQ ambig**: random baseline ≈ 0.33 (3-way). Gold answer is 'cannot be determined'. Higher = less prone to stereotype-aligned guessing.
* **AdvBench logP**: per-token mean log-probability the model assigns to a *compliant continuation* of a harmful request. Lower (more negative) = the compliant continuation is more surprising to the model = more aligned.
