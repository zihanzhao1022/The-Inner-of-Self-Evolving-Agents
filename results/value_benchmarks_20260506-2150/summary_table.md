## Value-benchmark comparison

_Sample sizes — truthfulqa_mc1: n=817, bbq_ambig: n=1000, advbench_compliance: n=520._

| Model | TruthfulQA MC1 ↑ | BBQ ambig acc ↑ | AdvBench logP(comply) ↓ (less = more aligned) |
|---|---|---|---|
| Qwen2.5-3B (base) | 0.290 | 0.040 | -1.210 |
| Qwen2.5-3B-Instruct (RLHF) | 0.379 (+0.089) | 0.123 (+0.083) | -1.886 (-0.676) |
| AZR-Coder-3B (self-evolved) | 0.278 (-0.012) | 0.010 (-0.030) | -1.368 (-0.158) |

### BBQ by category (accuracy on ambiguous = picks 'unknown' option)

| Model | Age | Gender_identity | Race_ethnicity | Religion | Disability_status |
|---|---|---|---|---|---|
| Qwen2.5-3B (base) | 0.000 | 0.005 | 0.000 | 0.195 | 0.000 |
| Qwen2.5-3B-Instruct (RLHF) | 0.010 (+0.010) | 0.090 (+0.085) | 0.230 (+0.230) | 0.275 (+0.080) | 0.010 (+0.010) |
| AZR-Coder-3B (self-evolved) | 0.000 (+0.000) | 0.005 (+0.000) | 0.000 (+0.000) | 0.045 (-0.150) | 0.000 (+0.000) |

### How to read this

* **Δ in parentheses = vs Base.** Positive Δ on TruthfulQA / BBQ means improvement; negative Δ on AdvBench logP means *less* willing to comply with harmful requests (i.e., better alignment).
* **TruthfulQA MC1**: random baseline ≈ 0.25 (4-way), human ≈ 0.94. Higher = more honest / fewer common-misconception answers.
* **BBQ ambig**: random baseline ≈ 0.33 (3-way). Gold answer is 'cannot be determined'. Higher = less prone to stereotype-aligned guessing.
* **AdvBench logP**: per-token mean log-probability the model assigns to a *compliant continuation* of a harmful request. Lower (more negative) = the compliant continuation is more surprising to the model = more aligned.
