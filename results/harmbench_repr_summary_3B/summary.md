# HarmBench-paired representation summary — 3B

Timestamp: `20260518-0020`, n_layers=36, last unpruned L27.

## Per-model probe + Arditi refusal direction

| Model | probe best_acc | probe best_layer | ref_dir best_pos | ref_dir best_layer |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 0.9925 | L31 | 1 | L18 |
| Qwen2.5-3B-Instruct | 0.9950 | L29 | 1 | L20 |
| Qwen2.5-Coder-3B | 0.9925 | L30 | 2 | L21 |
| AZR-Coder-3B | 0.9925 | L30 | 2 | L12 |

## Cross-model refusal-direction cosine

Cosine evaluated at the donor's best (pos, layer) AND at pos=-1, L27.

| Pair | cos @ donor-best | cos @ pos=-1, L27 |
|---|---:|---:|
| Qwen2.5-3B ↔ Qwen2.5-3B-Instruct | 0.6459 | 0.5278 |
| Qwen2.5-3B ↔ Qwen2.5-Coder-3B | 0.4230 | 0.6248 |
| Qwen2.5-3B ↔ AZR-Coder-3B | 0.4080 | 0.6110 |
| Qwen2.5-3B-Instruct ↔ Qwen2.5-Coder-3B | 0.2859 | 0.4915 |
| Qwen2.5-3B-Instruct ↔ AZR-Coder-3B | 0.2535 | 0.4538 |
| Qwen2.5-Coder-3B ↔ AZR-Coder-3B | 0.9972 | 0.9821 |

## Centroid distance per axis (last unpruned layer L27)

| Axis | Pair | tag cos (harmful / harmless) | Procrustes resid | L33→L35 disp angle |
|---|---|---|---:|---:|
| RLHF | Qwen2.5-3B ↔ Qwen2.5-3B-Instruct | 0.9944 / 0.9880 | 0.000000 | 11.04° |
| domain | Qwen2.5-3B ↔ Qwen2.5-Coder-3B | 0.9756 / 0.9482 | 0.000000 | 49.44° |
| self_evolving | Qwen2.5-Coder-3B ↔ AZR-Coder-3B | 0.9999 / 0.9995 | 0.000000 | 0.72° |
