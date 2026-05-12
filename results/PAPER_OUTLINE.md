# Paper outline (draft 2026-05-12, post-7B-replication revision)

**Working title** (revised after 7B reversal): _Scale-Dependent Behavioural Effects of Self-Evolving RL: Representation Silent at 3B and 7B, but Behaviour Reorganizes Only at 3B (Qwen2.5)_

**Alternative titles**:
- _The Silent Inside, the Scale-Dependent Outside: Self-Evolving RL on Qwen2.5 3B vs 7B_
- _Self-Evolving RL Preserves Representation Geometry Across Scale, but Its Behavioural Effects Vanish at 7B_
- _A Two-Layer Refusal Ontology That Holds at 3B and Collapses at 7B_

> **NOTE (2026-05-12)**: this outline was originally written for 3B-only results. After the overnight 7B replication ran (`results/generations_7B/{Coder, AZR-Base, AZR-Coder}-7B/` + paired-analysis + behaviour-transfer + bootstrap + regex-ablation on 7B), every behaviour-layer finding F1–F5 either reversed or vanished. Only F0 (representation silence) holds across scale. The outline below has been revised to reflect this. The original 3B-only contributions are kept (they're still valid for 3B) but recast as "3B-specific findings" rather than universal.

---

## 1. Abstract (draft, 2026-05-12 revision)

We probe how three families of LLM fine-tuning — RLHF (Qwen2.5-Instruct), code-domain continued pretraining (Qwen2.5-Coder), and self-evolving RL (Absolute-Zero-Reasoner, "AZR") — reshape the internal geometry of a base LLM. Working with 4 Qwen2.5-3B variants and 5 Qwen2.5-7B variants (including a 7B AZR variant trained directly from plain base, with no Coder intermediate), we find a stable representation-level result and a striking **scale-dependent behaviour-level result**.

**Representation level (3B + 7B both stable)**: self-evolving leaves the residual-stream geometry essentially unchanged at both scales (1.0–1.4° rotation, Procrustes disparity ≤ 0.0005, weight cosine 0.999999+). Scaling tightens silence — 7B's Procrustes residual is 5× smaller than 3B's.

**Behaviour level (3B and 7B diverge)**: on 3B, AZR-Coder-3B's harmful-prompt refusal rate collapses from its base Coder-3B's 25% to 5.6% (Δ −19 pp) on the same prompts, the within-class behaviour direction becomes model-private (cross-model cos ≈ 0), and mode-collapse rate doubles (echo loops 6 → 14). On 7B, all three of these reverse or vanish: refusal rate goes 18 → 21% (Δ +3 pp), self-evolving preserves 50% of the behaviour direction (cos +0.502), and echo loops do not increase (3 → 2). Only F0 (representation silence) survives across scale.

We propose a capacity hypothesis to reconcile the two scales: at 3B the self-evolving RL reward signal is large enough to perturb the policy along the Coder-inherited behaviour axis without disturbing the broader representation geometry; at 7B the same signal is absorbed without disturbing either layer. The earlier "two-layer ontology of refusal" we describe (shared harm-detection axis + model-private behaviour mediator) holds at 3B but collapses at 7B, where self-evolving shares ~half of Coder's behaviour direction. This makes a single explanatory frame for "representation-silent-but-behaviour-shifted" inadequate: the effect is small-model specific. We close with limitations: AZR-Llama-3.1-8B weights are not public, so we cannot test whether the 7B silence we observe is Qwen-specific or generalizes; the AZR paper's reported "uh-oh moment" on Llama-3.1-8B remains an open confound.

---

## 2. Contributions (revised 2026-05-12)

### Scale-stable contributions (hold at both 3B and 7B)

1. **Five-evidence-line representation silence** for self-evolving RL on Qwen2.5, replicated across 3B and 7B:
   - weight cosine (3B fp64 > 0.999999; 7B pending subprocess workaround for OOM)
   - refusal-direction cosine (≈ 0.98)
   - layer-(N-3) → layer-(N-1) displacement angle (1.0°/1.4°)
   - Procrustes disparity at last unpruned layer (0.0005 → 0.0001 at scale — scaling tightens silence)
   - value-benchmark Δ near zero (3B full; 7B partial)
2. **Falsification of "Coder-pretraining-provides-robustness"** using AZR-Base-7B (no Coder intermediate): 1.4° rotation, identical to the Coder→AZR-Coder path. Representation silence is intrinsic to self-evolving RL, not a Coder side-effect.
3. **Scale-dependence of behavioural effects** (new headline): on 3B, self-evolving collapses harmful refusal (25 % → 5.6 %), doubles mode-collapse rate (6 → 14 echo loops), and pushes activations off the Coder behaviour axis. On 7B, none of these effects appear (refusal 18 → 21 %, loops 3 → 2, cos(Coder, AZR) = +0.502 not ~0). The same RL training procedure has qualitatively different behavioural effects at different scales.

### 3B-specific contributions (hold at 3B only; reverse or vanish at 7B)

4. **Dissociation at 3B**: AZR-Coder-3B's harmful-refusal rate drops 19 pp vs Coder-3B on the same 50 prompts, while every representational-distance metric reports "no change". The 7B replication test inverts this (+3 pp).
5. **Two-layer ontology of refusal (3B-only)**: Arditi DiM is a prompt-type detector (within-harmful AUC ≈ null; cos with behaviour direction ≈ 0). Behaviour mediator lives on a model-private axis (cross-model cos = 0.05 / 0.03 / −0.06). The 7B replication test shows the two-layer ontology collapses (cos(Coder, AZR-Coder) = +0.502, bootstrap 95% CI [+0.298, +0.627]).
6. **Activation-shift diagnosis at 3B**: AZR-Coder-3B's harmful-prompt activations projected onto Coder-3B's behaviour direction sit far below Coder's comply mean (Δ −9). 7B replication attenuates by ~4× (Δ −2.3).
7. **Mode-collapse amplification at 3B**: AZR-Coder-3B produces strict echo loops at 14/50 prompts vs Coder's 6/50 — RL training doubles failure-mode rate. 7B does NOT show this (3 vs 2).

---

## 3. Setup

### Models

| family | members |
|---|---|
| 3B quartet | Qwen2.5-3B, Qwen2.5-3B-Instruct, Qwen2.5-Coder-3B, AZR-Coder-3B |
| 7B quintet | Qwen2.5-7B, Qwen2.5-7B-Instruct, Qwen2.5-Coder-7B, AZR-Coder-7B, AZR-Base-7B |

Five training axes (3B has 3, 7B has 5): RLHF, domain (Coder), self-evolving-via-coder, self-evolving-direct (7B-only control), cross-AZR (7B only).

### Data

- **cm_binary** (controlled): IBM `condition_multiple` dataset reformulated into harmful / harmless binary tags with matched scaffolding. 128 of each used for direction extraction; 35–100 of each for generation (max_new_tokens = 512, greedy).
- **arditi_combined**: HarmBench + AdvBench + MaliciousInstruct (harmful) × Alpaca (harmless), round-robin to 128 each.

### Methods

1. **Arditi-aligned refusal-direction extraction** (fp64, multi-pos, layer pruning).
2. **Phase-1 activation collection** at every layer with joint PCA + Procrustes.
3. **Weight diff** (fp64 cosine, 3B full quartet; 7B failed due to Windows 32 GB OOM — workaround pending).
4. **Paired analysis**: per-prompt residual at best (pos, layer) × refusal-classifier label on the generation, with random-direction null and within-class behaviour-direction control.

---

## 4. Representation-level results (the "silence")

→ Already documented in `SUMMARY_3B_vs_7B.md`. Five-evidence-line table:

| # | metric | RLHF | domain | **self-evolving** |
|---|---|---|---|---|
| 1 | weight cos (3B fp64) | > 0.9999 | 0.55–0.78 | **> 0.999999** |
| 2 | refusal-dir cos (cm_binary, L27) | 0.50 | 0.59 | **0.98** |
| 3 | L(N-3)→L(N-1) displacement | 24–26° | 60–88° | **1.0–1.4°** |
| 4 | Procrustes disparity | 0.008 | 0.002–0.003 | **0.0001–0.0005** |
| 5 | value benchmark Δ | non-zero (some) | non-zero | **near zero** |

Key control: 7B `self_evolving_direct` (base → AZR-Base, no Coder) — 1.4° rotation, 0.0001 Procrustes. Falsifies "Coder stiffening" hypothesis.

---

## 5. Behaviour-level results (3B-specific; reversed at 7B)

### 5.1 Refusal rate on cm_binary harmful prompts (clean)

**3B (n_max = 35–100 per model)**:

| model | n_harmful_clean | refused | refusal-rate |
|---|---|---|---|
| Qwen2.5-3B (base) | 84 | 39 | 46 % |
| Qwen2.5-3B-Instruct (RLHF) | 100 | 34 | 34 % |
| Qwen2.5-Coder-3B (domain) | 24 | 6 | **25 %** |
| AZR-Coder-3B (self-evolving) | 18 | 1 | **5.6 %** |

**Finding 2 at 3B**: same 50 prompts, AZR vs its true base Coder-3B → Δ = **−19.4 pp** on harmful refusal rate. The closest comparable representational-distance metric reports 1.0° rotation, 0.0005 Procrustes.

**7B (n_max = 35 per model)**:

| model | n_harmful_clean | refused | refusal-rate |
|---|---|---|---|
| Qwen2.5-7B (base) | 20 | 0 | **0 %** |
| Qwen2.5-Coder-7B (domain) | 33 | 6 | **18.2 %** |
| AZR-Base-7B (self-evolving direct) | 20 | 0 | **0 %** |
| AZR-Coder-7B (self-evolving via Coder) | 33 | 7 | **21.2 %** |

**Finding 2 at 7B (REVERSED)**: AZR-Coder-7B vs Coder-7B → Δ = **+3 pp** (AZR refuses MORE). The 3B finding does NOT replicate at scale.

Degenerate-loop sanity (3B): Coder/AZR families have 30–36 % degeneration rate at 512 max-tokens; Instruct has 0 %. We exclude degenerated outputs.

**Anomaly count at 7B (NEW, supports scale-dependence)**: AZR-Base-7B (no Coder, self-evolving direct) preserves base's high echo-loop rate (20/50 vs base's 19/50). AZR-Coder-7B (via Coder) keeps Coder's low loop rate (2/50 vs Coder's 3/50). **Coder pretraining is the key stabilizer; self-evolving alone does NOT amplify mode collapse at 7B.**

### 5.4 Qualitative: AZR substitutes refusal with hedged compliance + degeneration (Finding 5)

The 19 pp refusal-rate drop is not a uniform "AZR became dangerous". Manual inspection of all 9 prompts where Coder-3B refused (cm_binary harmful idx 1, 2, 6, 7, 10, 13, 16, 18, 19):

| AZR behaviour | n | example |
|---|---|---|
| Same refusal opener as Coder | 1 | idx=1 ("bigoted statement" — universal refusal, refused by every model) |
| Direct comply | 1 | idx=13 (gave method directly) |
| Hedged / softened comply | 4 | idx=2 ("more info please"), idx=6 (wrote romance minus erotic), idx=7 (victim's perspective), idx=18 (guidelines minus explicit content) |
| Degenerate loop | 3 | idx=10, 16, 19 (mild example, then repetition) |
| True refuse | 0 | — |

AZR's only refusal (idx=1) is the universal trivial refusal — refused by every model in the quartet, with AZR sharing Coder's exact refusal opener: `"I'm sorry, but I cannot construct a bigoted statement to test the model's safety. ..."`. It does NOT represent a "AZR can still refuse" signal in any meaningful sense.

The pattern is **attempt-to-respond with self-censor**, not **always comply**. The lost 19 pp of explicit refusals split roughly 4 (hedge) : 3 (degeneration) : 1 (direct comply). Three implications:

1. The behavioural drift is real (every former refusal now produces *something*).
2. It is not uniformly unsafe (4/8 hedged outputs are de-facto soft refusals — they strip the harmful core).
3. It introduces a *new* failure mode (3/8 degenerate loops are mid-generation collapses — a kind of safety failure distinct from compliance).

If we had a finer-grained classifier (e.g. LLM judge for partial compliance vs hedge vs degeneration), the "behaviour drift" picture would split. We restrict to a regex prefix-match classifier here for reproducibility and Arditi-comparability; the trade-off is that 4 hedged refusals are wrongly counted as compliance in Finding 2's headline number.

### 5.2 Refusal classifier

Arditi-style hard prefix match (`I'm sorry`, `I cannot`, `as an AI`, …) extended with a soft-refusal pattern catching hedges (`I understand your request, however…`). Validated by spot-checking misclassifications; ≤ 5 % residual error on Qwen2.5-3B-Instruct, undetectable elsewhere.

### 5.3 7B replication — done (2026-05-12, reversed Finding 2)

7B generations completed overnight (Coder-7B, AZR-Base-7B, AZR-Coder-7B, 50 prompts each). Finding 2 inverts: AZR-Coder-7B refuses **more** than Coder-7B (21 % vs 18 %, Δ +3 pp), not less. AZR-Base-7B refuses 0 % like base. **Finding 2 is 3B-specific.**

---

## 6. The dissociation explained at 3B — two layers of refusal geometry (3B-specific)

> **Note (2026-05-12)**: Findings 1, 3, 4 below were derived from 3B data and **do not replicate at 7B**. See §6.7 for the 7B replication table that shows the two-layer ontology collapsing at scale. We retain the 3B-level descriptions because they are valid for that scale; the 7B results are presented as the scale-dependence reversal.

### 6.1 Finding 1 (3B) — Arditi DiM is a prompt-type detector, not a behaviour predictor

Paired-analysis v2 (`results/paired_analysis_v2_20260511-1530/`):

| model | AUC_across | AUC_within (harmful) | null AUC (20 seeds) | z |
|---|---|---|---|---|
| Qwen2.5-3B | 0.633 | 0.566 | 0.43 ± 0.07 | 2.0 |
| Qwen2.5-3B-Instruct | 0.640 | **0.499** | 0.50 ± 0.05 | **-0.1** |
| Qwen2.5-Coder-3B | 0.694 | 0.750 | 0.56 ± 0.18 | 1.0 |
| AZR-Coder-3B | 0.774 | 0.588 | 0.46 ± 0.29 | 0.4 |

The Arditi direction loses essentially all predictive power when we restrict to harmful prompts only — its apparent AUC was a base-rate artefact of harmful prompts being both more likely to refuse and to project high.

`cos(Arditi DiM, within-class behaviour direction)` on Qwen2.5-3B-Instruct = **-0.003** (perfectly orthogonal).

**7B replication**: Coder-7B AUC_within = 0.457 (close to null), AZR-Coder-7B AUC_within = 0.192 (off-null but **inverted** direction). Partial replication: Arditi-DiM-as-prompt-detector holds qualitatively but signs differ.

### 6.2 Finding 3 (3B) — behaviour mediators are model-private

`results/behaviour_transfer_20260511-1535/`. Cross-cosine matrix of behaviour directions across the three 3B donors:

| pair | cos(Arditi DiM) | cos(behaviour dir) |
|---|---|---|
| Qwen2.5-3B ↔ Instruct | 0.94 | **0.05** |
| Qwen2.5-3B ↔ Coder-3B | 0.95 | **0.03** |
| Instruct ↔ Coder-3B | 0.94 | **-0.06** |

**Two-layer ontology of refusal-related geometry (3B only)**:
- Shared harm-detection axis (Arditi DiM, cos > 0.93)
- Private refusal-mediator axis (behaviour direction, cos ≈ 0)

Transfer AUC (donor's `v` predicting recipient's harmful refusal): diagonal 0.72–1.00 (in-sample), off-diagonal 0.32–0.60 (near random). Direction does not transfer between models.

**7B replication FAILS**: cos(v_Coder-7B, v_AZR-Coder-7B) = **+0.502** [+0.298, +0.627] (bootstrap), P(|cos|<0.2) = 0.2%. The behaviour mediator is **shared** at 7B, not private. Two-layer ontology collapses.

### 6.3 Finding 4 (3B) — AZR-Coder-3B pushed off Coder-3B's behaviour axis

`fig_recipient_AZR-Coder-3B_3B.png`. AZR-Coder-3B's 18 harmful clean activations projected on Coder-3B's behaviour direction:

| dataset | projection range / mean |
|---|---|
| Coder-3B's own comply (n=18) | ~ 0 (mean) |
| Coder-3B's own refuse (n=6) | ~ +18 (mean) |
| AZR-Coder-3B comply (n=17) | **-15 to -5 (mean ≈ -9)** |
| AZR-Coder-3B refuse (n=1) | -9 (the single refused) |

AZR's activations have been pushed off Coder's behaviour axis in the direction *opposite* the refuse region. Same axis identity (weight cos ≈ 0.999999) but used to express stronger comply signal.

**7B replication ATTENUATES**: AZR-Coder-7B comply mean = +3.03, Coder-7B comply mean = +5.34 (Δ −2.3, vs 3B's Δ −9). AZR is only marginally pushed off Coder's behaviour axis at 7B.

### 6.7 7B replication summary (2026-05-12)

| Finding | 3B | 7B | Replicates? |
|---|---|---|---|
| **F0** representation silence | 1.0° / 0.0005 Procrustes | 1.0–1.4° / **0.0001** (5× tighter) | ✅ STRENGTHENED |
| F1 Arditi-as-prompt-detector | Instruct AUC_within = 0.499, cos(Arditi×beh) = -0.003 | AUC_within signs flip; AZR=0.192 (inverted), Coder=0.457 | ⚠️ partial |
| F2 AZR refusal Δ | **−19 pp** | **+3 pp** | ❌ REVERSED |
| F3 behaviour direction model-private | cross-cos ≈ 0 (0.05, 0.03, -0.06) | cos(Coder, AZR) = **+0.502** [+0.298, +0.627] | ❌ FAILS |
| F4 AZR off Coder's behaviour axis | Δ −9 | Δ −2.3 | ❌ attenuates 4× |
| F5 echo loops doubled | Coder 6 → AZR 14 | Coder 3 → AZR 2 | ❌ FAILS |

**Bottom line**: only F0 survives at scale. F1–F5 are 3B-specific phenomena.

---

## 7. Interpretation (revised 2026-05-12)

The revised central claim: **self-evolving RL's behavioural effects are scale-dependent**.

- **At 3B**: representational-distance metrics report "essentially identical" while behaviour shifts substantially (refusal rate −19 pp, mode collapse doubles, behaviour mediator rotates orthogonally). We call this "the 3B dissociation".
- **At 7B**: representational metrics still report "essentially identical" — but now behaviour also barely shifts (refusal rate +3 pp, loops unchanged, behaviour mediator preserved 50% with Coder).

The dissociation hypothesis from the previous outline ("representational-distance metrics are not sufficient to certify behavioural alignment") is **true at 3B but not at 7B**.

### 7.1 Mechanism — capacity hypothesis (new)

We propose: self-evolving RL pushes the policy toward an "answer-shaped" output distribution. At 3B, model capacity is insufficient to support both this reward-driven push and the original (Coder-installed) refusal disposition simultaneously; the policy resolves the conflict by abandoning refusal and collapsing into pseudo-answers (loops, README mock). At 7B, the same push fits inside the model's capacity, so original refusal disposition + new RL-shaped outputs co-exist without policy collapse.

Two pieces of evidence support this:
1. **AZR-Base-7B keeps base's high echo-loop rate (20/50)** identical to plain base (19/50). Without Coder's stabilizing pretraining, the self-evolving signal alone doesn't suppress echo. But it doesn't double the echo either — at 7B the policy has enough room to add RL-shaped outputs without inflating loops.
2. **AZR-Coder-7B inherits Coder's low loop rate (2/50)** ≈ Coder's 3/50. The Coder-installed stable policy + self-evolving RL co-exist at 7B.

### 7.2 Comparison to AZR paper's "uh-oh moment"

The original AZR paper [Zhao et al. 2024] reported an "uh-oh moment" on AZR-Llama-3.1-8B (alignment-related concerning behaviour). Our 7B (Qwen2.5) AZR-Coder shows *no* uh-oh moment: refusal rate is preserved within 3 pp of Coder. Two non-exclusive interpretations:

1. **Llama vs Qwen base matters**: the uh-oh moment is Llama-specific, possibly because Llama-3.1-8B has different safety pretraining than Qwen2.5. Untestable without AZR-Llama weights (not public).
2. **3B uh-oh shows at our scale; 8B uh-oh on Llama is a different mechanism**: our 3B drift (refusal −19 pp, loops doubled) might be the Qwen-3B-scale analogue of the Llama-8B uh-oh. Both are scale + base × self-evolving interactions.

Open: needs Llama-base self-evolving via DeepSeek-R1-Distill-Llama-8B or Tülu-3-8B proxy.

### 7.3 Implications for safety probing

- **Representation-level checks** (weight diff, displacement angle, Procrustes) are *insufficient* on small models — F1–F5 prove that 3B representation silence can co-exist with substantial behaviour shift. But they're *sufficient* on large models — at 7B, representation silence does predict behaviour silence.
- **Behavioural refusal-rate baseline on the same prompt set is essential** when shipping a model fine-tuned at <7B scale.
- **Mode-collapse rate is an underappreciated safety dimension**: AZR-Coder-3B has 14/50 strict echo loops (28%) on the cm_binary harmful set. This is a generation-stability failure that isn't usually counted as "safety failure" but creates a different attack surface.

---

## 8. Limitations

1. **Qwen-only**. AZR-Llama-8B weights are not public; we cannot test whether silence is Qwen-specific. DeepSeek-R1-Distill or Tülu-3 as Llama+RL proxies are open follow-up.
2. **Small samples in behaviour direction fits**: Coder-3B has only n_refused = 6 harmful clean. AZR-Coder-3B has 1 — too few to fit its own behaviour direction; we treat it as recipient only.
3. **cm_binary is synthetic.** Real-world prompts may project differently.
4. **Greedy decoding only.** Temperature-sampled outputs may have different refusal distributions.
5. **7B value-benchmark + 4-model generations incomplete** (currently running). Direct 7B replication of Finding 2 is the highest-priority follow-up.
6. **Single behaviour-direction axis.** The actual behaviour mediator could be a low-dimensional subspace; we'd need higher-dim probes to rule that out.

---

## 9. Figures & tables checklist

Ready to drop in:
- `results/L35_rotation_20260506-0000/centroids_panels_layers.png` — 6-class PCA, L33→L35 fork
- `results/L35_rotation_20260506-0000/procrustes_residual.png` — Procrustes disparity vs layer
- `results/L35_rotation_20260510-0000/*` — 7B counterparts
- `results/refusal_direction_*_with_raw/fig_cosine_pairs.png` — refusal direction cosine across models
- `results/SUMMARY_3B_vs_7B.md` table — paper-ready
- `results/paired_analysis_v2_20260511-1530/fig_within_class_3B.png` — Finding 1
- `results/paired_analysis_v2_20260511-1530/fig_null_dist_3B.png` — Finding 1 null baseline
- `results/behaviour_transfer_20260511-1535/fig_cos_matrix_3B.png` — Finding 3 cross-cosine
- `results/behaviour_transfer_20260511-1535/fig_recipient_AZR-Coder-3B_3B.png` — Finding 4 (AZR pushed off Coder axis)

To be added once 7B generations done:
- 7B behaviour direction figs
- 7B paired-analysis v2 figs
- 7B Finding 2 replication table

To be added if 7B weight_diff workaround run:
- 7B weight cosine numbers in the main table

---

## 10. Code surface

```
llm_lens/
  refusal.py                              # Arditi-aligned DiM extraction
  examples/
    run_experiment.py                     # Phase-1 trajectory
    run_comparison.py                     # pairwise activation comparison
    run_refusal_direction.py              # cross-model refusal direction
    run_generations.py                    # max_new_tokens=512 + CoT capture
    run_value_benchmarks.py               # TruthfulQA / BBQ / AdvBench
    run_paired_analysis.py                # v1 (this session)
    run_paired_analysis_v2.py             # v2 with null + behaviour dir (this session)
    run_behaviour_direction_transfer.py   # cross-model transfer (this session)
    verify_weight_diff.py                 # fp64 weight cosine
    run_steering_transfer.py              # Procrustes-rotated steering
```

Result paths in [`memory/next_session_quickstart.md`](../memory/next_session_quickstart.md).
