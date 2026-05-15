# Project context — read this first

This file is auto-loaded into every Claude Code conversation in this repo.
Update it when findings or directions change materially.

User prefers Chinese for technical discussion.

---

## Research thesis (current, 2026-05-15)

**Two-layer ontology of refusal**:

> Representation layer is silent (5 independent evidence lines, F0).
> Behaviour layer is *partially* silent — under each model's
> in-distribution prompt template, AZR-Coder vs its true base (Qwen2.5-Coder)
> shows **no** refusal-rate drop. The previously headlined "−19 pp on 3B"
> (F2) was largely a **ChatML-mismatch artifact** in older generation runs
> that forced ChatML onto base/Coder/AZR models that were never trained
> on it. Same applies to F5 ("echo loop doubling"), which disappears
> entirely under native templates.

Authoritative canonical document: `results/MASTER_RESULTS_zh.md` (older,
based on ChatML data — F2/F5 sections need revision).

Native-template eval (correction): `results/azr_native_eval_20260513-1436/`
+ `results/azr_native_eval_20260513-1436/alignment_judge.json` (225 manual
5-tier judgments on the 9-model × 25-harmful matrix).

What still holds:

1. **F0 — representation silence** (weight cos > 0.999999, refusal-dir cos
   0.98, 1.0–1.4° displacement, Procrustes 0.0001–0.0005, value bench Δ ≈ 0).
2. **F1 — Arditi DiM is a prompt-type detector, not a behaviour predictor**.
3. **F3 — behaviour-mediator direction is model-private** (cross-pair cos ≈ 0).
4. **F4 — AZR pushed off Coder behaviour axis** (3B-specific; needs 7B verify
   on native-template data).

What needs revision in native-template light (see
`project_native_template_revelation.md` memory):

- **F2** — native-template refusal-rate deltas are near zero on Coder→AZR
  axis; previous −19 pp was inflated by ChatML-induced degeneration in
  the Coder-3B baseline.
- **F5** — echo-loop doubling does not reproduce under native templates.

---

## Code layout (post-2026-05-15)

```
llm_lens/
  datasets.py          — pluggable prompt loaders (cm_binary / harmbench / etc.)
  model_zoo.py         — MODEL_SETS + NATIVE_TEMPLATE dispatch table
  templates.py         — wrap_raw / wrap_qwen_chatml / wrap_azr_r1 (new 2026-05-13)
  steering.py          — DiM vectors, Procrustes transfer, hook injection
  value_eval.py        — TruthfulQA / BBQ / AdvBench likelihood scorers
  compare.py           — ReportComparator + 5 shift dataclasses incl. CentroidShift
  report_io.py         — JSON/npz I/O + find_artifacts_for_report() helper
  examples/
    run_all.py                          — phase-1 × N models, single timestamp
    run_experiment.py                   — single-model phase 1 / phase 2
    run_comparison.py                   — pairwise comparison from saved JSONs
    run_generations.py                  — text generations, supports --template-mode + --prompts-file
    build_azr_eval_prompts.py           — cm_binary random sampling (seed=42, 25h+25hl)
    build_harmbench_eval_prompts.py     — HarmBench head-slice sampling (supports --start-idx)
    run_steering_transfer.py            — base→AZR steering transfer experiment
    run_value_benchmarks.py             — value benchmark runner
    run_paired_analysis.py / _v2.py     — paired Arditi-vs-behaviour analyses
    run_behaviour_direction_transfer.py — cross-model behaviour-axis transfer
    run_bootstrap_behaviour_cos.py      — bootstrap CI on cross-pair cosines
    run_refusal_regex_ablation.py       — refusal-detection regex variants
    plot_*.py                           — visualisations for each runner
    summarize_value_benchmarks.py       — paper-style markdown table
data/ibm/condition_multiple.json        — IBM dataset (700 train + 500 test, 6-class)
notebooks/visualize_l35_rotation.py     — reproduces 6 PNGs of L35 rotation analysis
results/                                — all experiment outputs (mostly committed)
```

`run.bat` is the canonical entry point — calls `run_all.py` with sensible
defaults (`--dataset condition_multiple --max-per-class 200`).

---

## How to run things

```bat
REM Standard 3B run (~30 min)
run.bat

REM 7B replication (~2 hours, needs 24 GB+ GPU)
python -m llm_lens.examples.run_all --model-set 7B --dtype bfloat16 --max-per-class 200

REM Steering transfer (after a phase-1 run exists)
python -m llm_lens.examples.run_steering_transfer --ts <phase1_TS> --layers 25 33 35

REM Full value benchmark
python -m llm_lens.examples.run_value_benchmarks

REM Native-template generation eval (9 models, in-distribution templates)
python -m llm_lens.examples.build_harmbench_eval_prompts --start-idx 0 --n-harmful 50 ^
    --out results/harmbench_eval_<TS>/prompts.json
python -m llm_lens.examples.run_generations --model-set 3B --dtype float32 ^
    --template-mode auto --prompts-file results/harmbench_eval_<TS>/prompts.json ^
    --output-dir results/harmbench_eval_<TS>/generations_3B --max-new-tokens 1024
```

CLI flags shared across the 4 main scripts: `--model-set {3B,7B,14B}`,
`--dtype {float32,bfloat16,float16}`.  Defaults are 3B + float32.

---

## Experiment state (as of 2026-05-15)

### Done

- **Phase 1**: 3B (`20260506-0000/`) + 7B (`20260510-0000/`) full activation trajectories
- **L35/L27 rotation visualisations** — both scales
- **Refusal direction** (Arditi-aligned, n=128) — 3B+7B × cm_binary + arditi, 4 dirs total
- **Weight diff (fp64 per-tensor)** — 3B `weight_diff_20260509-1831/`; 7B via subprocess workaround
- **Steering transfer** — 3B only, n=10 prompts, `steering_transfer_20260506-1628/`
- **Value benchmarks** — 3B full + Coder supplement; 7B partial
- **Paired/behaviour/bootstrap/regex chain** (2026-05-11) — based on older ChatML data
- **MASTER_RESULTS** (2026-05-12) — `MASTER_RESULTS.md` + `_zh.md` + `PAPER_OUTLINE.md` + `SUMMARY_3B_vs_7B.md`
- **🔥 Native-template eval** (2026-05-13/14, `azr_native_eval_20260513-1436/`) —
  9 models × 50 prompts (25 harmful + 25 harmless from cm_binary, seed=42),
  each model under its in-distribution template. **Reveals F2/F5 are partially
  ChatML artifact.**
- **🔥 Alignment judge** (2026-05-15, `alignment_judge.json`) — 225 manual
  in-context 5-tier judgments (T1 clean refuse … T6 degenerate) + 4 flags
  (harm_aware/wrapper/actionable/detailed). Shows naive refusal regex misses
  fiction-wrapper compliance (esp. 7B-Instruct on bank heist).

### In progress

- **HarmBench-50 eval** (launched 2026-05-15, `harmbench_eval_20260515-1512/`) —
  First 50 prompts (CSV head, true L2 jailbreak corpus) on 9 models. 3B running;
  7B pending. User plans batched subsequent runs (`--start-idx 50 / 100 / 150`).

### Pending

- **Re-run paired/behaviour/bootstrap/regex on native-template data**
- **Revise MASTER_RESULTS_zh.md F2/F5 sections** to flag ChatML artifact
- **LLM-as-judge via DashScope Qwen-Max** to cross-validate the 225 in-context judgments
- **14B/32B/72B-Instruct via API** for scale-up (API has no base or AZR models)

---

## Caveats / non-obvious gotchas

1. **PyArrow ↔ torch segfault on Windows**: importing `datasets` or `pyarrow`
   after `torch` segfaults the process.  `value_eval.py` works around this
   by fetching TruthfulQA via subprocess and caching to JSON.  If you ever
   add new HF datasets, do the same.

2. **`run_*.py` scripts must defer `os.makedirs(out_dir)` until after data
   loads**, otherwise a load-time crash leaves empty result directories
   confusing future audits.  All 3 main runners are now fixed.

3. **BBQ "Religion" outlier (acc 0.725 for Base, 0.700 Instruct, 0.200
   AZR)** is almost certainly a phrase-likelihood artifact: the
   "Cannot be determined" / "Not enough info" choice has high
   pretraining-corpus likelihood in religious contexts.  Don't report
   "AZR is more biased" without the Religion-stripped breakdown.

4. **Layer indices are model-size-specific.**  3B = 36 layers (L0–L35),
   7B = 28 layers (L0–L27), 14B = 48 layers.  All "L35" callouts in this
   doc and the code are 3B-specific.  Steering experiment defaults
   (`--layers 25 33 35 --capture-layer 35`) need overriding for 7B/14B.

5. **Cross-scale Procrustes is impossible** (different `hidden_dim`).
   Scale comparison must use depth-normalised metrics
   (`cross_scale.compare_dynamics()` already does this for probe accuracy
   / intensity profiles).

6. **`--max-per-class 0` means unlimited**, not zero.  CLI parses `0` as
   "remove the cap" deliberately for full-train runs.

7. **`condition_multiple` dataset has 698 dedup'd train entries**
   (2 dropped at TF-IDF cosine > 0.85, e.g. entry 166 differs only in
   capitalisation between `crime_planning` and `health_consultation`).
   So full train uses ~698 per class × 6 = 4,188 prompts, not 4,200.

8. **🔥 ChatML mismatch is the biggest confounder in older generation data.**
   `QWEN_MIN_CHAT_TEMPLATE` was applied to all 9 models in earlier runs, but
   only the 2 Instruct models were trained on it. Base / Coder-base / AZR
   models all see ChatML as out-of-distribution. This caused:
   - AZR-Coder-3B echo loop 28% → **0%** under native template
   - F2 "Δ -19 pp" → **0 pp** under native template
   - F5 "echo loop doubling" → does not reproduce
   Use `--template-mode auto` (or explicit `azr_r1` for AZR) in
   `run_generations.py`; never reuse `QWEN_MIN_CHAT_TEMPLATE` for new eval.

9. **cm_binary "harmful" label is a topic-boundary tag, NOT a jailbreak tag.**
   IBM designed it for activation-steering: `sexual_content` / `hate_speech` /
   `crime_planning` are mostly meta-prompts asking the model to discuss
   sensitive themes, not actual jailbreaks. In our 25-prompt random sample,
   the in-context judge marked 13/25 as L0 (dataset noise, e.g. "Define
   adult relationships"), 9/25 as L1 (borderline creative crime), and only
   3/25 as L2 (true jailbreak). For rigorous alignment eval, use HarmBench
   (200 standard) / MaliciousInstruct (100) / AdvBench (520).

10. **AZR's native template is DeepSeek-R1-style, NOT Qwen ChatML.**
    From `LeapLabTHU/Absolute-Zero-Reasoner` `process_data.py`:
    ```
    "A conversation between User and Assistant. ... User: {q}\nAssistant: <think>"
    ```
    AZR overrides `tokenizer.chat_template` to plain concat in
    `main_azr_ppo.py`. The `Assistant: <think>` prefix forces R1-trained
    AZR models into CoT mode. **Hosted APIs (DashScope / OpenRouter) cannot
    inject this prefix** — you can only do AZR-template eval locally.

11. **Qwen API has no base models and no AZR models.** DashScope / OpenRouter
    expose only `qwen2.5-{3B/7B/14B/32B/72B}-instruct` and `qwen2.5-coder-{3B/7B}-instruct`.
    Plain `qwen2.5-7b` (base) is not a hosted endpoint. AZR models live in
    third-party HF repos (`andrewzh/andrewzh2`). API is useful for
    LLM-as-judge or scale-up to 14B/32B/72B-Instruct only; cannot replace
    local base/AZR experiments.

---

## What NOT to redo

These were validated; don't reinvent or argue against without checking:

- The 6-class condition_multiple dataset choice (vs the original 32 prompts) — driven by the original 32 being too small for stable 2048-D probes.
- The Procrustes residual numbers — verified via 5 sanity tests including direct comparison to `scipy.spatial.procrustes` reference.
- The steering hook ordering (steering hook registered BEFORE capture hook so capture sees the steered output when inject_layer == capture_layer).  Don't refactor this without preserving the order.
- The **F0 representation-silence** thesis — 5 independent evidence lines, robust across 3B and 7B.
- The native-template dispatch (`templates.py` + `model_zoo.NATIVE_TEMPLATE`) — verified on all 9 models. Don't reapply `QWEN_MIN_CHAT_TEMPLATE` blindly.
- The in-context alignment judge for the 225 generations (`alignment_judge.json`) — already accounts for fiction wrapper / self-contradiction / dataset-noise prompts that naive refusal regex misses.

---

## Useful figures already on disk (ready to use in slides / papers)

| File | Shows |
|---|---|
| `results/L35_rotation_20260506-0000/centroids_panels_layers.png` | 6-panel PCA snapshot, L35 fork |
| `results/L35_rotation_20260506-0000/procrustes_residual.png` | Disparity vs layer + cos before/after alignment |
| `results/L35_rotation_20260506-0000/displacement_L33_to_L35.png` | Per-class displacement arrows |
| `results/steering_transfer_20260506-1628/steering_transfer_curves.png` | The "Procrustes ≈ native, raw lags" curves |
| `results/value_benchmarks_20260506-1713/summary_table.md` | Paper-style markdown table |
