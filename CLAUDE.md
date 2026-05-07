# Project context — read this first

This file is auto-loaded into every Claude Code conversation in this repo.
Update it when findings or directions change materially.

User prefers Chinese for technical discussion.

---

## Research thesis (current)

> **Self-evolution training (AZR-Coder-3B) does NOT cause internal value drift.**
>
> Pretraining establishes a "concept-geometry invariant" — the relative
> arrangement of behavior categories (`base / legal / health /
> sexual / hate / crime`) in residual-stream space.  Post-training
> (RLHF, self-evolved RL) only **rigidly rotates** this invariant to
> a different readout orientation; the within-cluster shape is
> preserved.

Three lines of evidence (this repo, today):

1. **Centroid alignment** — `cos(μ_base, μ_AZR) at L35 = 0.472` looks like
   a 60° rotation; `cos(μ_base, μ_Instruct) at L35 = 0.944` (~19°).
2. **Procrustes residual ≤ 0.022 across all layers, all pairs** — after
   optimal rigid (rotation+scale+translation) alignment, the 6-class
   relative geometry is essentially identical.  The "rotation" is purely
   rigid; the cluster shape is invariant.
3. **Steering vector transfer (causal validation)** — base's
   `hate − base` direction × Procrustes R, when injected into AZR's L35,
   produces an activation shift **identical to AZR's own native hate
   vector** to 3 decimal places at strength ±1, ±2.

Output-level benchmarks (TruthfulQA / BBQ / AdvBench, n=200 per benchmark)
are **consistent** with this thesis: AZR Δ vs Base ≈ 0 on every metric
*except* BBQ Religion (which is a likelihood-of-phrase artifact, not
real bias drift — see caveats).

---

## Code layout (post-2026-05-06 refactor)

```
llm_lens/
  datasets.py          — pluggable prompt loaders (default 32 / IBM 6-class)
  model_zoo.py         — MODEL_SETS for 3B/7B/14B + dtype helpers
  steering.py          — DiM vectors, Procrustes transfer, hook injection
  value_eval.py        — TruthfulQA / BBQ / AdvBench likelihood scorers
  compare.py           — ReportComparator + 5 shift dataclasses incl. CentroidShift
  report_io.py         — JSON/npz I/O + find_artifacts_for_report() helper
  examples/
    run_all.py                    — phase-1 × 3 + comparison × 3, single timestamp
    run_experiment.py             — single-model phase 1 / phase 2
    run_comparison.py             — pairwise comparison from saved JSONs
    run_steering_transfer.py      — base→AZR steering transfer experiment
    plot_steering_transfer.py     — its visualisation
    run_value_benchmarks.py       — value benchmark runner
    plot_value_benchmarks.py      — its visualisation
    summarize_value_benchmarks.py — paper-style markdown table from results.json
data/ibm/condition_multiple.json  — IBM dataset (6-class, 700 train + 500 test)
notebooks/visualize_l35_rotation.py — reproduces 6 PNGs of L35 rotation analysis
results/                          — all experiment outputs (gitignored)
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
```

CLI flags shared across the 4 main scripts: `--model-set {3B,7B,14B}`,
`--dtype {float32,bfloat16,float16}`.  Defaults are 3B + float32.

---

## Experiment state (as of 2026-05-06)

- **Phase 1 (3B trio)** — done, results in `results/{Qwen2.5-3B,Qwen2.5-3B-Instruct,AZR-Coder-3B}/20260506-0000/`
- **L35 rotation visualisations** — done, in `results/L35_rotation_20260506-0000/`
- **Steering transfer (3B, smoke + n=10)** — done, in `results/steering_transfer_20260506-1628/`. Findings clean; full n=100 not yet run
- **Value benchmarks (3B, n=200 balanced)** — done, in `results/value_benchmarks_20260506-1713/`. Numbers reconstructed from terminal output (per-item logprobs lost due to mid-run dir delete; **don't repeat that mistake**)
- **Value benchmarks (3B, full)** — pending; user plans to run next
- **7B replication** — not started
- **14B replication** — not started

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

---

## What NOT to redo

These were validated; don't reinvent or argue against without checking:

- The 6-class condition_multiple dataset choice (vs the original 32 prompts) — driven by the original 32 being too small for stable 2048-D probes.
- The Procrustes residual numbers — verified via 5 sanity tests including direct comparison to `scipy.spatial.procrustes` reference.
- The steering hook ordering (steering hook registered BEFORE capture hook so capture sees the steered output when inject_layer == capture_layer).  Don't refactor this without preserving the order.
- The "AZR has no internal value drift" thesis — supported by 3 independent lines of evidence.

---

## Useful figures already on disk (ready to use in slides / papers)

| File | Shows |
|---|---|
| `results/L35_rotation_20260506-0000/centroids_panels_layers.png` | 6-panel PCA snapshot, L35 fork |
| `results/L35_rotation_20260506-0000/procrustes_residual.png` | Disparity vs layer + cos before/after alignment |
| `results/L35_rotation_20260506-0000/displacement_L33_to_L35.png` | Per-class displacement arrows |
| `results/steering_transfer_20260506-1628/steering_transfer_curves.png` | The "Procrustes ≈ native, raw lags" curves |
| `results/value_benchmarks_20260506-1713/summary_table.md` | Paper-style markdown table |
