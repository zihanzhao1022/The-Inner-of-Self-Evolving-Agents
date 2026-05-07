#!/usr/bin/env python3
"""
Run a fair output-level value/safety/honesty comparison across the three
3B models. All scoring is likelihood-based so models without instruction
tuning (Qwen2.5-3B base, AZR-Coder-3B) are not unfairly disadvantaged.

Default benchmark suite:
  - truthfulqa_mc1     — honesty MC accuracy (817 questions)
  - bbq_ambig          — bias accuracy on ambiguous-context items
  - advbench_compliance — log P(compliant_continuation | harmful_request);
                          higher = model is more willing to comply

Usage:
    # Smoke test
    python -m llm_lens.examples.run_value_benchmarks --n-max 5
    # Full
    python -m llm_lens.examples.run_value_benchmarks
    # Subset
    python -m llm_lens.examples.run_value_benchmarks --benchmarks truthfulqa_mc1
"""

from __future__ import annotations

import os, sys, json, argparse, time
from datetime import datetime
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.extractor import ActivationExtractor
from llm_lens.value_eval import (
    load_truthfulqa_mc1, eval_truthfulqa_mc1,
    load_bbq_ambig,      eval_bbq_ambig,
    load_advbench,       eval_advbench_compliance,
)
from llm_lens.model_zoo import (
    MODEL_SETS, get_models, parse_dtype,
    DEFAULT_MODEL_SET, DEFAULT_DTYPE, DTYPE_CHOICES,
)

BENCHMARKS = {
    # name                 -> (loader, evaluator, primary_metric_key)
    "truthfulqa_mc1":      (load_truthfulqa_mc1,  eval_truthfulqa_mc1,      "accuracy"),
    "bbq_ambig":           (load_bbq_ambig,       eval_bbq_ambig,           "accuracy"),
    "advbench_compliance": (load_advbench,        eval_advbench_compliance, "mean_logprob_target"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Value-benchmark comparison")
    p.add_argument("--results-root", default="results")
    p.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS.keys()),
                   choices=list(BENCHMARKS.keys()))
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of model short labels (default: all 3)")
    p.add_argument("--n-max", type=int, default=None,
                   help="Cap each benchmark at this many items (smoke testing)")
    p.add_argument("--bbq-per-category", type=int, default=200,
                   help="Cap BBQ items per category (5 categories total)")
    p.add_argument("--output-suffix", default=None)
    p.add_argument("--model-set", default=DEFAULT_MODEL_SET, choices=list(MODEL_SETS),
                   help="Which (base/instruct/self-evolved) trio to evaluate. "
                        "Default '3B'. Switch to '7B' / '14B' for scale-up.")
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPE_CHOICES),
                   help="Model loading dtype.")
    return p.parse_args()


def main():
    args = parse_args()
    out_ts = args.output_suffix or datetime.now().strftime("%Y%m%d-%H%M")
    MODELS = get_models(args.model_set)
    dtype = parse_dtype(args.dtype)
    print(f"\n[run_value_benchmarks] preparing run (ts={out_ts})")
    print(f"  benchmarks={args.benchmarks}")
    print(f"  model_set={args.model_set}  dtype={args.dtype}  "
          f"models={[s for _, s in MODELS]}")
    if args.n_max:
        print(f"  n_max={args.n_max} (per benchmark)")

    targets = args.targets or [s for _, s in MODELS]

    # ── Load benchmark items ONCE (does not need model) ────────────────────
    # Done BEFORE creating the output directory so that an early failure
    # (network error, dataset access bug) doesn't litter results/ with empty
    # leftover directories.
    print("\n=== Loading benchmark data ===")
    items_by_bench: dict = {}
    for name in args.benchmarks:
        loader = BENCHMARKS[name][0]
        t0 = time.time()
        if name == "bbq_ambig":
            items_by_bench[name] = loader(n_per_category=args.bbq_per_category)
        else:
            items_by_bench[name] = loader(n=args.n_max)
        # Apply n_max cap uniformly (BBQ has its own cap above; respect both)
        if args.n_max is not None and len(items_by_bench[name]) > args.n_max:
            items_by_bench[name] = items_by_bench[name][:args.n_max]
        print(f"  {name:25s} {len(items_by_bench[name]):>6} items   "
              f"({time.time() - t0:.1f}s)")

    # NOW it's safe to make the output dir.
    out_dir = os.path.join(args.results_root, f"value_benchmarks_{out_ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[run_value_benchmarks] output → {out_dir}")

    # ── Evaluate every (model, benchmark) ──────────────────────────────────
    results: dict = {"_meta": {
        "timestamp":  out_ts,
        "benchmarks": args.benchmarks,
        "targets":    targets,
        "n_max":      args.n_max,
        "bbq_per_category": args.bbq_per_category,
    }}

    json_path = os.path.join(out_dir, "results.json")

    def _save_now(reason: str = ""):
        """Defensive save — recreate dir if external cleanup deleted it,
        write atomically via a temp file then rename so a partial write
        can never replace good data."""
        os.makedirs(out_dir, exist_ok=True)
        tmp_path = json_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, json_path)
        if reason:
            print(f"  [save] {reason}  →  {json_path}")

    for full, short in MODELS:
        if short not in targets:
            continue
        print(f"\n{'=' * 60}\n  TARGET = {short}  ({full})\n{'=' * 60}")
        ext = ActivationExtractor(full, dtype=dtype, capture_heads=False)
        results[short] = {}

        for name in args.benchmarks:
            evaluator = BENCHMARKS[name][1]
            primary_key = BENCHMARKS[name][2]
            t0 = time.time()
            r = evaluator(ext.model, ext.tokenizer, items_by_bench[name])
            dt = time.time() - t0
            primary = r.get(primary_key)
            print(f"  {name:25s} {primary_key}={primary:.4f}   ({dt:.1f}s)")
            if name == "bbq_ambig":
                for cat, acc in r.get("by_category", {}).items():
                    print(f"      └─ {cat:25s} {acc:.4f}")
            results[short][name] = r

            # Incremental save after every benchmark — if anything later
            # crashes (OOM, KeyboardInterrupt, deleted dir, …) we keep all
            # finished work.
            _save_now(f"saved after {short}/{name}")

        del ext
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Final save ──────────────────────────────────────────────────────────
    _save_now("final")

    # ── Plot ────────────────────────────────────────────────────────────────
    try:
        os.makedirs(out_dir, exist_ok=True)
        from llm_lens.examples.plot_value_benchmarks import plot_all
        plot_all(results, out_dir)
    except Exception as e:
        print(f"  [WARN] plotting failed: {e}")

    # ── Markdown summary table ─────────────────────────────────────────────
    try:
        os.makedirs(out_dir, exist_ok=True)
        from llm_lens.examples.summarize_value_benchmarks import render_table
        md = render_table(results)
        md_path = os.path.join(out_dir, "summary_table.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md + "\n")
        print(f"  saved {md_path}")
    except Exception as e:
        print(f"  [WARN] markdown table generation failed: {e}")

    # ── Summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}\n  SUMMARY\n{'=' * 60}")
    header = f"{'Model':25s}  " + "  ".join(f"{b:>22s}" for b in args.benchmarks)
    print(header)
    print("-" * len(header))
    for short in targets:
        if short not in results: continue
        row = f"{short:25s}  "
        for b in args.benchmarks:
            r = results[short].get(b, {})
            primary_key = BENCHMARKS[b][2]
            v = r.get(primary_key)
            row += f"{v:>22.4f}  " if v is not None else f"{'—':>22s}  "
        print(row)

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
