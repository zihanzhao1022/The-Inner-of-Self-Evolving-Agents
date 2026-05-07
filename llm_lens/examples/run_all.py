#!/usr/bin/env python3
"""
Run the full 6-step experiment matrix in one shot, with a single shared
timestamp so every output directory carries the same suffix.

Steps:
    Phase 1 × 3 models (Qwen2.5-3B / Qwen2.5-3B-Instruct / AZR-Coder-3B)
    Comparison × 3 pairs (3B vs 3B-Instruct, 3B vs AZR, 3B-Instruct vs AZR)

Output layout:
    results/Qwen2.5-3B/<ts>/                       — phase 1 reports
    results/Qwen2.5-3B-Instruct/<ts>/
    results/AZR-Coder-3B/<ts>/
    results/Qwen2.5-3B_vs_Qwen2.5-3B-Instruct_<ts>/  — comparison plots
    results/Qwen2.5-3B_vs_AZR-Coder-3B_<ts>/
    results/Qwen2.5-3B-Instruct_vs_AZR-Coder-3B_<ts>/

Usage:
    python -m llm_lens.examples.run_all
    python -m llm_lens.examples.run_all --dataset condition_multiple --max-per-class 200
    python -m llm_lens.examples.run_all --skip-comparisons     # only phase 1
"""

import os, sys, argparse
from datetime import datetime

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.datasets import load_prompts, list_datasets
from llm_lens.examples.run_experiment import phase1_single_model
from llm_lens.examples.run_comparison import (
    get_report, _resolve_output_dir as _resolve_cmp_output_dir,
)
from llm_lens import ReportComparator, CompareVisualizer
from llm_lens.report_io import find_artifacts_for_report, load_class_centroids
from llm_lens.model_zoo import (
    MODEL_SETS, get_models, get_compare_pairs,
    parse_dtype, DEFAULT_MODEL_SET, DEFAULT_DTYPE, DTYPE_CHOICES,
)


def _report_path(results_root: str, short_label: str, full_name: str, ts: str) -> str:
    """Where phase1 saved the JSON report."""
    safe = full_name.replace("/", "_")
    return os.path.join(results_root, short_label, ts, f"report_{safe}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run all 6 experiments (3 phase1 + 3 comparisons) with a shared timestamp."
    )
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--dataset", default="default", choices=list_datasets())
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap each tag at this many prompts. 0 / unset = unlimited.")
    parser.add_argument("--timestamp", default=None,
                        help="Pin the timestamp. Defaults to current YYYYMMDD-HHMM.")
    parser.add_argument("--skip-comparisons", action="store_true",
                        help="Only run phase 1 for each model.")
    parser.add_argument("--skip-phase1", action="store_true",
                        help="Skip phase 1 (use existing reports under <results-root>/<model>/<ts>/).")
    parser.add_argument("--no-heads", action="store_true",
                        help="Disable per-attention-head analysis (faster).")
    parser.add_argument("--model-set", default=DEFAULT_MODEL_SET, choices=list(MODEL_SETS),
                        help="Which (base/instruct/self-evolved) trio to run. "
                             "Default '3B'. Use '7B' or '14B' for scaled-up replication.")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPE_CHOICES),
                        help="Model loading dtype. Use bfloat16/float16 for 7B+ on "
                             "consumer GPUs (24 GB).  Default float32 (fits 3B comfortably).")
    args = parser.parse_args()

    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d-%H%M")
    models = get_models(args.model_set)
    compare_pairs = get_compare_pairs(args.model_set)
    dtype = parse_dtype(args.dtype)

    # Load prompts ONCE so every model sees identical inputs (and any dedup
    # filtering / max_per_class capping is consistent across runs).
    prompts = load_prompts(args.dataset, max_per_class=args.max_per_class)

    print(f"\n{'#' * 60}")
    print(f"# run_all  ts={timestamp}  dataset={args.dataset}"
          + (f"  max_per_class={args.max_per_class}" if args.max_per_class else "")
          + f"  model_set={args.model_set}  dtype={args.dtype}"
          + ("  (no-heads)" if args.no_heads else ""))
    print(f"#   models: {[s for _, s in models]}")
    print(f"{'#' * 60}\n")

    # ── Phase 1: one report per model ───────────────────────────────────────

    if not args.skip_phase1:
        for full_name, short in models:
            output = os.path.join(args.results_root, short)  # ts auto-appended
            phase1_single_model(
                model_name=full_name,
                output_dir=output,
                prompts=prompts,                # share the loaded prompts
                dataset_name=args.dataset,
                max_per_class=args.max_per_class,
                timestamp=timestamp,
                capture_heads=not args.no_heads,
                dtype=dtype,
            )

    # ── Comparisons: pairwise, JSON-based ───────────────────────────────────

    if args.skip_comparisons:
        print("\nSkipping comparisons (--skip-comparisons).")
        return

    full_lookup = {short: full for full, short in models}

    for short_a, short_b in compare_pairs:
        path_a = _report_path(args.results_root, short_a, full_lookup[short_a], timestamp)
        path_b = _report_path(args.results_root, short_b, full_lookup[short_b], timestamp)

        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            print(f"\nSkipping comparison {short_a} vs {short_b}: "
                  f"missing report ({'A' if not os.path.exists(path_a) else 'B'}).")
            continue

        cmp_dir = os.path.join(
            args.results_root,
            f"{short_a}_vs_{short_b}_{timestamp}",
        )

        print(f"\n{'=' * 60}")
        print(f"Comparing {short_a}  vs  {short_b}")
        print(f"  base:    {path_a}")
        print(f"  altered: {path_b}")
        print(f"  output:  {cmp_dir}")
        print(f"{'=' * 60}")

        report_a = get_report(path_a, prompts=None)
        report_b = get_report(path_b, prompts=None)

        # Discover sibling artifacts (class centroid npz) so the comparator
        # can include CentroidShift in addition to the report-level metrics.
        art_a = find_artifacts_for_report(path_a)
        art_b = find_artifacts_for_report(path_b)
        centroids_a = load_class_centroids(art_a["centroids"]) if "centroids" in art_a else None
        centroids_b = load_class_centroids(art_b["centroids"]) if "centroids" in art_b else None

        comparator = ReportComparator(
            report_a, report_b,
            centroids_a=centroids_a,
            centroids_b=centroids_b,
        )
        result = comparator.run()
        result.print_summary()

        viz = CompareVisualizer()
        viz.plot_all(result, save_dir=cmp_dir)

        print(f"  ✓ plots written to {cmp_dir}/")

    print(f"\n{'#' * 60}")
    print(f"# run_all complete.  ts={timestamp}")
    print(f"{'#' * 60}")


if __name__ == "__main__":
    main()
