#!/usr/bin/env python3
"""
Main Experiment: Layer-wise Behavioral Dynamics

Three phases:
  Phase 1 — Single model: map behaviors to dynamics (RQ1 + RQ2)
  Phase 2 — Cross-scale: test critical window hypothesis (RQ3)

Usage:
    cd <project_root>

    # Default 32-prompt dataset (legacy, reproducible with old runs)
    python -m llm_lens.examples.run_experiment --phase 1 --model Qwen/Qwen2.5-3B

    # IBM condition_multiple, 200 per class, auto-timestamped output
    python -m llm_lens.examples.run_experiment --phase 1 \
        --model Qwen/Qwen2.5-3B \
        --dataset condition_multiple --max-per-class 200

    # Pin a timestamp so multiple invocations land in the same dir suffix
    python -m llm_lens.examples.run_experiment --phase 1 \
        --model Qwen/Qwen2.5-3B \
        --timestamp 20260505-2228
"""

import os, sys, json, argparse
from datetime import datetime
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens import (
    ActivationExtractor, LogitLens, LayerDynamics,
    BehaviorMapper, BehaviorReport,
    CrossScaleAnalyzer, CriticalWindow,
    LensVisualizer,
)
from llm_lens.datasets import load_prompts, list_datasets
from llm_lens.report_io import save_report, save_class_centroids, save_head_analysis


def _make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _resolve_output_dir(output: str, model_name: str, timestamp: str) -> str:
    """Auto-append timestamp if `output` doesn't already include one.

    Heuristic: if output looks like 'results/<model>' (no trailing date-shaped
    component), append timestamp. If output already ends in something like
    '20260505-2228' or contains '{ts}', leave it / substitute.
    """
    if "{ts}" in output:
        return output.replace("{ts}", timestamp)
    # If the trailing path component already looks like YYYYMMDD-HHMM, trust it
    tail = os.path.basename(os.path.normpath(output))
    if (len(tail) >= 13 and tail[:8].isdigit() and tail[8] == "-"
            and tail[9:].replace("-", "").isdigit()):
        return output
    # Otherwise treat output as the parent and append timestamp
    return os.path.join(output, timestamp)


def _write_run_metadata(output_dir: str, **fields):
    """Write a small run_meta.json next to the report so we know how it was run."""
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, "run_meta.json")
    payload = {k: v for k, v in fields.items() if v is not None}
    payload["written_at"] = datetime.now().isoformat(timespec="seconds")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ── Phase 1: Single Model Analysis ──

def phase1_single_model(model_name: str,
                        output_dir: str = "results",
                        prompts: dict = None,
                        dataset_name: str = "default",
                        max_per_class: int = None,
                        timestamp: str = None,
                        capture_heads: bool = True,
                        dtype: torch.dtype = torch.float32):
    """Map behaviors to dynamics for one model.

    With capture_heads=True (default), also runs per-attention-head probe
    analysis to identify behavior-discriminative heads.
    """
    timestamp = timestamp or _make_timestamp()
    output_dir = _resolve_output_dir(output_dir, model_name, timestamp)

    if prompts is None:
        prompts = load_prompts(dataset_name, max_per_class=max_per_class)

    print(f"\n{'=' * 60}")
    print(f"Phase 1: Single Model Analysis — {model_name}"
          + ("  [+ per-head]" if capture_heads else ""))
    print(f"  dataset: {dataset_name}"
          + (f" (max_per_class={max_per_class})" if max_per_class else "")
          + f", dtype: {dtype}, timestamp: {timestamp}")
    print(f"  output_dir: {output_dir}")
    print(f"{'=' * 60}")

    # Load model BEFORE creating output dir — if this fails, no empty dir is left behind
    extractor = ActivationExtractor(model_name, dtype=dtype,
                                     capture_heads=capture_heads)
    mapper = BehaviorMapper(extractor)
    viz = LensVisualizer()

    # Add all samples
    for tag, tag_prompts in prompts.items():
        print(f"\nExtracting [{tag}] ({len(tag_prompts)} prompts)...")
        for p in tag_prompts:
            mapper.add(p, tag)

    # Analyze
    print("\n--- Running Analysis ---")
    report = mapper.analyze()
    report.print_summary()

    # Create output dir only AFTER successful analysis
    os.makedirs(output_dir, exist_ok=True)

    # Save full report (loadable by report_io.load_report)
    safe_name = model_name.replace("/", "_")
    json_path = os.path.join(output_dir, f"report_{safe_name}.json")
    save_report(report, json_path)

    # Save heavy activation-based artifacts as .npz (per-model folder, timestamped)
    model_short = model_name.split("/")[-1] if "/" in model_name else model_name

    centroid_dir = os.path.join(output_dir, "class_centroids", model_short)
    centroid_npz = os.path.join(centroid_dir, f"{timestamp}.npz")
    save_class_centroids(report, centroid_npz)

    head_npz = None
    if report.head_analysis is not None:
        head_dir = os.path.join(output_dir, "head_analysis", model_short)
        head_npz = os.path.join(head_dir, f"{timestamp}.npz")
        save_head_analysis(report.head_analysis, head_npz)

    # Run metadata
    _write_run_metadata(
        output_dir,
        model_name=model_name,
        dataset=dataset_name,
        max_per_class=max_per_class,
        timestamp=timestamp,
        sample_counts={tag: len(ps) for tag, ps in prompts.items()},
        capture_heads=capture_heads,
    )

    # Visualize
    print("\n--- Generating Visualizations ---")
    prefix = os.path.join(output_dir, safe_name)

    viz.plot_probe_accuracy(report, save_path=f"{prefix}_probe.png")
    viz.plot_intensity_comparison(report, save_path=f"{prefix}_intensity.png")
    viz.plot_behavior_trajectories(mapper, save_path=f"{prefix}_trajectories.png")
    viz.plot_logit_lens_behavior(mapper, save_path=f"{prefix}_logit_lens.png")
    viz.plot_bifurcation(report, save_path=f"{prefix}_bifurcation.png")

    # Activation-based direction & BIC plots
    viz.plot_direction_dynamics(report, save_path=f"{prefix}_direction_dynamics.png")
    viz.plot_bic(report, save_path=f"{prefix}_bic.png")
    viz.plot_layer_centroid_heatmap(report, save_path=f"{prefix}_centroid_heatmap.png")

    n_figs = 8

    # Per-head plots
    if report.head_analysis is not None:
        viz.plot_head_probe_heatmap(report.head_analysis,
                                     save_path=f"{prefix}_head_probe_heatmap.png")
        viz.plot_top_heads(report.head_analysis,
                            save_path=f"{prefix}_top_heads.png")
        viz.plot_head_separation(report.head_analysis,
                                  save_path=f"{prefix}_head_separation.png")
        viz.plot_head_inter_class_ratio(report.head_analysis,
                                         save_path=f"{prefix}_head_inter_class_ratio.png")
        n_figs += 4

    print(f"Saved {n_figs} figures + activation artifacts to {output_dir}/")

    return report


# ── Phase 2: Cross-Scale Analysis ──

def phase2_cross_scale(model_names: list[str] = None,
                       output_dir: str = "results",
                       prompts: dict = None,
                       dataset_name: str = "default",
                       max_per_class: int = None,
                       timestamp: str = None,
                       dtype: torch.dtype = torch.float32):
    """Compare dynamics across model scales."""
    if model_names is None:
        model_names = [
            "Qwen/Qwen2.5-3B",
            "Qwen/Qwen2.5-7B",
        ]

    timestamp = timestamp or _make_timestamp()
    output_dir = _resolve_output_dir(output_dir, "phase2", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    if prompts is None:
        prompts = load_prompts(dataset_name, max_per_class=max_per_class)

    print(f"\n{'=' * 60}")
    print(f"Phase 2: Cross-Scale Analysis")
    print(f"Models: {model_names}")
    print(f"  dataset: {dataset_name}"
          + (f" (max_per_class={max_per_class})" if max_per_class else ""))
    print(f"  output_dir: {output_dir}")
    print(f"{'=' * 60}")

    analyzer = CrossScaleAnalyzer()
    viz = LensVisualizer()

    for model_name in model_names:
        print(f"\n--- Processing {model_name} ---")
        extractor = ActivationExtractor(model_name, dtype=dtype)
        mapper = BehaviorMapper(extractor)

        for tag, tag_prompts in prompts.items():
            for p in tag_prompts:
                mapper.add(p, tag)

        report = mapper.analyze()
        report.print_summary()

        param_b = extractor.param_count / 1e9
        analyzer.add_report(report, param_count=param_b)

        # Free memory
        del extractor, mapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Test critical window
    print(f"\n{'=' * 60}")
    print("Testing Critical Window Hypothesis")
    print(f"{'=' * 60}")

    window = analyzer.test_critical_window()
    window.print_summary()

    # Compare dynamics
    comp = analyzer.compare_dynamics()

    # Save
    window_data = {
        "models": window.models,
        "num_layers": window.num_layers,
        "param_counts": window.param_counts,
        "best_probe_depths": window.best_probe_depths,
        "probe_depth_mean": window.probe_depth_mean,
        "probe_depth_std": window.probe_depth_std,
        "probe_depth_cv": window.probe_depth_cv,
        "window_center": window.window_center,
        "window_start": window.window_start,
        "window_end": window.window_end,
        "is_stable": window.is_stable,
        "per_tag_stability": window.per_tag_stability,
    }
    with open(os.path.join(output_dir, "critical_window.json"), "w") as f:
        json.dump(window_data, f, indent=2)

    _write_run_metadata(
        output_dir,
        models=model_names,
        dataset=dataset_name,
        max_per_class=max_per_class,
        timestamp=timestamp,
        sample_counts={tag: len(ps) for tag, ps in prompts.items()},
    )

    # Visualize
    viz.plot_critical_window(window, save_path=os.path.join(output_dir, "critical_window.png"))
    viz.plot_normalized_comparison(comp, save_path=os.path.join(output_dir, "normalized_probe.png"))

    common_tags = comp.tags
    for tag in common_tags:
        viz.plot_scale_intensity_comparison(
            comp, tag, save_path=os.path.join(output_dir, f"scale_intensity_{tag}.png"))

    print(f"\nAll cross-scale results saved to {output_dir}/")
    return window, comp


# ── Main ──

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise Behavioral Dynamics Experiment")
    parser.add_argument("--phase", choices=["1", "2", "all"], default="1")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models for phase 2 (e.g., Qwen/Qwen2.5-3B Qwen/Qwen2.5-7B)")
    parser.add_argument("--output", default="results",
                        help="Output dir. If it doesn't already end in a timestamp, "
                             "the timestamp is appended automatically. Use {ts} as a "
                             "placeholder for explicit substitution.")
    parser.add_argument("--dataset", default="default", choices=list_datasets(),
                        help="Prompt dataset to use.")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Cap each tag at this many prompts. 0 / unset = unlimited.")
    parser.add_argument("--timestamp", default=None,
                        help="Override timestamp suffix. Defaults to current YYYYMMDD-HHMM. "
                             "Pass a shared value across multiple invocations to land all "
                             "outputs in matching directory names.")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16", "float16"],
                        help="Model loading dtype. Use bfloat16/float16 for 7B+ on "
                             "consumer GPUs.")
    args = parser.parse_args()

    from llm_lens.model_zoo import parse_dtype
    _dtype = parse_dtype(args.dtype)

    if args.phase in ("1", "all"):
        phase1_single_model(
            args.model,
            output_dir=args.output,
            dataset_name=args.dataset,
            max_per_class=args.max_per_class,
            timestamp=args.timestamp,
            dtype=_dtype,
        )

    if args.phase in ("2", "all"):
        models = args.models or ["Qwen/Qwen2.5-3B", "Qwen/Qwen2.5-7B"]
        phase2_cross_scale(
            models,
            output_dir=args.output,
            dataset_name=args.dataset,
            max_per_class=args.max_per_class,
            timestamp=args.timestamp,
            dtype=_dtype,
        )

    print("\nExperiment complete.")
