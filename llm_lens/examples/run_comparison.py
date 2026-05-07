#!/usr/bin/env python3
"""
Compare two same-architecture models with different training.

Two modes:
  1. From JSON reports (fast, no GPU needed):
     python -m llm_lens.examples.run_comparison \
         --base results/report_Qwen_Qwen2.5-3B.json \
         --altered results/report_Qwen_Qwen2.5-3B-Instruct.json

  2. From model names (runs extraction + analysis first):
     python -m llm_lens.examples.run_comparison \
         --base Qwen/Qwen2.5-3B \
         --altered Qwen/Qwen2.5-3B-Instruct \
         --dataset condition_multiple --max-per-class 200

  Auto-detected: if the argument ends with .json, load from file;
  otherwise treat as model name.

Output dir: if --output doesn't end in a timestamp, the current timestamp
is appended automatically. Pass --timestamp to pin a shared value across
multiple invocations.
"""

import os, sys, argparse
from datetime import datetime
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens import (
    ActivationExtractor, BehaviorMapper,
    ReportComparator, CompareVisualizer,
)
from llm_lens.datasets import load_prompts, list_datasets
from llm_lens.report_io import (
    save_report, load_report,
    find_artifacts_for_report, load_class_centroids,
)


def _make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M")


def _resolve_output_dir(output: str, timestamp: str) -> str:
    if "{ts}" in output:
        return output.replace("{ts}", timestamp)
    tail = os.path.basename(os.path.normpath(output))
    if (len(tail) >= 13 and tail[:8].isdigit() and tail[8] == "-"
            and tail[9:].replace("-", "").isdigit()):
        return output
    # Append the timestamp to whatever was given (so e.g. .../A_vs_B → .../A_vs_B/<ts>)
    return os.path.join(output, timestamp)


def build_report_from_model(model_name: str,
                            prompts: dict,
                            save_dir: str = None):
    """Run extraction + analysis on a live model, optionally save."""
    print(f"\n--- Building report for {model_name} ---")
    extractor = ActivationExtractor(model_name, dtype=torch.float32)
    mapper = BehaviorMapper(extractor)

    for tag, tag_prompts in prompts.items():
        print(f"  Extracting [{tag}] ({len(tag_prompts)} prompts)...")
        for p in tag_prompts:
            mapper.add(p, tag)

    report = mapper.analyze()
    report.print_summary()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_name = model_name.replace("/", "_")
        path = os.path.join(save_dir, f"report_{safe_name}.json")
        save_report(report, path)

    del extractor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return report


def get_report(source: str, prompts: dict, save_dir: str = None):
    """Load from JSON or build from model. Auto-detected by .json suffix."""
    if source.endswith(".json"):
        if not os.path.exists(source):
            print(f"ERROR: File not found: {source}")
            sys.exit(1)
        return load_report(source)
    return build_report_from_model(source, prompts=prompts, save_dir=save_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two same-architecture models",
        epilog="Pass .json paths to skip extraction, or model names to run from scratch.",
    )
    parser.add_argument("--base", required=True,
                        help="Model A: model name or path to report JSON")
    parser.add_argument("--altered", required=True,
                        help="Model B: model name or path to report JSON")
    parser.add_argument("--output", default="results/comparison",
                        help="Directory for output plots. Timestamp auto-appended "
                             "if not already present. Use {ts} for explicit slot.")
    parser.add_argument("--save-reports", default="results",
                        help="Save reports when building from model (set 'none' to skip)")
    parser.add_argument("--dataset", default="default", choices=list_datasets(),
                        help="Prompt dataset (only used when building from model names).")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--timestamp", default=None,
                        help="Override timestamp suffix. Defaults to current YYYYMMDD-HHMM.")
    args = parser.parse_args()

    timestamp = args.timestamp or _make_timestamp()
    args.output = _resolve_output_dir(args.output, timestamp)

    save_dir = None if args.save_reports == "none" else args.save_reports

    # Only load prompts if at least one source is a model name
    needs_prompts = not (args.base.endswith(".json") and args.altered.endswith(".json"))
    prompts = load_prompts(args.dataset, max_per_class=args.max_per_class) if needs_prompts else None

    # Load or build
    print(f"Source A: {args.base}")
    print(f"Source B: {args.altered}")
    print(f"Output:   {args.output}")
    report_a = get_report(args.base, prompts=prompts, save_dir=save_dir)
    report_b = get_report(args.altered, prompts=prompts, save_dir=save_dir)

    # Auto-discover sibling artifacts for extended comparisons (centroid npz).
    centroids_a = centroids_b = None
    if args.base.endswith(".json"):
        art = find_artifacts_for_report(args.base)
        if "centroids" in art:
            centroids_a = load_class_centroids(art["centroids"])
    if args.altered.endswith(".json"):
        art = find_artifacts_for_report(args.altered)
        if "centroids" in art:
            centroids_b = load_class_centroids(art["centroids"])

    # Compare
    print(f"\n{'=' * 60}")
    print(f"Comparing: {report_a.model_name} vs {report_b.model_name}")
    print(f"{'=' * 60}")

    comparator = ReportComparator(
        report_a, report_b,
        centroids_a=centroids_a,
        centroids_b=centroids_b,
    )
    result = comparator.run()
    result.print_summary()

    # Visualize
    viz = CompareVisualizer()
    viz.plot_all(result, save_dir=args.output)

    print(f"\nDone. Plots in {args.output}/")


if __name__ == "__main__":
    main()
