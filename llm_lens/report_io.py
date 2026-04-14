"""
Report I/O — Save and load BehaviorReport as JSON with full numpy arrays.

Converts all numpy arrays to lists on save, reconstructs them on load.
This lets you run Phase 1 once, save the report, and reuse it for
comparison / cross-scale analysis without re-running the model.

Usage:
    from llm_lens.report_io import save_report, load_report

    # Save
    save_report(report, "results/report_qwen3b.json")

    # Load
    report = load_report("results/report_qwen3b.json")
"""

from __future__ import annotations

import json
import numpy as np
from typing import Optional

from .behavior import BehaviorReport, BehaviorSignature, LayerProbeResult


def save_report(report: BehaviorReport, path: str):
    """
    Serialize a BehaviorReport to JSON.
    All numpy arrays are converted to lists.
    """
    data = {
        "_version": 1,
        "model_name": report.model_name,
        "num_layers": report.num_layers,
        "tags": report.tags,
        "sample_counts": report.sample_counts,

        "layer_probe": {
            "num_layers": report.layer_probe.num_layers,
            "probe_accuracy": report.layer_probe.probe_accuracy.tolist(),
            "mean_separation": report.layer_probe.mean_separation.tolist(),
            "best_layer": report.layer_probe.best_layer,
            "best_accuracy": float(report.layer_probe.best_accuracy),
            "best_depth_ratio": float(report.layer_probe.best_depth_ratio),
        },

        "signatures": {},
        "pairwise_bifurcation": {},
    }

    # Signatures
    for tag, sig in report.signatures.items():
        sig_data = {
            "tag": sig.tag,
            "n_samples": sig.n_samples,
            "mean_intensity": sig.mean_intensity.tolist(),
            "std_intensity": sig.std_intensity.tolist(),
            "mean_norm": sig.mean_norm.tolist(),
            "mean_most_active_layer": float(sig.mean_most_active_layer),
            "mean_early_change": float(sig.mean_early_change),
            "mean_late_change": float(sig.mean_late_change),
            "logit_lens_consensus": None,
        }
        if sig.logit_lens_consensus is not None:
            sig_data["logit_lens_consensus"] = [
                {"token": tok, "agreement": float(agr)}
                for tok, agr in sig.logit_lens_consensus
            ]
        data["signatures"][tag] = sig_data

    # Pairwise bifurcation — keys are tuples, convert to "tag_a||tag_b"
    for (ta, tb), bif in report.pairwise_bifurcation.items():
        bif_data = {}
        for k, v in bif.items():
            if isinstance(v, np.ndarray):
                bif_data[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                bif_data[k] = v.item()
            else:
                bif_data[k] = v
        data["pairwise_bifurcation"][f"{ta}||{tb}"] = bif_data

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved report to {path}")


def load_report(path: str) -> BehaviorReport:
    """
    Deserialize a BehaviorReport from JSON.
    Supports both old format (from original run_experiment.py) and
    new format (from save_report).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Detect format: new format has "_version" or "layer_probe"
    if "layer_probe" in data:
        return _load_new_format(data, path)
    else:
        return _load_old_format(data, path)


def _load_new_format(data: dict, path: str) -> BehaviorReport:
    """Load from save_report() format."""
    lp = data["layer_probe"]
    layer_probe = LayerProbeResult(
        num_layers=lp["num_layers"],
        probe_accuracy=np.array(lp["probe_accuracy"]),
        mean_separation=np.array(lp["mean_separation"]),
        best_layer=lp["best_layer"],
        best_accuracy=lp["best_accuracy"],
        best_depth_ratio=lp["best_depth_ratio"],
    )

    signatures = {}
    for tag, sd in data["signatures"].items():
        consensus = None
        if sd.get("logit_lens_consensus") is not None:
            consensus = [(item["token"], item["agreement"])
                         for item in sd["logit_lens_consensus"]]

        signatures[tag] = BehaviorSignature(
            tag=sd["tag"],
            n_samples=sd["n_samples"],
            mean_intensity=np.array(sd["mean_intensity"]),
            std_intensity=np.array(sd["std_intensity"]),
            mean_norm=np.array(sd["mean_norm"]),
            mean_most_active_layer=sd["mean_most_active_layer"],
            mean_early_change=sd["mean_early_change"],
            mean_late_change=sd["mean_late_change"],
            logit_lens_consensus=consensus,
        )

    pairwise = {}
    for key_str, bif_data in data["pairwise_bifurcation"].items():
        ta, tb = key_str.split("||")
        restored = {}
        for k, v in bif_data.items():
            if isinstance(v, list):
                restored[k] = np.array(v)
            else:
                restored[k] = v
        pairwise[(ta, tb)] = restored

    report = BehaviorReport(
        model_name=data["model_name"],
        num_layers=data["num_layers"],
        tags=data["tags"],
        sample_counts=data["sample_counts"],
        layer_probe=layer_probe,
        signatures=signatures,
        pairwise_bifurcation=pairwise,
    )
    print(f"Loaded report (new format) from {path}: {report.model_name}, {report.num_layers} layers")
    return report


def _load_old_format(data: dict, path: str) -> BehaviorReport:
    """
    Load from the old run_experiment.py format.
    Missing fields are filled with zeros or reasonable defaults.

    Old format keys:
        model, num_layers, best_probe_layer, best_probe_depth,
        best_probe_accuracy, probe_accuracy_per_layer,
        signatures.{tag}.{mean_most_active_layer, mean_most_active_depth,
                          early_change, late_change, intensity_profile},
        bifurcation.{tag_a_vs_tag_b}.{bifurcation_layer, bifurcation_depth,
                                       sharpest_drop_layer, sharpest_drop_depth}
    """
    model_name = data.get("model") or data.get("model_name", "unknown")
    num_layers = data["num_layers"]
    probe_acc = np.array(data["probe_accuracy_per_layer"])

    # Layer probe — mean_separation is missing, fill with zeros
    layer_probe = LayerProbeResult(
        num_layers=num_layers,
        probe_accuracy=probe_acc,
        mean_separation=np.zeros(num_layers),  # not available in old format
        best_layer=data["best_probe_layer"],
        best_accuracy=data["best_probe_accuracy"],
        best_depth_ratio=data["best_probe_depth"],
    )

    # Signatures — old format lacks std_intensity, mean_norm, n_samples
    signatures = {}
    for tag, sd in data.get("signatures", {}).items():
        intensity = np.array(sd["intensity_profile"])
        signatures[tag] = BehaviorSignature(
            tag=tag,
            n_samples=0,  # unknown
            mean_intensity=intensity,
            std_intensity=np.zeros_like(intensity),  # not available
            mean_norm=np.zeros(num_layers),           # not available
            mean_most_active_layer=sd["mean_most_active_layer"],
            mean_early_change=sd["early_change"],
            mean_late_change=sd["late_change"],
            logit_lens_consensus=None,
        )

    tags = sorted(signatures.keys())

    # Sample counts — not available, set to 0
    sample_counts = {tag: 0 for tag in tags}

    # Pairwise bifurcation — old format uses "tag_a_vs_tag_b" separator
    # and lacks cosine_per_layer
    pairwise = {}
    for key_str, bif_data in data.get("bifurcation", {}).items():
        # Parse "harmful_vs_safe" -> ("harmful", "safe")
        # Handle multi-word tags: split on "_vs_" (not just "_")
        parts = key_str.split("_vs_")
        if len(parts) == 2:
            ta, tb = parts
        else:
            # Fallback: skip malformed keys
            continue

        pairwise[(ta, tb)] = {
            "cosine_per_layer": np.zeros(num_layers),  # not available
            "bifurcation_layer": bif_data.get("bifurcation_layer"),
            "bifurcation_depth_ratio": bif_data.get("bifurcation_depth"),
            "sharpest_drop_layer": bif_data.get("sharpest_drop_layer"),
            "sharpest_drop_depth_ratio": bif_data.get("sharpest_drop_depth"),
        }

    report = BehaviorReport(
        model_name=model_name,
        num_layers=num_layers,
        tags=tags,
        sample_counts=sample_counts,
        layer_probe=layer_probe,
        signatures=signatures,
        pairwise_bifurcation=pairwise,
    )

    missing = []
    if all(s.std_intensity.sum() == 0 for s in signatures.values()):
        missing.append("std_intensity")
    if all(s.mean_norm.sum() == 0 for s in signatures.values()):
        missing.append("mean_norm")
    if layer_probe.mean_separation.sum() == 0:
        missing.append("mean_separation")

    print(f"Loaded report (old format) from {path}: {model_name}, {num_layers} layers")
    if missing:
        print(f"  Warning: fields filled with zeros (not in old format): {missing}")
        print(f"  Re-run Phase 1 with updated run_experiment.py for complete data.")

    return report