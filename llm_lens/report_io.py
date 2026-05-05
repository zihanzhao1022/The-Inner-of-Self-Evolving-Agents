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

import os
from .behavior import BehaviorReport, BehaviorSignature, LayerProbeResult
from .attention import HeadAnalysisResult


def save_report(report: BehaviorReport, path: str):
    """
    Serialize a BehaviorReport to JSON.

    All numpy arrays of bounded size (per-layer scalars / per-(layer,head)
    matrices) are written inline. Heavy artifacts that scale with hidden_dim
    — class_centroids (L, C, d), behavior_direction (L, d), and per-head
    centroids — go to companion .npz files via save_class_centroids /
    save_head_analysis.
    """
    lp = report.layer_probe
    data = {
        "_version": 2,
        "model_name": report.model_name,
        "num_layers": report.num_layers,
        "tags": report.tags,
        "sample_counts": report.sample_counts,

        "layer_probe": {
            "num_layers": lp.num_layers,
            "probe_accuracy": lp.probe_accuracy.tolist(),
            "mean_separation": lp.mean_separation.tolist(),
            "best_layer": lp.best_layer,
            "best_accuracy": float(lp.best_accuracy),
            "best_depth_ratio": float(lp.best_depth_ratio),
            "tags_order": lp.tags_order,
            # Activation-based per-layer arrays (lightweight, fit in JSON):
            "direction_cosine": lp.direction_cosine.tolist()
                if lp.direction_cosine is not None else None,
            "direction_norm": lp.direction_norm.tolist()
                if lp.direction_norm is not None else None,
            "bic_effective_rank": lp.bic_effective_rank.tolist()
                if lp.bic_effective_rank is not None else None,
            "bic_explained_ratio": lp.bic_explained_ratio.tolist()
                if lp.bic_explained_ratio is not None else None,
            "bic_inter_class_ratio": lp.bic_inter_class_ratio.tolist()
                if lp.bic_inter_class_ratio is not None else None,
            # NOTE: class_centroids and behavior_direction are too heavy
            # for JSON. Use save_class_centroids() to persist them as .npz.
        },

        "signatures": {},
        "pairwise_bifurcation": {},
        "head_analysis": None,
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

    # Per-head analysis — store the (num_layers, num_heads) matrices inline.
    # Per-head centroids (heavy) go to .npz via save_head_analysis().
    if report.head_analysis is not None:
        ha = report.head_analysis
        data["head_analysis"] = {
            "num_layers": ha.num_layers,
            "num_heads": ha.num_heads,
            "head_dim": ha.head_dim,
            "tags_order": list(ha.tags_order),
            "head_probe_accuracy": ha.head_probe_accuracy.tolist(),
            "head_separation": ha.head_separation.tolist(),
            "head_inter_class_ratio": ha.head_inter_class_ratio.tolist(),
        }

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
    """Load from save_report() format (v1 or v2)."""
    lp = data["layer_probe"]
    layer_probe = LayerProbeResult(
        num_layers=lp["num_layers"],
        probe_accuracy=np.array(lp["probe_accuracy"]),
        mean_separation=np.array(lp["mean_separation"]),
        best_layer=lp["best_layer"],
        best_accuracy=lp["best_accuracy"],
        best_depth_ratio=lp["best_depth_ratio"],
        tags_order=lp.get("tags_order"),
        direction_cosine=np.array(lp["direction_cosine"]) if lp.get("direction_cosine") is not None else None,
        direction_norm=np.array(lp["direction_norm"]) if lp.get("direction_norm") is not None else None,
        bic_effective_rank=np.array(lp["bic_effective_rank"]) if lp.get("bic_effective_rank") is not None else None,
        bic_explained_ratio=np.array(lp["bic_explained_ratio"]) if lp.get("bic_explained_ratio") is not None else None,
        bic_inter_class_ratio=np.array(lp["bic_inter_class_ratio"]) if lp.get("bic_inter_class_ratio") is not None else None,
        # v2 adds class_centroids and behavior_direction, but they are stored in .npz.
        # Use load_class_centroids() to attach them after this call if needed.
        class_centroids=None,
        behavior_direction=None,
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
        # Ensure CI fields exist (may be absent in pre-bootstrap reports)
        restored.setdefault("cosine_lower", None)
        restored.setdefault("cosine_upper", None)
        pairwise[(ta, tb)] = restored

    # Optional per-head analysis (v2+)
    head_analysis = None
    ha_data = data.get("head_analysis")
    if ha_data is not None:
        head_analysis = HeadAnalysisResult(
            num_layers=ha_data["num_layers"],
            num_heads=ha_data["num_heads"],
            head_dim=ha_data["head_dim"],
            tags_order=list(ha_data["tags_order"]),
            head_probe_accuracy=np.array(ha_data["head_probe_accuracy"]),
            head_separation=np.array(ha_data["head_separation"]),
            head_inter_class_ratio=np.array(ha_data["head_inter_class_ratio"]),
            head_centroids=None,  # heavy — load via load_head_analysis() if needed
        )

    report = BehaviorReport(
        model_name=data["model_name"],
        num_layers=data["num_layers"],
        tags=data["tags"],
        sample_counts=data["sample_counts"],
        layer_probe=layer_probe,
        signatures=signatures,
        pairwise_bifurcation=pairwise,
        head_analysis=head_analysis,
    )
    print(f"Loaded report (new format, v{data.get('_version', 1)}) from {path}: "
          f"{report.model_name}, {report.num_layers} layers"
          + (f", with head analysis ({report.head_analysis.num_heads} heads)"
             if head_analysis is not None else ""))
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
            "cosine_lower": None,
            "cosine_upper": None,
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


# ─────────────────────────────────────────
# Activation-based artifacts (.npz)
# ─────────────────────────────────────────

def save_class_centroids(report: BehaviorReport, npz_path: str):
    """
    Save the heavy activation-based artifacts of LayerProbeResult to .npz.

    Contents:
        class_centroids:        (num_layers, n_classes, hidden_dim)
        behavior_direction:     (num_layers, hidden_dim) — top SVD direction of centered class means
        tags_order:             class label ordering for centroids
        probe_accuracy:         (num_layers,)
        mean_separation:        (num_layers,)
        direction_cosine:       (num_layers-1,)
        direction_norm:         (num_layers,)
        bic_effective_rank:     (num_layers,)
        bic_explained_ratio:    (num_layers,)
        bic_inter_class_ratio:  (num_layers,)
    """
    probe = report.layer_probe
    if probe.class_centroids is None:
        print(f"Warning: no class_centroids in report, skipping npz save.")
        return

    parent = os.path.dirname(npz_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    save_dict = {
        "class_centroids": probe.class_centroids,
        "probe_accuracy": probe.probe_accuracy,
        "mean_separation": probe.mean_separation,
    }
    if probe.behavior_direction is not None:
        save_dict["behavior_direction"] = probe.behavior_direction
    for k in ("direction_cosine", "direction_norm",
              "bic_effective_rank", "bic_explained_ratio", "bic_inter_class_ratio"):
        v = getattr(probe, k)
        if v is not None:
            save_dict[k] = v
    if probe.tags_order is not None:
        save_dict["tags_order"] = np.array(probe.tags_order, dtype=object)

    np.savez_compressed(npz_path, **save_dict)
    print(f"Saved class centroids to {npz_path}  "
          f"[shape: {probe.class_centroids.shape}, "
          f"size: {os.path.getsize(npz_path) / 1024:.0f} KB]")


def load_class_centroids(npz_path: str) -> dict:
    """Load activation-based class centroid artifacts from .npz. Returns dict of arrays."""
    data = dict(np.load(npz_path, allow_pickle=True))
    if "tags_order" in data:
        data["tags_order"] = list(data["tags_order"])
    print(f"Loaded class centroids from {npz_path}")
    return data


def save_head_analysis(head_result: HeadAnalysisResult, npz_path: str):
    """
    Save per-head analysis to .npz.

    Contents:
        head_probe_accuracy:    (num_layers, num_heads)
        head_separation:        (num_layers, num_heads)
        head_inter_class_ratio: (num_layers, num_heads)
        head_centroids:         (num_layers, num_heads, n_classes, head_dim) — if present
        tags_order, num_layers, num_heads, head_dim
    """
    parent = os.path.dirname(npz_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    save_dict = {
        "head_probe_accuracy": head_result.head_probe_accuracy,
        "head_separation": head_result.head_separation,
        "head_inter_class_ratio": head_result.head_inter_class_ratio,
        "tags_order": np.array(head_result.tags_order, dtype=object),
        "num_layers": np.array(head_result.num_layers),
        "num_heads": np.array(head_result.num_heads),
        "head_dim": np.array(head_result.head_dim),
    }
    if head_result.head_centroids is not None:
        save_dict["head_centroids"] = head_result.head_centroids

    np.savez_compressed(npz_path, **save_dict)
    msg = f"Saved head analysis to {npz_path} [{head_result.num_layers}×{head_result.num_heads}"
    if head_result.head_centroids is not None:
        msg += f", with centroids {head_result.head_centroids.shape}"
    msg += f", size: {os.path.getsize(npz_path) / 1024:.0f} KB]"
    print(msg)


def load_head_analysis(npz_path: str) -> HeadAnalysisResult:
    """Load per-head analysis from .npz into a HeadAnalysisResult."""
    data = dict(np.load(npz_path, allow_pickle=True))
    centroids = data.get("head_centroids")
    if centroids is not None and centroids.size == 0:
        centroids = None
    result = HeadAnalysisResult(
        num_layers=int(data["num_layers"]),
        num_heads=int(data["num_heads"]),
        head_dim=int(data["head_dim"]),
        tags_order=list(data["tags_order"]),
        head_probe_accuracy=data["head_probe_accuracy"],
        head_separation=data["head_separation"],
        head_inter_class_ratio=data["head_inter_class_ratio"],
        head_centroids=centroids,
    )
    print(f"Loaded head analysis from {npz_path}")
    return result


# ─────────────────────────────────────────
# Legacy (read-only): old probe-coef npz format
# ─────────────────────────────────────────

def load_probe_vectors(npz_path: str) -> dict:
    """
    [LEGACY] Read an old-format probe-vector .npz file (probe_coefs, etc.).

    The new analysis pipeline replaces probe weights with class centroids;
    use save_class_centroids/load_class_centroids for new data. This loader
    is kept only so the legacy probe_vector_explorer.ipynb can still read
    old npz files captured before the refactor.
    """
    data = dict(np.load(npz_path, allow_pickle=True))
    if "tags_order" in data:
        data["tags_order"] = list(data["tags_order"])
    print(f"Loaded LEGACY probe vectors from {npz_path}")
    return data