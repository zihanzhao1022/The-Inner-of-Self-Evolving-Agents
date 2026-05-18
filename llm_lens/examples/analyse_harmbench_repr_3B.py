#!/usr/bin/env python3
"""
Representation-layer analysis on HarmBench-paired data, 3B quartet.

Inputs (must exist):
  results/20260518-0020/report_<full-model-name>.json
  results/20260518-0020/class_centroids/<short-folder>/20260518-0020.npz
  results/refusal_direction_20260518-0020_harmbench_paired_3B_<SHORT>/
    candidate_directions.npz
    per_model_metrics.json

Outputs:
  results/harmbench_repr_summary_3B/
    summary.json                       — all numeric findings
    summary.md                         — markdown table for paper
    fig_probe_acc_3B.png               — per-layer probe acc, 4 models
    fig_pairwise_cos_layer_3B.png      — within-model harm/harmless centroid
                                         cosine vs layer, 4 models
    fig_cross_model_dir_cos_3B.png     — cross-model refusal direction
                                         cosine at last unpruned layer

Numeric outputs reported:
  per-model:    best_layer, best_acc, emergence_layer,
                centroid cos at last unpruned layer
  cross-model:  refusal-direction cosine for 6 pairs
                (RLHF / domain / self_evolving / 3 misc)
                last-layer centroid Procrustes residual for axis pairs
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

TS = "20260518-0020"
PHASE1_ROOT = f"results/{TS}"
RD_DIR_FMT  = f"results/refusal_direction_{TS}_harmbench_paired_3B_" + "{short}"
OUT_DIR     = "results/harmbench_repr_summary_3B"

# Short label -> (HF short basename used in phase-1 paths, axis-pair role).
# Phase-1 outputs sanitise '/' to '_': Qwen/Qwen2.5-3B -> Qwen_Qwen2.5-3B
MODELS = [
    ("Qwen2.5-3B",          "Qwen_Qwen2.5-3B",                       "Qwen2.5-3B"),
    ("Qwen2.5-3B-Instruct", "Qwen_Qwen2.5-3B-Instruct",              "Qwen2.5-3B-Instruct"),
    ("Qwen2.5-Coder-3B",    "Qwen_Qwen2.5-Coder-3B",                 "Qwen2.5-Coder-3B"),
    ("AZR-Coder-3B",        "andrewzh_Absolute_Zero_Reasoner-Coder-3b",
                            "Absolute_Zero_Reasoner-Coder-3b"),
]

AXIS_PAIRS = {
    "RLHF":          ("Qwen2.5-3B",       "Qwen2.5-3B-Instruct"),
    "domain":        ("Qwen2.5-3B",       "Qwen2.5-Coder-3B"),
    "self_evolving": ("Qwen2.5-Coder-3B", "AZR-Coder-3B"),
}

ALL_PAIRS = [
    ("Qwen2.5-3B",          "Qwen2.5-3B-Instruct"),
    ("Qwen2.5-3B",          "Qwen2.5-Coder-3B"),
    ("Qwen2.5-3B",          "AZR-Coder-3B"),
    ("Qwen2.5-3B-Instruct", "Qwen2.5-Coder-3B"),
    ("Qwen2.5-3B-Instruct", "AZR-Coder-3B"),
    ("Qwen2.5-Coder-3B",    "AZR-Coder-3B"),
]


def safe_cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b) / (na * nb))


def procrustes_residual(A: np.ndarray, B: np.ndarray) -> float:
    """SciPy-equivalent Procrustes disparity (sum of squared residuals after
    optimal rigid alignment), on column-centered & unit-Frobenius-norm
    standardised inputs. Lower = better-aligned. 0 means A and B are related
    by a pure rotation/reflection."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    na = np.linalg.norm(A); nb = np.linalg.norm(B)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    A = A / na; B = B / nb
    U, _, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
    R = U @ Vt
    return float(np.sum((A @ R - B) ** 2))


def load_phase1_report(short_full: str) -> dict:
    p = os.path.join(PHASE1_ROOT, f"report_{short_full}.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_centroids(folder: str) -> dict:
    p = os.path.join(PHASE1_ROOT, "class_centroids", folder, f"{TS}.npz")
    return dict(np.load(p))


def load_refusal_direction(short: str) -> dict:
    d = RD_DIR_FMT.format(short=short)
    npz = np.load(os.path.join(d, "candidate_directions.npz"))
    pm  = json.load(open(os.path.join(d, "per_model_metrics.json")))
    return {
        "npz":          dict(npz),
        "per_model":    pm.get(short, pm.get(list(pm)[0], {})),
    }


def get_last_unpruned_layer(n_layers: int, prune_pct: float = 0.2) -> int:
    """Arditi-style: last 20% of layers excluded from candidate set.
    For 36 layers and 0.2 prune, last unpruned is L27 (3B); for 28, L21 (7B).
    """
    return int(n_layers * (1.0 - prune_pct)) - 1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 1. Load per-model artifacts ───────────────────────────────────
    reports = {}
    cents   = {}
    refdirs = {}
    for short, phase1_basename, centroid_folder in MODELS:
        reports[short] = load_phase1_report(phase1_basename)
        cents[short]   = load_centroids(centroid_folder)
        refdirs[short] = load_refusal_direction(short)

    n_layers = reports[MODELS[0][0]]["num_layers"]
    last_unpruned = get_last_unpruned_layer(n_layers)
    print(f"n_layers = {n_layers}, last unpruned = L{last_unpruned}")

    # ── 2. Per-model probe + emergence ────────────────────────────────
    per_model = {}
    for short, _, _ in MODELS:
        rep = reports[short]
        lp  = rep["layer_probe"]
        rd  = refdirs[short]["per_model"]
        per_model[short] = {
            "n_layers":       rep["num_layers"],
            "best_layer":     lp["best_layer"],
            "best_accuracy":  lp["best_accuracy"],
            "depth_ratio":    lp["best_depth_ratio"],
            "probe_acc_per_layer": lp["probe_accuracy"],
            # Arditi refusal-direction best (pos, layer)
            "ref_dir_best_pos":   int(rd["npz_best_pos"]) if "npz_best_pos" in rd else None,
            "ref_dir_best_layer": rd.get("best_layer"),
            "ref_dir_best_acc":   rd.get("best_acc"),
        }
        # Pull best_pos/best_layer from the npz if metric json was incomplete
        npz = refdirs[short]["npz"]
        bp_key = f"{short}__best_pos"; bl_key = f"{short}__best_layer"
        if bp_key in npz:
            per_model[short]["ref_dir_best_pos"]   = int(npz[bp_key][0])
        if bl_key in npz:
            per_model[short]["ref_dir_best_layer"] = int(npz[bl_key][0])

    # ── 3. Pairwise refusal-direction cosine at last_unpruned layer ───
    # Use position dimension index = best_pos for the donor (left side).
    # Arditi convention: cosine evaluated at the donor's best_pos.
    cross_dir_cos: Dict[str, Dict] = {}
    for a, b in ALL_PAIRS:
        dirs_a = refdirs[a]["npz"][f"{a}__directions"]  # (n_pos, L, D)
        dirs_b = refdirs[b]["npz"][f"{b}__directions"]
        # Pick best (pos, layer) for the donor a, and evaluate cos(a, b) at
        # the same coordinates (Arditi-compatible).
        pa = per_model[a]["ref_dir_best_pos"]
        la = per_model[a]["ref_dir_best_layer"]
        v_a = dirs_a[pa, la, :]
        v_b = dirs_b[pa, la, :]
        c_at_best = safe_cos(v_a, v_b)
        # Also report cosine at last unpruned layer (canonical Arditi-paper choice)
        # at pos = -1 (last eoi token)
        v_a_lu = dirs_a[-1, last_unpruned, :]
        v_b_lu = dirs_b[-1, last_unpruned, :]
        c_at_lu = safe_cos(v_a_lu, v_b_lu)
        # Per-layer cosine at pos=-1 across all layers
        a_norm = dirs_a[-1] / (np.linalg.norm(dirs_a[-1], axis=-1, keepdims=True) + 1e-12)
        b_norm = dirs_b[-1] / (np.linalg.norm(dirs_b[-1], axis=-1, keepdims=True) + 1e-12)
        cos_per_layer = (a_norm.astype(np.float64) * b_norm.astype(np.float64)).sum(axis=-1)
        cross_dir_cos[f"{a}__{b}"] = {
            "cos_at_donor_best": c_at_best,
            "donor_best_pos":    int(pa),
            "donor_best_layer":  int(la),
            "cos_at_last_unpruned": c_at_lu,
            "last_unpruned_layer":  last_unpruned,
            "cos_per_layer_pos-1":  [float(x) for x in cos_per_layer],
        }

    # ── 4. Last-layer centroid cosine + Procrustes per axis-pair ──────
    # Schema: cents[short] has 'class_centroids' of shape (n_layer, n_tags, D)
    # and 'tags_order' of shape (n_tags,) listing tag names in that index order.
    centroid_metrics: Dict[str, Dict] = {}
    for axis, (a, b) in AXIS_PAIRS.items():
        cc_a = cents[a]["class_centroids"]  # (n_layer, n_tags, D)
        cc_b = cents[b]["class_centroids"]
        tags = [str(t) for t in cents[a]["tags_order"]]
        # (n_tag, D) at last unpruned layer
        Ma = cc_a[last_unpruned]  # (n_tags, D)
        Mb = cc_b[last_unpruned]
        per_tag_cos = [safe_cos(Ma[i], Mb[i]) for i in range(len(tags))]
        proc = procrustes_residual(Ma, Mb)
        # Displacement angle between L(N-3) and L(N-1), using tag 0 (harmful)
        L_n3, L_n1 = n_layers - 3, n_layers - 1
        disp_a = cc_a[L_n1, 0] - cc_a[L_n3, 0]
        disp_b = cc_b[L_n1, 0] - cc_b[L_n3, 0]
        disp_cos = safe_cos(disp_a, disp_b)
        disp_angle_deg = float(np.degrees(np.arccos(np.clip(disp_cos, -1.0, 1.0))))
        centroid_metrics[axis] = {
            "models":             [a, b],
            "tags":               tags,
            "per_tag_cos":        per_tag_cos,
            "procrustes_residual": proc,
            "displacement_angle_deg": disp_angle_deg,
            "displacement_cos":   disp_cos,
            "displacement_layers": [L_n3, L_n1],
        }

    # ── 5. Save summary.json ─────────────────────────────────────────
    summary = {
        "_ts":                   TS,
        "n_layers":              n_layers,
        "last_unpruned_layer":   last_unpruned,
        "models":                [s for s, _, _ in MODELS],
        "per_model":             per_model,
        "cross_model_refusal_direction_cosine": cross_dir_cos,
        "centroid_metrics_by_axis":             centroid_metrics,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT_DIR}/summary.json")

    # ── 6. Markdown table ────────────────────────────────────────────
    lines = []
    lines.append(f"# HarmBench-paired representation summary — 3B")
    lines.append(f"")
    lines.append(f"Timestamp: `{TS}`, n_layers={n_layers}, last unpruned L{last_unpruned}.")
    lines.append(f"")
    lines.append(f"## Per-model probe + Arditi refusal direction")
    lines.append(f"")
    lines.append(f"| Model | probe best_acc | probe best_layer | ref_dir best_pos | ref_dir best_layer |")
    lines.append(f"|---|---:|---:|---:|---:|")
    for short, _, _ in MODELS:
        p = per_model[short]
        lines.append(
            f"| {short} | {p['best_accuracy']:.4f} | L{p['best_layer']} | "
            f"{p['ref_dir_best_pos']} | L{p['ref_dir_best_layer']} |"
        )

    lines.append(f"")
    lines.append(f"## Cross-model refusal-direction cosine")
    lines.append(f"")
    lines.append(f"Cosine evaluated at the donor's best (pos, layer) AND at pos=-1, L{last_unpruned}.")
    lines.append(f"")
    lines.append(f"| Pair | cos @ donor-best | cos @ pos=-1, L{last_unpruned} |")
    lines.append(f"|---|---:|---:|")
    for key, v in cross_dir_cos.items():
        a, b = key.split("__")
        lines.append(f"| {a} ↔ {b} | {v['cos_at_donor_best']:.4f} | {v['cos_at_last_unpruned']:.4f} |")

    lines.append(f"")
    lines.append(f"## Centroid distance per axis (last unpruned layer L{last_unpruned})")
    lines.append(f"")
    lines.append(f"| Axis | Pair | tag cos (harmful / harmless) | Procrustes resid | L33→L35 disp angle |")
    lines.append(f"|---|---|---|---:|---:|")
    for axis, m in centroid_metrics.items():
        a, b = m["models"]
        tags = m["tags"]
        cos_str = " / ".join(f"{c:.4f}" for c in m["per_tag_cos"])
        lines.append(f"| {axis} | {a} ↔ {b} | {cos_str} | {m['procrustes_residual']:.6f} | "
                     f"{m['displacement_angle_deg']:.2f}° |")

    md = "\n".join(lines) + "\n"
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(f"wrote {OUT_DIR}/summary.md")

    # ── 7. Figures ───────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 7a. Per-layer probe acc, 4 models
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for short, _, _ in MODELS:
        ax.plot(range(n_layers), per_model[short]["probe_acc_per_layer"],
                label=short, linewidth=2)
    ax.axvspan(int(n_layers * 0.8), n_layers, alpha=0.12, color="red",
               label=f"Arditi prune zone (L{int(n_layers * 0.8)}+)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Binary probe accuracy")
    ax.set_title(f"3B / HarmBench-paired: probe acc by layer")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_probe_acc_3B.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/fig_probe_acc_3B.png")

    # 7b. Cross-model refusal-direction cosine per layer
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, v in cross_dir_cos.items():
        a, b = key.split("__")
        ax.plot(range(n_layers), v["cos_per_layer_pos-1"],
                label=f"{a} ↔ {b}", linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvspan(int(n_layers * 0.8), n_layers, alpha=0.12, color="red")
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(refusal_dir_A, refusal_dir_B)  @ pos=-1")
    ax.set_title("3B / HarmBench-paired: cross-model refusal direction cosine")
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_cross_model_dir_cos_3B.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/fig_cross_model_dir_cos_3B.png")

    # 7c. Per-tag bifurcation cosine vs layer (harm vs harmless centroids), 4 models
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for short, _, _ in MODELS:
        bif = reports[short]["pairwise_bifurcation"]["harmful||harmless"]
        ax.plot(range(n_layers), bif["cosine_per_layer"],
                label=short, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(μ_harmful, μ_harmless)")
    ax.set_title("3B / HarmBench-paired: harm/harmless centroid cosine by layer")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fig_pairwise_cos_layer_3B.png"), dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_DIR}/fig_pairwise_cos_layer_3B.png")

    print()
    print("=== summary ===")
    print(md)


if __name__ == "__main__":
    main()
