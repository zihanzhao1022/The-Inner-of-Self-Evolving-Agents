#!/usr/bin/env python3
"""
Single-model refusal direction visualisations + cross-model group viz.

Four single-model figures per model:
  1. fig_singlemodel_scatter.png        — harmful vs harmless prompts at best
                                            (pos, layer) in 2D PCA. Requires
                                            raw_safe_best / raw_harm_best in npz
                                            (added by --save-raw flag).
  2. fig_singlemodel_self_cosine.png    — (n_pos*L) × (n_pos*L) cosine matrix
                                            of refusal directions across all
                                            (pos, layer) cells. Block structure
                                            shows which (pos, layer) cells share
                                            a direction.
  3. fig_singlemodel_vs_class.png       — cos(r_l, centroid_class_l) vs layer,
                                            6 lines per model (one per IBM class).
                                            Reveals whether refusal direction
                                            aligns with any specific harmful class.
  4. fig_singlemodel_3d_trajectory.png  — refusal direction trajectory through
                                            layers in joint 3D PCA. Best-position
                                            slice only (n_layers points × 1).

Plus 4 group figures per size class (3B / 7B):
  group_scatter.png, group_self_cosine.png, group_vs_class.png,
  group_3d_trajectory.png — same content, all models from the size class
  overlaid / panelled side-by-side.

Reads:
    results/refusal_direction_<TS>/candidate_directions.npz
    results/<model_short>/<phase1_TS>/class_centroids/<...>.npz

Writes everything into the same refusal_direction_<TS>/ dir.

Usage:
    python -m llm_lens.examples.plot_refusal_singlemodel \
        --refusal-dir results/refusal_direction_quartet_arditi_n128 \
        --phase1-ts 20260506-0000 \
        --model-set 3B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.model_zoo import get_models
from llm_lens.report_io import find_artifacts_for_report, load_class_centroids


# ── Helpers ─────────────────────────────────────────────────────────────────

def _model_role(short: str) -> str:
    s = short.lower()
    if "instruct" in s:           return "instruct"
    if "coder" in s and "azr" in s: return "azr_coder"
    if "azr" in s and "base" in s:  return "azr_base"
    if "azr" in s:                  return "azr"
    if "coder" in s:                return "coder"
    return "base"


ROLE_COLOR = {
    "base":      "#000000",
    "instruct":  "#1976D2",
    "coder":     "#388E3C",
    "azr_base":  "#FF6F00",
    "azr_coder": "#D32F2F",
    "azr":       "#D32F2F",
}

ROLE_MARKER = {
    "base":      "o", "instruct": "s", "coder": "D",
    "azr_base":  "v", "azr_coder": "^", "azr": "^",
}

CLASS_COLORS = {
    "base":                "#888888",
    "legal_opinion":       "#1976D2",
    "health_consultation": "#0097A7",
    "sexual_content":      "#FBC02D",
    "hate_speech":         "#D32F2F",
    "crime_planning":      "#7B1FA2",
}


def load_npz_per_model(npz_path: Path, model_shorts: list[str]) -> dict:
    """Pull each model's tensors out of the candidate_directions npz."""
    data = np.load(npz_path)
    keys = set(data.files)
    out = {}
    for short in model_shorts:
        d = {}
        for suffix in ["directions", "norms", "probe_acc",
                       "safe_mean", "harm_mean",
                       "raw_safe_best", "raw_harm_best",
                       "best_pos", "best_layer"]:
            key = f"{short}__{suffix}"
            if key in keys:
                d[suffix] = data[key]
        if "directions" in d:
            out[short] = d
    return out


def load_class_centroids_for(model_short: str,
                               phase1_ts: str,
                               results_root: str) -> tuple[np.ndarray, list[str]] | None:
    """Load class_centroids npz for a model. Returns (centroids[L, C, D], tags)
    or None if not found."""
    # find report.json under results/<model>/<phase1_ts>/
    model_dir = Path(results_root) / model_short / phase1_ts
    reports = list(model_dir.glob("report_*.json"))
    if not reports:
        return None
    art = find_artifacts_for_report(str(reports[0]))
    if "centroids" not in art:
        return None
    c = load_class_centroids(art["centroids"])
    return c["class_centroids"], list(c["tags_order"])


# ── Single-model figures ────────────────────────────────────────────────────

def fig_singlemodel_scatter(model_short: str, d: dict, out_path: Path) -> bool:
    """Harmful vs harmless prompts in 2D PCA at best (pos, layer)."""
    if "raw_safe_best" not in d or "raw_harm_best" not in d:
        return False
    safe = d["raw_safe_best"]
    harm = d["raw_harm_best"]
    best_pos = int(d["best_pos"][0])
    best_layer = int(d["best_layer"][0])
    eoi_len = d["norms"].shape[0]
    pos_label = f"-{eoi_len - best_pos}"

    X = np.concatenate([safe, harm], axis=0)
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    n_safe = len(safe)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(Z[:n_safe, 0], Z[:n_safe, 1], s=18, alpha=0.6,
               c="#388E3C", label=f"harmless (n={n_safe})")
    ax.scatter(Z[n_safe:, 0], Z[n_safe:, 1], s=18, alpha=0.6,
               c="#D32F2F", label=f"harmful (n={len(harm)})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"{model_short} — harmful vs harmless prompts\n"
                 f"at best refusal layer (pos={pos_label}, L{best_layer})")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_singlemodel_self_cosine(model_short: str, d: dict, out_path: Path) -> None:
    """Self-cosine matrix of refusal directions across all (pos, layer) cells."""
    dirs = d["directions"]               # (n_pos, L, D)
    n_pos, L, D = dirs.shape
    flat = dirs.reshape(n_pos * L, D).astype(np.float64)
    flat = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12)
    cos_mat = flat @ flat.T              # ((n_pos*L), (n_pos*L))

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    # Layer block dividers — every n_pos cells is one layer
    for i in range(1, L):
        ax.axhline(i * n_pos - 0.5, color="black", lw=0.3, alpha=0.3)
        ax.axvline(i * n_pos - 0.5, color="black", lw=0.3, alpha=0.3)
    plt.colorbar(im, ax=ax, label="cos(r_i, r_j)")
    ax.set_xlabel(f"(pos, layer) flat index — outer: layer 0..{L-1}, inner: pos 0..{n_pos-1}")
    ax.set_ylabel("(pos, layer)")
    ax.set_title(f"{model_short} — refusal direction self-cosine across "
                 f"all (pos × layer) cells\n"
                 f"(block-diag: same layer; off-diag: cross-layer alignment)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_singlemodel_vs_class(model_short: str, d: dict,
                              centroids: np.ndarray, tags: list[str],
                              out_path: Path) -> None:
    """Per-layer cos(r_l, centroid_l_c) for each of the 6 classes (best position)."""
    dirs = d["directions"]               # (n_pos, L, D)
    norms = d["norms"]
    best_pos = int(d["best_pos"][0])
    eoi_len = norms.shape[0]
    pos_label = f"-{eoi_len - best_pos}"

    L_dirs = dirs.shape[1]
    L_cent = centroids.shape[0]
    L = min(L_dirs, L_cent)
    r = dirs[best_pos, :L].astype(np.float64)             # (L, D)
    r = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-12)
    centroids = centroids[:L].astype(np.float64)          # (L, C, D)
    cents = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-12)

    cos_per_layer = np.einsum("ld,lcd->lc", r, cents)     # (L, C)

    fig, ax = plt.subplots(figsize=(10, 5))
    layers = np.arange(L)
    for ci, t in enumerate(tags):
        ax.plot(layers, cos_per_layer[:, ci], "-", lw=1.5,
                color=CLASS_COLORS.get(t, "#666666"),
                label=t)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"cos(refusal_direction, class_centroid) at pos={pos_label}")
    ax.set_title(f"{model_short} — refusal direction vs each class centroid\n"
                 f"(positive = refusal points 'into' that class's region)")
    ax.legend(fontsize=9, ncol=2, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_singlemodel_3d_trajectory(model_short: str, d: dict, out_path: Path) -> None:
    """3D PCA of refusal direction across layers (best position slice)."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    dirs = d["directions"]
    norms = d["norms"]
    best_pos = int(d["best_pos"][0])
    eoi_len = norms.shape[0]
    pos_label = f"-{eoi_len - best_pos}"

    r = dirs[best_pos].astype(np.float64)                 # (L, D)
    r_unit = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-12)
    pca = PCA(n_components=3, random_state=0)
    Z = pca.fit_transform(r_unit)                         # (L, 3)
    L = Z.shape[0]

    color = ROLE_COLOR[_model_role(model_short)]

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Z[:, 0], Z[:, 1], Z[:, 2], "-o", color=color, lw=1.5, ms=4,
            alpha=0.85, label=model_short)
    # mark first / last layer
    ax.scatter(*Z[0], c=color, s=200, marker="o", edgecolor="black", lw=1.2)
    ax.scatter(*Z[-1], c=color, s=400, marker="*", edgecolor="black", lw=1.2)
    ax.text(*Z[0],  f"  L0",  fontsize=9)
    ax.text(*Z[-1], f"  L{L-1}", fontsize=9, fontweight="bold")
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)")
    ax.set_title(f"{model_short} — refusal direction trajectory\n"
                 f"(best pos={pos_label}, layers L0→L{L-1}, unit-normalised)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Group figures ───────────────────────────────────────────────────────────

def fig_group_scatter(per_model: dict, out_path: Path, model_set: str) -> bool:
    """Side-by-side panels of the harmful/harmless scatter for each model."""
    models_with_raw = [m for m, d in per_model.items() if "raw_safe_best" in d]
    if not models_with_raw:
        return False
    n = len(models_with_raw)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.atleast_2d(axes)
    else:
        axes = np.array(axes).reshape(rows, cols)

    for idx, m in enumerate(models_with_raw):
        ax = axes[idx // cols, idx % cols]
        d = per_model[m]
        safe = d["raw_safe_best"]
        harm = d["raw_harm_best"]
        X = np.concatenate([safe, harm], axis=0)
        pca = PCA(n_components=2, random_state=0)
        Z = pca.fit_transform(X)
        n_safe = len(safe)
        ax.scatter(Z[:n_safe, 0], Z[:n_safe, 1], s=12, alpha=0.5, c="#388E3C", label="harmless")
        ax.scatter(Z[n_safe:, 0], Z[n_safe:, 1], s=12, alpha=0.5, c="#D32F2F", label="harmful")
        bp = int(d["best_pos"][0]); bl = int(d["best_layer"][0])
        eoi_len = d["norms"].shape[0]
        ax.set_title(f"{m}  (pos={-(eoi_len-bp)}, L{bl})", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    # blank unused cells
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.suptitle(f"{model_set}: harmful vs harmless at each model's best refusal layer",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_group_self_cosine(per_model: dict, out_path: Path, model_set: str) -> None:
    n = len(per_model)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4.5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.atleast_2d(axes)
    else:
        axes = np.array(axes).reshape(rows, cols)

    for idx, (m, d) in enumerate(per_model.items()):
        ax = axes[idx // cols, idx % cols]
        dirs = d["directions"]
        n_pos, L, D = dirs.shape
        flat = dirs.reshape(n_pos * L, D).astype(np.float64)
        flat = flat / (np.linalg.norm(flat, axis=-1, keepdims=True) + 1e-12)
        cos_mat = flat @ flat.T
        im = ax.imshow(cos_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(1, L):
            ax.axhline(i * n_pos - 0.5, color="black", lw=0.2, alpha=0.3)
            ax.axvline(i * n_pos - 0.5, color="black", lw=0.2, alpha=0.3)
        ax.set_title(m, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)
    for idx in range(n, rows * cols):
        axes[idx // cols, idx % cols].axis("off")
    fig.suptitle(f"{model_set}: refusal direction self-cosine across (pos × layer) cells",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_group_vs_class(per_model: dict, centroids_by_model: dict,
                        out_path: Path, model_set: str) -> bool:
    """4 panels (one per harmful class), each with all models' cos(r_l, centroid_l_c)."""
    models_with_cent = [m for m in per_model if m in centroids_by_model and centroids_by_model[m] is not None]
    if not models_with_cent:
        return False
    # Pick representative classes — show 4 (3 harmful + 1 base) or all 6
    plot_classes = ["base", "hate_speech", "crime_planning", "sexual_content"]
    # determine layer count from first model
    first = per_model[models_with_cent[0]]
    L = first["directions"].shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for ax_idx, klass in enumerate(plot_classes):
        ax = axes[ax_idx]
        for m in models_with_cent:
            d = per_model[m]
            cents, tags = centroids_by_model[m]
            if klass not in tags:
                continue
            ci = tags.index(klass)
            best_pos = int(d["best_pos"][0])
            L_dirs = d["directions"].shape[1]
            L_cent = cents.shape[0]
            Lm = min(L_dirs, L_cent)
            r = d["directions"][best_pos, :Lm].astype(np.float64)
            r = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-12)
            c = cents[:Lm, ci].astype(np.float64)
            c = c / (np.linalg.norm(c, axis=-1, keepdims=True) + 1e-12)
            cos_l = (r * c).sum(axis=-1)
            ax.plot(np.arange(Lm), cos_l, "-", lw=1.5,
                    color=ROLE_COLOR[_model_role(m)],
                    label=m)
        ax.axhline(0, color="k", lw=0.5, alpha=0.5)
        ax.set_xlabel("Layer")
        ax.set_ylabel("cos(refusal_direction, class_centroid)")
        ax.set_title(f"vs '{klass}' centroid", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{model_set}: refusal direction × class centroid alignment",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def fig_group_3d_trajectory(per_model: dict, out_path: Path, model_set: str) -> None:
    """All models' refusal direction trajectories overlaid in joint 3D PCA."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    # Joint PCA on all models' (best-pos slice) per-layer directions
    all_r = []
    boundaries = []
    cursor = 0
    for m, d in per_model.items():
        best_pos = int(d["best_pos"][0])
        r = d["directions"][best_pos].astype(np.float64)
        r = r / (np.linalg.norm(r, axis=-1, keepdims=True) + 1e-12)
        all_r.append(r)
        boundaries.append((cursor, cursor + r.shape[0], m))
        cursor += r.shape[0]
    X = np.vstack(all_r)
    pca = PCA(n_components=3, random_state=0)
    Z = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    for (a, b, m) in boundaries:
        Zm = Z[a:b]
        color = ROLE_COLOR[_model_role(m)]
        ax.plot(Zm[:, 0], Zm[:, 1], Zm[:, 2], "-o",
                color=color, lw=1.6, ms=3.5, alpha=0.85, label=m)
        ax.scatter(*Zm[0], c=color, s=200, marker="o",
                    edgecolor="black", lw=1.2, zorder=5)
        ax.scatter(*Zm[-1], c=color, s=400, marker="*",
                    edgecolor="black", lw=1.2, zorder=5)
        ax.text(*Zm[-1], f"  {m}", fontsize=8, fontweight="bold")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)")
    ax.set_title(f"{model_set}: joint refusal direction trajectory through layers\n"
                 f"(unit-normalised; circle=L0, star=last layer)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Single-model + group refusal-direction visualizations")
    p.add_argument("--refusal-dir", required=True,
                   help="Path to results/refusal_direction_<TS>/ containing "
                        "candidate_directions.npz")
    p.add_argument("--phase1-ts", required=True,
                   help="Phase-1 timestamp (e.g. 20260506-0000) — used to "
                        "locate class_centroids npz under results/<model>/<ts>/")
    p.add_argument("--model-set", default="3B", choices=["3B", "7B", "14B"])
    p.add_argument("--results-root", default="results")
    p.add_argument("--targets", nargs="+", default=None)
    args = p.parse_args()

    refusal_dir = Path(args.refusal_dir)
    npz_path = refusal_dir / "candidate_directions.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing {npz_path}")

    out_root = refusal_dir / "viz_singlemodel"
    out_root.mkdir(exist_ok=True)
    out_group = refusal_dir / "viz_group"
    out_group.mkdir(exist_ok=True)

    # Resolve which model shorts the npz contains
    data = np.load(npz_path)
    available_shorts = sorted(set(k.split("__")[0] for k in data.files
                                    if not k.startswith("cos__")))
    print(f"  found {len(available_shorts)} models in npz: {available_shorts}")

    # Filter to model_set members
    model_zoo_shorts = [short for _, short in get_models(args.model_set)]
    shorts = [s for s in available_shorts if s in model_zoo_shorts]
    if args.targets:
        shorts = [s for s in shorts if s in args.targets]
    if not shorts:
        raise SystemExit(f"no models from {args.model_set} found in {npz_path}")
    print(f"  plotting for {len(shorts)} models: {shorts}")

    per_model = load_npz_per_model(npz_path, shorts)
    centroids_by_model: dict = {}
    for s in shorts:
        cents = load_class_centroids_for(s, args.phase1_ts, args.results_root)
        centroids_by_model[s] = cents
        if cents is None:
            print(f"  [warn] no centroids for {s} at {args.phase1_ts}")

    # ── Single-model figures ───────────────────────────────────────────────
    print("\nSingle-model figures:")
    for short in shorts:
        d = per_model[short]
        print(f"  {short}:", end=" ")
        # 1: scatter
        ok = fig_singlemodel_scatter(short, d, out_root / f"{short}_scatter.png")
        print("scatter=" + ("OK" if ok else "skip[no_raw]"), end=" ")
        # 2: self-cosine
        fig_singlemodel_self_cosine(short, d, out_root / f"{short}_self_cosine.png")
        print("self_cos=OK", end=" ")
        # 3: vs class centroids
        if centroids_by_model[short] is not None:
            cents, tags = centroids_by_model[short]
            fig_singlemodel_vs_class(short, d, cents, tags,
                                       out_root / f"{short}_vs_class.png")
            print("vs_class=OK", end=" ")
        else:
            print("vs_class=skip[no_cent]", end=" ")
        # 4: 3D trajectory
        fig_singlemodel_3d_trajectory(short, d, out_root / f"{short}_3d_traj.png")
        print("3d_traj=OK")

    # ── Group figures ──────────────────────────────────────────────────────
    print("\nGroup figures:")
    ok = fig_group_scatter(per_model, out_group / "group_scatter.png", args.model_set)
    print(f"  scatter: {'OK' if ok else 'skip[no_raw]'}")
    fig_group_self_cosine(per_model, out_group / "group_self_cosine.png", args.model_set)
    print("  self_cosine: OK")
    ok = fig_group_vs_class(per_model, centroids_by_model,
                             out_group / "group_vs_class.png", args.model_set)
    print(f"  vs_class: {'OK' if ok else 'skip[no_cent]'}")
    fig_group_3d_trajectory(per_model, out_group / "group_3d_trajectory.png", args.model_set)
    print("  3d_trajectory: OK")

    print(f"\nAll outputs in {out_root}/ and {out_group}/")


if __name__ == "__main__":
    main()
