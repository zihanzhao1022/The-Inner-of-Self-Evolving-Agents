"""Visualize the late-layer representation-space trajectory across the
post-training quartet/quintet (base / Instruct / Coder / AZR variants).

Generates six figures into results/L35_rotation_<TS>/ (or
L27_rotation_<TS>/ for 7B etc.):
  1. centroids_panels_layers.png      — 6 panels at depth ratios 0/0.22/0.5/0.78/0.92/1.0
  2. trajectory_3d_mean_class.png     — 3D path of each model's mean centroid
  3. pairwise_distance_vs_layer.png   — same-class centroid distance vs layer
  4. displacement_LN-2_to_LN.png      — arrows showing the last-2-layers jump per (class, model)
  5. procrustes_residual.png          — geometry mismatch after rigid alignment
  6. centroids_panels_layers_3d.png   — fig 1 but 3D PCA panels

Layer indices auto-derived from the loaded reports' num_layers (3B=36, 7B=28, 14B=48).

CLI:
  python notebooks/visualize_l35_rotation.py                            # 3B default
  python notebooks/visualize_l35_rotation.py --model-set 7B --ts 20260510-XXXX
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

# Make local llm_lens importable when running directly from the notebooks/ dir
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.report_io import find_artifacts_for_report, load_class_centroids
from llm_lens.model_zoo import get_models


def main():
    p = argparse.ArgumentParser(
        description="Late-layer rotation viz (auto-adapts to 3B/7B/14B)")
    p.add_argument("--model-set", default="3B", choices=["3B", "7B", "14B"])
    p.add_argument("--ts", default="20260506-0000",
                   help="Timestamp under results/<model_short>/<ts>/")
    p.add_argument("--results-root", default=None,
                   help="Override results/ root (default: <repo>/results)")
    args = p.parse_args()

    DEST = r"D:/Projects/The-Inner-of-Self-Evolving-Agents"
    if args.results_root:
        results_root = args.results_root
    else:
        results_root = os.path.join(DEST, "results")
    TS = args.ts

    # Build REPORTS from the model_zoo so the script auto-includes all
    # members of the size set (quartet for 3B/14B, quintet for 7B).
    REPORTS = {}
    for full, short in get_models(args.model_set):
        safe = full.replace("/", "_")
        REPORTS[short] = os.path.join(results_root, short, TS,
                                       f"report_{safe}.json")
    OUTDIR = os.path.join(results_root, f"L35_rotation_{TS}")
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Loading {len(REPORTS)} models from TS={TS}, model_set={args.model_set}")
    centroids = {s: load_class_centroids(find_artifacts_for_report(p)["centroids"])
                 for s, p in REPORTS.items()}

    models = list(centroids.keys())
    tags   = list(centroids[models[0]]["tags_order"])
    X = np.stack([centroids[m]["class_centroids"] for m in models])  # (M, L, C, D)
    M, L, C, D = X.shape

    # Joint PCA on all (M*L*C, D) centroid vectors
    pca = PCA(n_components=3, random_state=0)
    X_pca = pca.fit_transform(X.reshape(-1, D)).reshape(M, L, C, 3)
    ev = pca.explained_variance_ratio_
    print(f"PCA explained variance (top 3): {ev}  cum={ev.cumsum()[-1]:.3f}")

    class_colors = {
        "base":                "#888888",
        "legal_opinion":       "#1976D2",
        "health_consultation": "#0097A7",
        "sexual_content":      "#FBC02D",
        "hate_speech":         "#D32F2F",
        "crime_planning":      "#7B1FA2",
    }
    # Adaptive marker / colour assignment by model role (inferred from short label).
    # Anything unrecognised falls through to the "other" pool so 7B's AZR-Base-7B
    # / arbitrary additions still get distinct visuals.
    def _role_of(short: str) -> str:
        s = short.lower()
        if "instruct" in s:           return "instruct"
        if "coder" in s and "azr" in s: return "azr_coder"
        if "azr" in s and "base" in s: return "azr_base"
        if "azr" in s:                return "azr"
        if "coder" in s:              return "coder"
        return "base"
    role_marker = {
        "base":      "o",   # circle
        "instruct":  "s",   # square
        "coder":     "D",   # diamond
        "azr_base":  "v",   # downward triangle (only present in 7B quintet)
        "azr_coder": "^",   # upward triangle
        "azr":       "^",
    }
    role_edge = {
        "base":      "#000000",   # black
        "instruct":  "#1976D2",   # blue (RLHF)
        "coder":     "#388E3C",   # green (domain pretrain)
        "azr_base":  "#FF6F00",   # orange (self-evolving direct)
        "azr_coder": "#D32F2F",   # red (self-evolving via coder)
        "azr":       "#D32F2F",
    }
    model_markers = {m: role_marker[_role_of(m)] for m in models}
    model_edge    = {m: role_edge[_role_of(m)]   for m in models}

    class_handles = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=class_colors[t], markersize=12,
                            markeredgecolor="black", label=t)
                     for t in tags]
    model_handles = [Line2D([0], [0], marker=model_markers[m], color="w",
                            markerfacecolor="#CCCCCC", markersize=12,
                            markeredgecolor=model_edge[m], lw=1.5, label=m)
                     for m in models]

    # ── FIG 1: 6-panel snapshot at chosen layers ────────────────────────────
    # Pick 6 representative depths regardless of total layer count, including
    # the very last layer and a near-last one to capture the late rotation.
    panel_layers = sorted(set([
        0,
        max(1, int(round(L * 0.22))),
        int(round(L * 0.50)),
        int(round(L * 0.78)),
        L - 3 if L >= 4 else max(0, L - 2),
        L - 1,
    ]))
    last_layer = L - 1
    second_last = max(0, L - 3)  # used for displacement plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flat

    xall = X_pca[..., 0].flatten()
    yall = X_pca[..., 1].flatten()
    xlim = (xall.min() - 10, xall.max() + 10)
    ylim = (yall.min() - 10, yall.max() + 10)

    for ax, Lt in zip(axes, panel_layers):
        for mi, m in enumerate(models):
            for ci, t in enumerate(tags):
                x, y = X_pca[mi, Lt, ci, 0], X_pca[mi, Lt, ci, 1]
                ax.scatter(x, y, c=class_colors[t], marker=model_markers[m],
                           s=200, edgecolors=model_edge[m], lw=1.5)
        for ci, t in enumerate(tags):
            pts = X_pca[:, Lt, ci, :2]
            ax.plot(pts[:, 0], pts[:, 1], color=class_colors[t], lw=0.8, alpha=0.4)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
        ax.set_title(f"Layer L{Lt}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.legend(handles=class_handles + model_handles,
               loc="lower center", ncol=9, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Class centroids in shared PCA across {len(models)} models  ({args.model_set})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08, top=0.93)
    out1 = os.path.join(OUTDIR, "centroids_panels_layers.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out1}")

    # ── FIG 2: 3D mean-class trajectory through 36 layers ───────────────────
    fig = plt.figure(figsize=(13, 10))
    ax = fig.add_subplot(111, projection="3d")
    mean_traj = X_pca.mean(axis=2)   # (M, L, 3)
    for mi, m in enumerate(models):
        pts = mean_traj[mi]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                color=model_edge[m], lw=2, marker="o", ms=4, alpha=0.85, label=m)
        ax.scatter(*pts[0], color=model_edge[m], s=200, marker="o",
                   edgecolor="black", lw=1.5, zorder=5)
        ax.scatter(*pts[-1], color=model_edge[m], s=400, marker="*",
                   edgecolor="black", lw=1.5, zorder=5)
        ax.text(pts[-1, 0], pts[-1, 1], pts[-1, 2] + 8,
                f"{m} L{last_layer}", fontsize=10, fontweight="bold",
                color=model_edge[m], ha="center")
        ax.text(pts[0, 0], pts[0, 1], pts[0, 2] - 8,
                "L0", fontsize=8, color=model_edge[m], ha="center")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)")
    ax.set_title(f"Mean-class centroid trajectory through {L} layers (joint PCA)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    out2 = os.path.join(OUTDIR, "trajectory_3d_mean_class.png")
    fig.savefig(out2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out2}")

    # ── FIG 3: pairwise centroid distance per layer ─────────────────────────
    def pair_distances(a_idx, b_idx):
        diff = X_pca[a_idx] - X_pca[b_idx]
        d = np.linalg.norm(diff, axis=-1)
        return d.mean(axis=-1)

    # Build pair list: highlight named training-axis pairs in saturated
    # colours, mute the rest. Works for any N-member set.
    from llm_lens.model_zoo import get_axis_pairs
    axis_pairs_named = get_axis_pairs(args.model_set)   # {axis_name: (short_a, short_b)}
    short_to_idx = {m: i for i, m in enumerate(models)}

    AXIS_COLORS = {
        "RLHF":                    "#1976D2",
        "domain":                  "#388E3C",
        "self_evolving":           "#D32F2F",
        "self_evolving_direct":    "#FF6F00",
        "self_evolving_via_coder": "#D32F2F",
        "cross_AZR":               "#9C27B0",
    }
    cross_palette = ["#9E9E9E", "#BCAAA4", "#CE93D8", "#A1887F", "#90A4AE", "#B0BEC5"]

    pair_labels = []   # list of (ai, bi, lbl, color)
    seen_pairs = set()
    # Named axis pairs first
    for axis, (sa, sb) in axis_pairs_named.items():
        if sa not in short_to_idx or sb not in short_to_idx:
            continue
        ai, bi = sorted([short_to_idx[sa], short_to_idx[sb]])
        if (ai, bi) in seen_pairs:
            continue
        col = AXIS_COLORS.get(axis, "#666666")
        pair_labels.append((ai, bi, f"{axis}: {sa}  vs  {sb}", col))
        seen_pairs.add((ai, bi))
    # Remaining cross pairs (all C(N,2))
    cross_idx = 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            if (i, j) in seen_pairs:
                continue
            col = cross_palette[cross_idx % len(cross_palette)]
            cross_idx += 1
            pair_labels.append((i, j, f"(cross) {models[i]}  vs  {models[j]}", col))
            seen_pairs.add((i, j))

    fig, ax = plt.subplots(figsize=(13, 6))
    for (ai, bi, lbl, col) in pair_labels:
        d = pair_distances(ai, bi)
        ax.plot(np.arange(L), d, "o-", lw=2, ms=4, color=col, label=lbl)
        ax.annotate(f"L{last_layer}: {d[-1]:.0f}", xy=(last_layer, d[-1]),
                    xytext=(max(0, last_layer - 2), d[-1] + (max(d) * 0.06)),
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.6))

    ax.axvline(last_layer, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Euclidean distance between same-class centroids (joint PCA)")
    ax.set_title(f"Cross-model centroid distance per layer — {len(models)} models, {len(pair_labels)} pairs",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out3 = os.path.join(OUTDIR, "pairwise_distance_vs_layer.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out3}")

    # ── FIG 4: late-layer displacement arrows (Lsecond_last -> Llast_layer) ─
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for sub_idx, (pcA, pcB, lbl_ax) in enumerate([
        (0, 1, "PC1, PC2"),
        (0, 2, "PC1, PC3"),
    ]):
        ax = axes[sub_idx]
        for mi, m in enumerate(models):
            for ci, t in enumerate(tags):
                p_pre  = X_pca[mi, second_last, ci, [pcA, pcB]]
                p_last = X_pca[mi, last_layer,  ci, [pcA, pcB]]
                ax.annotate("", xy=p_last, xytext=p_pre,
                            arrowprops=dict(arrowstyle="->",
                                            color=model_edge[m],
                                            lw=1.6, alpha=0.85))
                ax.scatter(*p_pre, c=class_colors[t], marker=model_markers[m],
                           s=80, edgecolors=model_edge[m], lw=1.0, alpha=0.5)
                ax.scatter(*p_last, c=class_colors[t], marker=model_markers[m],
                           s=200, edgecolors=model_edge[m], lw=1.5)
        ax.set_xlabel(f"PC{pcA+1} ({ev[pcA]*100:.1f}%)")
        ax.set_ylabel(f"PC{pcB+1} ({ev[pcB]*100:.1f}%)")
        ax.set_title(f"L{second_last} -> L{last_layer} displacement ({lbl_ax})",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"L{second_last} -> L{last_layer} displacement of every (class x model)",
                 fontsize=14, fontweight="bold")
    fig.legend(handles=class_handles + model_handles,
               loc="lower center", ncol=9, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, top=0.90)
    out4 = os.path.join(OUTDIR, f"displacement_L{second_last}_to_L{last_layer}.png")
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out4}")

    # ── FIG 5: Procrustes residual per layer ────────────────────────────────
    # Tests whether the apparent late-layer displacement is "just rigid"
    # (small residual after optimal rotate+scale+translate) or genuine shape
    # change of the 6-class geometry (large residual).
    from scipy.spatial import procrustes as scipy_procrustes

    def fast_procrustes(A_raw, B_raw):
        """Procrustes for 2 small point sets in high-D space.

        Projects to the joint thin-SVD basis first (rank <= 2*n - 1 << d),
        then runs scipy procrustes in the low-D representation. Result is
        identical to running procrustes on the raw (n, d) matrices, but
        avoids the O(d^3) SVD inside scipy when d >> n.
        """
        joint = np.vstack([A_raw, B_raw])
        joint_c = joint - joint.mean(axis=0, keepdims=True)
        # Tall-skinny thin SVD: cost dominated by (n, d) -> (n, n) gram.
        U, S, _ = np.linalg.svd(joint_c, full_matrices=False)
        pts_local = U * S                         # (2n, 2n)  isometric coords
        n = A_raw.shape[0]
        A_local = pts_local[:n]
        B_local = pts_local[n:]
        mtx1, mtx2, disparity = scipy_procrustes(A_local, B_local)
        return mtx1, mtx2, disparity

    # Reuse the dynamic pair_labels list built earlier for fig 3 — same shape
    # (ai, bi, lbl, col), so all axis pairs (3 for quartet, 5 for quintet)
    # plus all cross pairs are included.
    pair_idx = list(pair_labels)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Procrustes residual (= geometry mismatch after optimal rigid alignment)
    ax = axes[0]
    raw_centroids = np.stack([centroids[m]["class_centroids"] for m in models])  # (M, L, C, D)

    pair_residuals = {}      # cache so panel B can reuse it
    pair_pre_cos   = {}
    pair_post_cos  = {}

    for ai, bi, lbl, col in pair_idx:
        residual = np.zeros(L)
        pre_cos  = np.zeros(L)
        post_cos = np.zeros(L)
        for Li in range(L):
            A = raw_centroids[ai, Li]
            B = raw_centroids[bi, Li]
            mtx1, mtx2, disparity = fast_procrustes(A, B)
            residual[Li] = disparity

            # Pre-alignment mean class cosine (raw vectors from origin)
            na = np.linalg.norm(A, axis=1) + 1e-10
            nb = np.linalg.norm(B, axis=1) + 1e-10
            pre_cos[Li] = ((A * B).sum(axis=1) / (na * nb)).mean()

            # Post-alignment cosine inside the standardized Procrustes coords
            n1 = np.linalg.norm(mtx1, axis=1) + 1e-10
            n2 = np.linalg.norm(mtx2, axis=1) + 1e-10
            post_cos[Li] = ((mtx1 * mtx2).sum(axis=1) / (n1 * n2)).mean()

        pair_residuals[(ai, bi)] = residual
        pair_pre_cos[(ai, bi)]   = pre_cos
        pair_post_cos[(ai, bi)]  = post_cos

        ax.plot(np.arange(L), residual, "o-", lw=2, ms=4, color=col, label=lbl)
        ax.annotate(f"L{last_layer}: {residual[-1]:.4f}",
                    xy=(last_layer, residual[-1]),
                    xytext=(max(0, last_layer - 7), residual[-1] + max(residual) * 0.08),
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.6))

    ax.axvline(last_layer, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Procrustes disparity (lower = same shape)")
    ax.set_title("Shape mismatch after optimal rigid alignment\n"
                 "(if just rotation/translation, this stays near 0)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Panel B: pre vs post-alignment cosine at each layer (uses cache)
    ax = axes[1]
    for ai, bi, lbl, col in pair_idx:
        pre_cos  = pair_pre_cos[(ai, bi)]
        post_cos = pair_post_cos[(ai, bi)]
        ax.plot(np.arange(L), pre_cos,  "-",  lw=1.6, color=col, alpha=0.55,
                label=f"{lbl} (raw cos)")
        ax.plot(np.arange(L), post_cos, "--", lw=2.2, color=col,
                label=f"{lbl} (Procrustes-aligned)")

    ax.axvline(last_layer, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean centroid cosine across 6 classes")
    ax.set_title("Cosine before vs after optimal rigid alignment\n"
                 "(if dashed line = 1, the gap is just rotation)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Procrustes analysis — does the 6-class relative geometry survive across models?",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    out5 = os.path.join(OUTDIR, "procrustes_residual.png")
    fig.savefig(out5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out5}")

    # ── FIG 6: 3D version of the 6-panel snapshot ──────────────────────────
    fig = plt.figure(figsize=(20, 13))
    panel_layers_3d = panel_layers
    xall3 = X_pca[..., 0].flatten()
    yall3 = X_pca[..., 1].flatten()
    zall3 = X_pca[..., 2].flatten()
    pad = 8
    xlim3 = (xall3.min() - pad, xall3.max() + pad)
    ylim3 = (yall3.min() - pad, yall3.max() + pad)
    zlim3 = (zall3.min() - pad, zall3.max() + pad)

    for idx, Lt in enumerate(panel_layers_3d):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        for mi, m in enumerate(models):
            for ci, t in enumerate(tags):
                x, y, z = X_pca[mi, Lt, ci]
                ax.scatter(x, y, z, c=class_colors[t], marker=model_markers[m],
                           s=140, edgecolors=model_edge[m], lw=1.2)
        # Connect same-class points across the 3 models
        for ci, t in enumerate(tags):
            pts = X_pca[:, Lt, ci]
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                    color=class_colors[t], lw=0.8, alpha=0.4)

        ax.set_xlim(xlim3); ax.set_ylim(ylim3); ax.set_zlim(zlim3)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.0f}%)", fontsize=9)
        ax.set_ylabel(f"PC2 ({ev[1]*100:.0f}%)", fontsize=9)
        ax.set_zlabel(f"PC3 ({ev[2]*100:.0f}%)", fontsize=9)
        ax.set_title(f"Layer L{Lt}", fontsize=13, fontweight="bold")
        # Consistent viewing angle so panels are visually comparable
        ax.view_init(elev=18, azim=-60)

    fig.legend(handles=class_handles + model_handles,
               loc="lower center", ncol=9, fontsize=10, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Class centroids in shared PCA (3D) — {len(models)} models ({args.model_set})",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08, top=0.93)
    out6 = os.path.join(OUTDIR, "centroids_panels_layers_3d.png")
    fig.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out6}")

    # ── Quantify the rotation: per-pair angle of mean late-layer displacement
    print(f"\nL{second_last} -> L{last_layer} displacement vector statistics (joint PCA, top-3 PCs):")
    for mi, m in enumerate(models):
        v = X_pca[mi, last_layer].mean(0) - X_pca[mi, second_last].mean(0)
        print(f"  {m:22s}  Δmean = ({v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f})  ‖Δ‖={np.linalg.norm(v):.1f}")

    print(f"\nAngle (degrees) between L{second_last}->L{last_layer} mean displacement vectors:")
    vs = [X_pca[mi, last_layer].mean(0) - X_pca[mi, second_last].mean(0) for mi in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            cos = np.dot(vs[i], vs[j]) / (np.linalg.norm(vs[i]) * np.linalg.norm(vs[j]) + 1e-10)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            print(f"  {models[i]:22s} vs {models[j]:22s}: cos={cos:+.3f}  angle={ang:.1f}°")

    # Procrustes residual at key depths (4 sample points distributed through the depth)
    key_layers = sorted(set([
        0,
        int(round(L * 0.5)),
        max(0, last_layer - 2),
        last_layer,
    ]))
    print("\nProcrustes disparity at key layers (lower = more identical shape after optimal rigid alignment):")
    header = f"  {'pair':45s}  " + "   ".join(f"L{l:>2}".ljust(8) for l in key_layers)
    print(header)
    for ai, bi, lbl, _ in pair_idx:
        row = [f"{pair_residuals[(ai, bi)][Li]:.4f}" for Li in key_layers]
        print(f"  {lbl:45s}  " + "   ".join(row))

    print(f"\nAll outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
