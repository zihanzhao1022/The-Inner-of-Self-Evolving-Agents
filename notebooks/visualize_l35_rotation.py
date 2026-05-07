"""Visualize the L35 representation-space rotation across the three 3B models.

Generates four figures into results/L35_rotation_<TS>/:
  1. centroids_panels_layers.png      — 6 panels at L0/8/18/28/33/35, joint PCA
  2. trajectory_3d_mean_class.png     — 3D path of each model's mean centroid through 36 layers
  3. pairwise_distance_vs_layer.png   — same-class centroid distance vs layer (3 pairs)
  4. displacement_L33_to_L35.png      — arrows showing the L33->L35 jump per (class, model)
"""

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


def main():
    DEST = r"D:/Projects/The-Inner-of-Self-Evolving-Agents"
    TS = "20260506-0000"
    OUTDIR = os.path.join(DEST, "results", f"L35_rotation_{TS}")
    os.makedirs(OUTDIR, exist_ok=True)

    REPORTS = {
        "Qwen2.5-3B":          os.path.join(DEST, "results", "Qwen2.5-3B", TS, "report_Qwen_Qwen2.5-3B.json"),
        "Qwen2.5-3B-Instruct": os.path.join(DEST, "results", "Qwen2.5-3B-Instruct", TS, "report_Qwen_Qwen2.5-3B-Instruct.json"),
        "AZR-Coder-3B":        os.path.join(DEST, "results", "AZR-Coder-3B", TS, "report_andrewzh_Absolute_Zero_Reasoner-Coder-3b.json"),
    }
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
    model_markers = {
        "Qwen2.5-3B":          "o",
        "Qwen2.5-3B-Instruct": "s",
        "AZR-Coder-3B":        "^",
    }
    model_edge = {
        "Qwen2.5-3B":          "#000000",
        "Qwen2.5-3B-Instruct": "#1976D2",
        "AZR-Coder-3B":        "#D32F2F",
    }

    class_handles = [Line2D([0], [0], marker="o", color="w",
                            markerfacecolor=class_colors[t], markersize=12,
                            markeredgecolor="black", label=t)
                     for t in tags]
    model_handles = [Line2D([0], [0], marker=model_markers[m], color="w",
                            markerfacecolor="#CCCCCC", markersize=12,
                            markeredgecolor=model_edge[m], lw=1.5, label=m)
                     for m in models]

    # ── FIG 1: 6-panel snapshot at chosen layers ────────────────────────────
    panel_layers = [0, 8, 18, 28, 33, 35]
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
    fig.suptitle("Class centroids in shared PCA — Base / Instruct stay aligned, AZR rotates at L35",
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
                f"{m} L35", fontsize=10, fontweight="bold",
                color=model_edge[m], ha="center")
        ax.text(pts[0, 0], pts[0, 1], pts[0, 2] - 8,
                "L0", fontsize=8, color=model_edge[m], ha="center")
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({ev[2]*100:.1f}%)")
    ax.set_title("Mean-class centroid trajectory through 36 layers (joint PCA)",
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

    pair_labels = [
        (0, 1, "3B  vs  3B-Instruct"),
        (0, 2, "3B  vs  AZR-Coder"),
        (1, 2, "3B-Instruct  vs  AZR-Coder"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6))
    colors = ["#4CAF50", "#F44336", "#FF9800"]
    for (ai, bi, lbl), col in zip(pair_labels, colors):
        d = pair_distances(ai, bi)
        ax.plot(np.arange(L), d, "o-", lw=2, ms=4, color=col, label=lbl)
        # Annotate L35 value
        ax.annotate(f"L35: {d[-1]:.0f}", xy=(35, d[-1]),
                    xytext=(33, d[-1] + (max(d) * 0.06)),
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.6))

    ax.axvline(35, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Euclidean distance between same-class centroids (joint PCA)")
    ax.set_title("Cross-model centroid distance per layer — AZR diverges only at L35",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out3 = os.path.join(OUTDIR, "pairwise_distance_vs_layer.png")
    fig.savefig(out3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out3}")

    # ── FIG 4: L33 -> L35 displacement arrows ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for sub_idx, (pcA, pcB, lbl_ax) in enumerate([
        (0, 1, "PC1, PC2"),
        (0, 2, "PC1, PC3"),
    ]):
        ax = axes[sub_idx]
        for mi, m in enumerate(models):
            for ci, t in enumerate(tags):
                p33 = X_pca[mi, 33, ci, [pcA, pcB]]
                p35 = X_pca[mi, 35, ci, [pcA, pcB]]
                ax.annotate("", xy=p35, xytext=p33,
                            arrowprops=dict(arrowstyle="->",
                                            color=model_edge[m],
                                            lw=1.6, alpha=0.85))
                ax.scatter(*p33, c=class_colors[t], marker=model_markers[m],
                           s=80, edgecolors=model_edge[m], lw=1.0, alpha=0.5)
                ax.scatter(*p35, c=class_colors[t], marker=model_markers[m],
                           s=200, edgecolors=model_edge[m], lw=1.5)
        ax.set_xlabel(f"PC{pcA+1} ({ev[pcA]*100:.1f}%)")
        ax.set_ylabel(f"PC{pcB+1} ({ev[pcB]*100:.1f}%)")
        ax.set_title(f"L33 -> L35 displacement ({lbl_ax})",
                     fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    fig.suptitle("L33 -> L35 displacement of every (class x model) — AZR moves coherently in its own direction",
                 fontsize=14, fontweight="bold")
    fig.legend(handles=class_handles + model_handles,
               loc="lower center", ncol=9, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12, top=0.90)
    out4 = os.path.join(OUTDIR, "displacement_L33_to_L35.png")
    fig.savefig(out4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out4}")

    # ── FIG 5: Procrustes residual per layer ────────────────────────────────
    # Tests whether the apparent L35 displacement is "just rigid transform"
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

    pair_idx = [
        (0, 1, "3B  vs  3B-Instruct",         "#4CAF50"),
        (0, 2, "3B  vs  AZR-Coder",           "#F44336"),
        (1, 2, "3B-Instruct  vs  AZR-Coder",  "#FF9800"),
    ]

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
        ax.annotate(f"L35: {residual[-1]:.4f}",
                    xy=(35, residual[-1]),
                    xytext=(28, residual[-1] + max(residual) * 0.08),
                    fontsize=9, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, alpha=0.6))

    ax.axvline(35, color="gray", ls="--", alpha=0.5)
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

    ax.axvline(35, color="gray", ls="--", alpha=0.5)
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
    fig.suptitle("Class centroids in shared PCA (3D) — PC3 is where Instruct stays with Base while AZR flips sign",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08, top=0.93)
    out6 = os.path.join(OUTDIR, "centroids_panels_layers_3d.png")
    fig.savefig(out6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out6}")

    # ── Quantify the rotation: per-pair angle of mean L35 displacement ──────
    print("\nL33 -> L35 displacement vector statistics (joint PCA, top-3 PCs):")
    for mi, m in enumerate(models):
        v = X_pca[mi, 35].mean(0) - X_pca[mi, 33].mean(0)   # (3,)
        print(f"  {m:22s}  Δmean = ({v[0]:+.1f}, {v[1]:+.1f}, {v[2]:+.1f})  ‖Δ‖={np.linalg.norm(v):.1f}")

    print("\nAngle (degrees) between L33->L35 mean displacement vectors:")
    vs = [X_pca[mi, 35].mean(0) - X_pca[mi, 33].mean(0) for mi in range(M)]
    for i in range(M):
        for j in range(i + 1, M):
            cos = np.dot(vs[i], vs[j]) / (np.linalg.norm(vs[i]) * np.linalg.norm(vs[j]) + 1e-10)
            ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            print(f"  {models[i]:22s} vs {models[j]:22s}: cos={cos:+.3f}  angle={ang:.1f}°")

    # Procrustes residual at key layers (reuse cache)
    print("\nProcrustes disparity at key layers (lower = more identical shape after optimal rigid alignment):")
    print(f"  {'pair':35s}  L0       L18      L33      L35")
    for ai, bi, lbl, _ in pair_idx:
        row = [f"{pair_residuals[(ai, bi)][Li]:.4f}" for Li in [0, 18, 33, 35]]
        print(f"  {lbl:35s}  " + "   ".join(row))

    print(f"\nAll outputs in: {OUTDIR}")


if __name__ == "__main__":
    main()
