"""
Lens Visualizer — All visualization in one place.

Groups:
1. Dynamics views — single model layer-wise dynamics
2. Behavior views — cross-behavior comparison
3. Cross-scale views — critical window hypothesis
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from typing import Optional

from .dynamics import DynamicsProfile
from .behavior import BehaviorMapper, BehaviorReport, BehaviorSignature
from .cross_scale import CriticalWindow, ScaleComparisonResult
from .attention import HeadAnalysisResult

TAG_COLORS = {
    "harmful": "#E53935", "safe": "#43A047", "biased": "#FB8C00",
    "hallucination": "#8E24AA", "refusal": "#1E88E5", "creative": "#00ACC1",
    "factual": "#7CB342", "jailbreak": "#D81B60", "sycophancy": "#FF7043",
    "reasoning": "#5C6BC0",
}

def _color(tag):
    import matplotlib.colors as mc
    return mc.to_rgb(TAG_COLORS.get(tag, "#888888"))


class LensVisualizer:
    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass

    # ─────────────────────────────────────────
    # 1. BEHAVIOR VIEWS
    # ─────────────────────────────────────────

    def plot_probe_accuracy(self, report: BehaviorReport, save_path=None) -> plt.Figure:
        """Bar chart: linear probe accuracy at each layer. Highlights best layer."""
        probe = report.layer_probe
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(probe.num_layers)
        bars = ax.bar(x, probe.probe_accuracy, color="#2196F3", alpha=0.6)
        bars[probe.best_layer].set_color("#FF5722")
        bars[probe.best_layer].set_alpha(0.9)

        ax.axhline(0.5, color="gray", ls="--", alpha=0.4, label="Chance")
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("Probe Accuracy", fontsize=13)
        ax.set_ylim(0.35, 1.05)

        ax.annotate(f"Best: L{probe.best_layer}\n(depth {probe.best_depth_ratio:.2f})\nacc={probe.best_accuracy:.3f}",
                    xy=(probe.best_layer, probe.best_accuracy),
                    xytext=(probe.best_layer + 2, probe.best_accuracy - 0.12),
                    arrowprops=dict(arrowstyle="->", color="#FF5722"),
                    fontsize=10, color="#FF5722", fontweight="bold")

        ax.set_title(f"Layer Discriminability — {report.model_name}", fontsize=14, fontweight="bold")
        ax.legend()
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_intensity_comparison(self, report: BehaviorReport, save_path=None) -> plt.Figure:
        """Overlaid processing intensity curves for each behavior."""
        fig, ax = plt.subplots(figsize=self.figsize)

        for tag, sig in report.signatures.items():
            c = _color(tag)
            x = np.arange(len(sig.mean_intensity))
            ax.plot(x, sig.mean_intensity, "o-", color=c, lw=2, ms=4, label=tag)
            ax.fill_between(x, sig.mean_intensity - sig.std_intensity,
                            sig.mean_intensity + sig.std_intensity, color=c, alpha=0.12)

        ax.set_xlabel("Layer Transition (i → i+1)", fontsize=12)
        ax.set_ylabel("Processing Intensity (1 − cos sim)", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_title(f"Processing Intensity by Behavior — {report.model_name}",
                     fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_behavior_trajectories(self, mapper: BehaviorMapper,
                                    tags=None, save_path=None) -> plt.Figure:
        """PCA trajectories colored by behavior tag."""
        if tags is None:
            tags = sorted(mapper._tags)

        samples = [s for s in mapper.samples if s.tag in tags]
        num_layers = samples[0].extraction.num_layers

        # Collect all vectors for joint PCA
        all_vecs = []
        for s in samples:
            traj = s.extraction.get_residuals(-1)
            all_vecs.append(traj.numpy())
        all_vecs = np.concatenate(all_vecs, axis=0)

        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_vecs)

        fig, ax = plt.subplots(figsize=(10, 8))

        for i, s in enumerate(samples):
            pts = projected[i * num_layers: (i + 1) * num_layers]
            c = _color(s.tag)
            alpha = 0.3 if len(samples) > 20 else 0.5
            ax.plot(pts[:, 0], pts[:, 1], "-", color=c, alpha=alpha, lw=1)

        # Mean trajectories
        for tag in tags:
            tag_s = [s for s in samples if s.tag == tag]
            if not tag_s:
                continue
            mean_pts = np.zeros((num_layers, 2))
            for s in tag_s:
                idx = samples.index(s)
                mean_pts += projected[idx * num_layers: (idx + 1) * num_layers]
            mean_pts /= len(tag_s)
            c = _color(tag)
            ax.plot(mean_pts[:, 0], mean_pts[:, 1], "-", color=c, lw=3, alpha=0.9, label=f"{tag}")
            ax.annotate("L0", mean_pts[0], fontsize=8, fontweight="bold", color=c)
            ax.annotate(f"L{num_layers - 1}", mean_pts[-1], fontsize=8, fontweight="bold", color=c)

        vr = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({vr[0]:.1%})", fontsize=12)
        ax.set_ylabel(f"PC2 ({vr[1]:.1%})", fontsize=12)
        ax.legend(fontsize=11)
        ax.set_title("Representation Trajectories by Behavior", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_logit_lens_behavior(self, mapper: BehaviorMapper,
                                 tags=None, save_path=None) -> plt.Figure:
        """Logit Lens top-1 prediction per layer, per behavior."""
        if tags is None:
            tags = sorted(mapper._tags)

        from collections import Counter
        num_layers = mapper.samples[0].extraction.num_layers
        n_tags = len(tags)

        fig, axes = plt.subplots(n_tags, 1, figsize=(max(14, num_layers * 0.5), 2.5 * n_tags), sharex=True)
        if n_tags == 1: axes = [axes]

        for ax, tag in zip(axes, tags):
            tag_s = [s for s in mapper.samples if s.tag == tag and s.logit_lens is not None]
            if not tag_s: continue
            c_base = _color(tag)

            for layer in range(num_layers):
                top1s = [s.logit_lens.top_tokens[layer][0] for s in tag_s]
                counts = Counter([t[0] for t in top1s])
                tok, cnt = counts.most_common(1)[0]
                consensus = cnt / len(top1s)

                rgba = (*c_base, 0.3 + 0.7 * consensus)
                ax.add_patch(plt.Rectangle((layer, 0), 1, 1, facecolor=rgba, edgecolor="white", lw=0.5))
                label = repr(tok).strip("'")
                if len(label) > 7: label = label[:6] + "…"
                ax.text(layer + 0.5, 0.6, label, ha="center", va="center", fontsize=7, fontweight="bold" if consensus > 0.7 else "normal")
                ax.text(layer + 0.5, 0.3, f"{consensus:.0%}", ha="center", va="center", fontsize=5, color="gray")

            ax.set_xlim(0, num_layers)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(tag, fontsize=12, fontweight="bold", color=c_base)

        axes[-1].set_xlabel("Layer", fontsize=12)
        fig.suptitle("Logit Lens: What the model 'wants to say' per behavior", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_bifurcation(self, report: BehaviorReport, save_path=None) -> plt.Figure:
        """Pairwise cosine similarity across layers with bootstrap CI bands."""
        pairs = report.pairwise_bifurcation
        if not pairs:
            raise ValueError("No pairwise bifurcation data in report")

        fig, ax = plt.subplots(figsize=self.figsize)
        for (ta, tb), bif in pairs.items():
            cos = bif["cosine_per_layer"]
            x = np.arange(len(cos))
            line = ax.plot(x, cos, "o-", lw=2, ms=4, label=f"{ta} vs {tb}")
            color = line[0].get_color()

            lower = bif.get("cosine_lower")
            upper = bif.get("cosine_upper")
            if lower is not None and upper is not None:
                ax.fill_between(x, lower, upper, color=color, alpha=0.12)

            if bif["bifurcation_layer"] is not None:
                ax.axvline(bif["bifurcation_layer"], ls="--", alpha=0.5, color=color)

        ax.axhline(0.9, color="gray", ls=":", alpha=0.5, label="Threshold (0.9)")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Cosine Similarity (mean representations)", fontsize=12)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=10)
        ax.set_title(f"Behavioral Bifurcation — {report.model_name}", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────
    # 1b. BEHAVIOR DIRECTION VIEWS (NEW)
    # ─────────────────────────────────────────

    def plot_direction_dynamics(self, report: BehaviorReport, save_path=None) -> plt.Figure:
        """Cross-layer behavior direction stability and magnitude.

        Direction = top SVD direction of centered class means (activation-based,
        not probe-weight-based). Sign is flipped arbitrarily by SVD, so the
        cosine plot uses absolute value.
        """
        probe = report.layer_probe
        if probe.direction_cosine is None:
            raise ValueError("No direction dynamics in report. Re-run with updated BehaviorMapper.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: direction cosine similarity between adjacent layers
        x_cos = np.arange(len(probe.direction_cosine))
        ax1.plot(x_cos, probe.direction_cosine, "o-", color="#4ecdc4", lw=2, ms=4)
        ax1.axhline(0.9, color="gray", ls="--", alpha=0.4, label="High stability (0.9)")
        ax1.set_xlabel("Layer Transition (i → i+1)", fontsize=12)
        ax1.set_ylabel("|cos(top inter-class direction)|", fontsize=12)
        ax1.set_title("Behavior Direction Stability (Activation-Based)",
                      fontsize=13, fontweight="bold")
        ax1.set_ylim(-0.1, 1.05)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Shade phases
        n = probe.num_layers
        ax1.axvspan(0, n * 0.3, alpha=0.05, color="#4ecdc4")
        ax1.axvspan(n * 0.3, n * 0.7, alpha=0.05, color="#ffd93d")
        ax1.axvspan(n * 0.7, n, alpha=0.05, color="#6bcb77")
        ax1.text(n * 0.15, -0.05, "Formation", ha="center", fontsize=9, color="#4ecdc4")
        ax1.text(n * 0.5, -0.05, "Differentiation", ha="center", fontsize=9, color="#daa520")
        ax1.text(n * 0.85, -0.05, "Crystallization", ha="center", fontsize=9, color="#6bcb77")

        # Right: inter-class signal magnitude (sqrt(tr Σ_b))
        x_norm = np.arange(len(probe.direction_norm))
        ax2.plot(x_norm, probe.direction_norm, "s-", color="#ff6b6b", lw=2, ms=4)
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel(r"$\sqrt{\mathrm{tr}(\Sigma_b)}$  (inter-class signal magnitude)", fontsize=12)
        ax2.set_title("Inter-Class Signal Strength (Activation-Based)",
                      fontsize=13, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        # Mark peak
        peak_layer = int(np.argmax(probe.direction_norm))
        peak_val = probe.direction_norm[peak_layer]
        ax2.annotate(f"Peak: L{peak_layer}\n({peak_val:.3f})",
                     xy=(peak_layer, peak_val),
                     xytext=(peak_layer + 2, peak_val * 0.85),
                     arrowprops=dict(arrowstyle="->", color="#ff6b6b"),
                     fontsize=10, color="#ff6b6b", fontweight="bold")

        fig.suptitle(f"Behavior Direction Dynamics — {report.model_name}",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_bic(self, report: BehaviorReport, save_path=None) -> plt.Figure:
        """Behavioral Information Concentration (activation-based).

        All three metrics are derived from the activation matrix X and class
        labels y at each layer; no probe weights involved.
        """
        probe = report.layer_probe
        if probe.bic_effective_rank is None:
            raise ValueError("No BIC data in report. Re-run with updated BehaviorMapper.")

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        x = np.arange(probe.num_layers)

        # 1. Effective rank of centered class-mean spectrum
        ax1.plot(x, probe.bic_effective_rank, "o-", color="#9C27B0", lw=2, ms=4)
        ax1.set_xlabel("Layer", fontsize=12)
        ax1.set_ylabel("exp(entropy(SVD spectrum))", fontsize=12)
        ax1.set_title("Class-Mean Subspace Effective Rank",
                      fontsize=13, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        n_classes = len(report.tags)
        ax1.axhline(n_classes - 1, color="gray", ls=":", alpha=0.4,
                    label=f"max = n_classes − 1 = {n_classes - 1}")
        ax1.legend(fontsize=10)

        # 2. Variance in inter-class subspace / total
        ax2.plot(x, probe.bic_explained_ratio, "s-", color="#FF9800", lw=2, ms=4)
        ax2.set_xlabel("Layer", fontsize=12)
        ax2.set_ylabel("var in top inter-class dirs / total var", fontsize=12)
        ax2.set_title("Inter-Class Subspace Coverage",
                      fontsize=13, fontweight="bold")
        ax2.set_ylim(0, max(0.1, probe.bic_explained_ratio.max() * 1.3))
        ax2.grid(True, alpha=0.3)

        # 3. tr(Σ_b) / tr(Σ_t) — direct LDA trace ratio on activations
        ax3.plot(x, probe.bic_inter_class_ratio, "D-", color="#2196F3", lw=2, ms=4)
        ax3.set_xlabel("Layer", fontsize=12)
        ax3.set_ylabel(r"$\mathrm{tr}(\Sigma_b)\,/\,\mathrm{tr}(\Sigma_t)$", fontsize=12)
        ax3.set_title("LDA Trace Ratio (Inter-Class / Total)",
                      fontsize=13, fontweight="bold")
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)

        fig.suptitle(f"Behavioral Information Concentration — {report.model_name}",
                     fontsize=15, fontweight="bold", y=1.02)
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_layer_centroid_heatmap(self, source, save_path=None) -> plt.Figure:
        """
        Cross-layer cosine similarity of class-centroid matrices (activation-based).

        At each layer we have a class-centroid matrix C_l (n_classes, hidden_dim).
        Flatten per layer and compute pairwise cosine sim between all layers —
        this is the activation-based analog of the old probe_coef heatmap.

        Args:
            source: either a BehaviorReport (with class_centroids attached),
                    or a path to a class-centroids .npz file produced by
                    save_class_centroids.
        """
        if isinstance(source, BehaviorReport):
            centroids = source.layer_probe.class_centroids
            if centroids is None:
                raise ValueError("Report has no class_centroids. Re-run analysis or "
                                 "load them from npz with load_class_centroids().")
        else:
            data = np.load(source, allow_pickle=True)
            if "class_centroids" not in data:
                raise ValueError(f"{source} has no 'class_centroids' array.")
            centroids = data["class_centroids"]

        num_layers, n_classes, hidden_dim = centroids.shape

        # Subtract per-layer grand mean before flattening — focus on inter-class
        # geometry, not the absolute residual-stream offset that dominates norms.
        grand = centroids.mean(axis=1, keepdims=True)  # (L, 1, d)
        centered = centroids - grand                   # (L, C, d)
        flat = centered.reshape(num_layers, -1)

        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = flat / norms
        sim_matrix = normalized @ normalized.T

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(sim_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Layer", fontsize=12)
        ax.set_title("Cross-Layer Class-Centroid Similarity (Activation-Based)",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

        tick_pos = list(range(0, num_layers, 5))
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)

        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────
    # 1c. ATTENTION-HEAD VIEWS (NEW)
    # ─────────────────────────────────────────

    def plot_head_probe_heatmap(self, head_result: HeadAnalysisResult,
                                top_k_mark: int = 5,
                                save_path=None) -> plt.Figure:
        """Per-head probe accuracy heatmap. Rows = heads, cols = layers.

        The K most discriminative heads (globally) are outlined in cyan.
        """
        H, L = head_result.num_heads, head_result.num_layers
        fig, ax = plt.subplots(figsize=(max(12, L * 0.35), max(6, H * 0.35)))

        im = ax.imshow(head_result.head_probe_accuracy.T, aspect="auto",
                       cmap="YlOrRd", vmin=0.0, vmax=1.0,
                       origin="lower")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Head", fontsize=12)
        ax.set_title(f"Per-Head Probe Accuracy "
                     f"(best={head_result.head_probe_accuracy.max():.3f}, "
                     f"mean={head_result.head_probe_accuracy.mean():.3f})",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Probe Accuracy", shrink=0.8)

        for l, h, _ in head_result.get_top_heads(top_k_mark):
            ax.add_patch(plt.Rectangle((l - 0.5, h - 0.5), 1, 1,
                                        fill=False, edgecolor="cyan", lw=2))

        ax.set_xticks(list(range(0, L, max(1, L // 18))))
        ax.set_yticks(list(range(0, H, max(1, H // 16))))
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_top_heads(self, head_result: HeadAnalysisResult, k: int = 15,
                       save_path=None) -> plt.Figure:
        """Bar chart of top-K most behavior-discriminative heads."""
        top = head_result.get_top_heads(k)
        labels = [f"L{l}H{h}" for l, h, _ in top]
        accs = [acc for _, _, acc in top]

        fig, ax = plt.subplots(figsize=(max(10, k * 0.6), 6))
        ax.bar(range(k), accs, color="#FF5722", alpha=0.85)
        ax.set_xticks(range(k))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylabel("Probe Accuracy", fontsize=12)
        ax.set_title(f"Top-{k} Behavior-Discriminative Attention Heads",
                     fontsize=14, fontweight="bold")
        ax.axhline(1.0 / max(1, len(head_result.tags_order)), color="gray", ls="--",
                   alpha=0.4, label="chance (uniform)")
        ax.set_ylim(0.0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_head_separation(self, head_result: HeadAnalysisResult,
                             save_path=None) -> plt.Figure:
        """Per-head class-centroid cosine distance heatmap (activation-based)."""
        H, L = head_result.num_heads, head_result.num_layers
        fig, ax = plt.subplots(figsize=(max(12, L * 0.35), max(6, H * 0.35)))

        im = ax.imshow(head_result.head_separation.T, aspect="auto",
                       cmap="viridis", origin="lower")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Head", fontsize=12)
        ax.set_title("Per-Head Mean Pairwise Centroid Cosine Distance",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Cosine Distance", shrink=0.8)
        ax.set_xticks(list(range(0, L, max(1, L // 18))))
        ax.set_yticks(list(range(0, H, max(1, H // 16))))
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_head_inter_class_ratio(self, head_result: HeadAnalysisResult,
                                    save_path=None) -> plt.Figure:
        """Per-head LDA trace ratio tr(Σ_b)/tr(Σ_t) heatmap."""
        H, L = head_result.num_heads, head_result.num_layers
        fig, ax = plt.subplots(figsize=(max(12, L * 0.35), max(6, H * 0.35)))

        im = ax.imshow(head_result.head_inter_class_ratio.T, aspect="auto",
                       cmap="magma", origin="lower")
        ax.set_xlabel("Layer", fontsize=12)
        ax.set_ylabel("Head", fontsize=12)
        ax.set_title(r"Per-Head $\mathrm{tr}(\Sigma_b)\,/\,\mathrm{tr}(\Sigma_t)$",
                     fontsize=14, fontweight="bold")
        fig.colorbar(im, ax=ax, label="Inter-Class Ratio", shrink=0.8)
        ax.set_xticks(list(range(0, L, max(1, L // 18))))
        ax.set_yticks(list(range(0, H, max(1, H // 16))))
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ─────────────────────────────────────────
    # 2. CROSS-SCALE VIEWS
    # ─────────────────────────────────────────

    def plot_critical_window(self, window: CriticalWindow, save_path=None) -> plt.Figure:
        """Visualize the critical window across model scales."""
        fig, ax = plt.subplots(figsize=(10, 6))

        models = window.models
        depths = window.best_probe_depths
        y_pos = np.arange(len(models))

        # Bars showing best probe depth
        bars = ax.barh(y_pos, depths, height=0.5, color="#2196F3", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{m}\n({window.num_layers[i]}L)" for i, m in enumerate(models)], fontsize=10)
        ax.set_xlabel("Best Probe Depth Ratio", fontsize=13)

        # Critical window band
        ax.axvspan(window.window_start, window.window_end, alpha=0.15, color="#FF9800",
                   label=f"Critical window [{window.window_start:.2f}, {window.window_end:.2f}]")
        ax.axvline(window.window_center, color="#FF9800", ls="--", lw=2, alpha=0.7)

        ax.set_xlim(0, 1)
        ax.legend(fontsize=11, loc="lower right")
        ax.set_title(f"Critical Window Hypothesis (CV={window.probe_depth_cv:.3f}, "
                     f"{'STABLE' if window.is_stable else 'UNSTABLE'})",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_normalized_comparison(self, comp: ScaleComparisonResult,
                                   save_path=None) -> plt.Figure:
        """Overlay normalized probe accuracy curves from different models."""
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
        x = np.linspace(0, 1, len(next(iter(comp.normalized_probe_accuracy.values()))))

        for i, model in enumerate(comp.models):
            c = colors[i % len(colors)]
            curve = comp.normalized_probe_accuracy[model]
            ax.plot(x, curve, "-", color=c, lw=2, label=model)

        ax.set_xlabel("Normalized Depth (0=first layer, 1=last layer)", fontsize=12)
        ax.set_ylabel("Probe Accuracy", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title("Normalized Probe Accuracy Across Scales", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_scale_intensity_comparison(self, comp: ScaleComparisonResult,
                                         tag: str, save_path=None) -> plt.Figure:
        """Compare processing intensity profiles for one behavior across scales."""
        fig, ax = plt.subplots(figsize=self.figsize)

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]
        for i, model in enumerate(comp.models):
            if tag in comp.normalized_intensity.get(model, {}):
                c = colors[i % len(colors)]
                curve = comp.normalized_intensity[model][tag]
                x = np.linspace(0, 1, len(curve))
                ax.plot(x, curve, "o-", color=c, lw=2, ms=3, label=model)

        ax.set_xlabel("Normalized Depth", fontsize=12)
        ax.set_ylabel("Processing Intensity", fontsize=12)
        ax.legend(fontsize=10)
        ax.set_title(f"Processing Intensity [{tag}] Across Scales", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig