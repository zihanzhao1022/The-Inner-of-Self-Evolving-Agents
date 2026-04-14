"""
Report Comparator — Compare two BehaviorReports from same-architecture models
with different training (e.g., base vs instruct, base vs DPO, v1 vs v2).

Answers: "What did alignment training change internally?"
  - Did the discriminative layer shift?
  - Which layers gained/lost behavior-separating ability?
  - Did processing intensity profiles change shape or just amplitude?
  - Did the Logit Lens prediction switch happen earlier or later?

Usage:
    from llm_lens.compare import ReportComparator

    comp = ReportComparator(report_base, report_chat)
    result = comp.run()
    result.print_summary()

    viz = LensVisualizer()
    viz_comp = CompareVisualizer()
    viz_comp.plot_all(result, save_dir="results/comparison/")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional

from .behavior import BehaviorReport, BehaviorSignature


# ─────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────

@dataclass
class ProbeShift:
    """How probe accuracy changed between two models."""
    num_layers: int
    accuracy_a: np.ndarray          # (num_layers,)
    accuracy_b: np.ndarray          # (num_layers,)
    diff: np.ndarray                # (num_layers,) = B - A, positive = B better
    best_layer_a: int
    best_layer_b: int
    best_layer_shifted: bool
    shift_distance: int             # abs(best_b - best_a)
    mean_accuracy_gain: float       # mean(diff), positive = B generally better
    max_gain_layer: int             # layer where B improved most over A
    max_loss_layer: int             # layer where B dropped most vs A


@dataclass
class IntensityShift:
    """How processing intensity changed for one behavior category."""
    tag: str
    intensity_a: np.ndarray         # (num_layers-1,)
    intensity_b: np.ndarray
    diff: np.ndarray                # B - A
    peak_layer_a: float             # most active layer in A
    peak_layer_b: float             # most active layer in B
    peak_shifted: bool
    shape_correlation: float        # Pearson r between A and B profiles
    amplitude_ratio: float          # mean(B) / mean(A), >1 = B processes more


@dataclass
class BifurcationShift:
    """How pairwise bifurcation changed."""
    tag_pair: tuple[str, str]
    cosine_a: Optional[np.ndarray]  # (num_layers,) or None if pair absent
    cosine_b: Optional[np.ndarray]
    bif_layer_a: Optional[int]
    bif_layer_b: Optional[int]
    bif_shifted: bool
    bif_direction: str              # "earlier", "later", "unchanged", "appeared", "disappeared"


@dataclass
class ComparisonResult:
    """Full comparison between two same-architecture models."""
    model_a: str
    model_b: str
    num_layers: int
    common_tags: list[str]

    probe_shift: ProbeShift
    intensity_shifts: dict[str, IntensityShift]     # tag -> shift
    bifurcation_shifts: dict[str, BifurcationShift] # "tag_a vs tag_b" -> shift

    def print_summary(self):
        print(f"\n{'=' * 64}")
        print(f"Comparison: {self.model_a}  vs  {self.model_b}")
        print(f"{'=' * 64}")
        print(f"Layers: {self.num_layers}")
        print(f"Common behaviors: {self.common_tags}")

        # Probe shift
        ps = self.probe_shift
        print(f"\n── Probe Discriminability ──")
        print(f"  Best layer:  A={ps.best_layer_a}  →  B={ps.best_layer_b}  "
              f"({'shifted by ' + str(ps.shift_distance) if ps.best_layer_shifted else 'same'})")
        print(f"  Mean accuracy gain (B-A): {ps.mean_accuracy_gain:+.4f}")
        print(f"  Layer with max gain:  {ps.max_gain_layer} ({ps.diff[ps.max_gain_layer]:+.4f})")
        print(f"  Layer with max loss:  {ps.max_loss_layer} ({ps.diff[ps.max_loss_layer]:+.4f})")

        # Intensity shifts
        print(f"\n── Processing Intensity Shifts ──")
        for tag, ish in self.intensity_shifts.items():
            peak_a_d = ish.peak_layer_a / self.num_layers
            peak_b_d = ish.peak_layer_b / self.num_layers
            print(f"  [{tag}]:")
            print(f"    Peak layer:  A={ish.peak_layer_a:.0f} ({peak_a_d:.2f})  →  "
                  f"B={ish.peak_layer_b:.0f} ({peak_b_d:.2f})  "
                  f"({'shifted' if ish.peak_shifted else 'same'})")
            print(f"    Shape correlation: {ish.shape_correlation:.3f}  "
                  f"({'similar shape' if ish.shape_correlation > 0.8 else 'different shape'})")
            print(f"    Amplitude ratio (B/A): {ish.amplitude_ratio:.2f}  "
                  f"({'B processes more' if ish.amplitude_ratio > 1.05 else 'B processes less' if ish.amplitude_ratio < 0.95 else 'similar'})")

        # Bifurcation shifts
        if self.bifurcation_shifts:
            print(f"\n── Bifurcation Shifts ──")
            for key, bsh in self.bifurcation_shifts.items():
                la = bsh.bif_layer_a if bsh.bif_layer_a is not None else "none"
                lb = bsh.bif_layer_b if bsh.bif_layer_b is not None else "none"
                print(f"  {bsh.tag_pair[0]} vs {bsh.tag_pair[1]}:  "
                      f"A=L{la}  →  B=L{lb}  ({bsh.bif_direction})")

        # Interpretation
        print(f"\n── Interpretation ──")
        if ps.mean_accuracy_gain > 0.02:
            print(f"  Model B is generally better at separating behaviors internally.")
        elif ps.mean_accuracy_gain < -0.02:
            print(f"  Model A is generally better at separating behaviors internally.")
        else:
            print(f"  Both models have similar overall discriminability.")

        shape_corrs = [ish.shape_correlation for ish in self.intensity_shifts.values()]
        mean_corr = np.mean(shape_corrs) if shape_corrs else 0
        if mean_corr > 0.85:
            print(f"  Training mostly changed WHERE decisions happen, not HOW "
                  f"(intensity profiles kept their shape, r={mean_corr:.2f}).")
        else:
            print(f"  Training fundamentally changed the processing pattern "
                  f"(intensity profiles differ, r={mean_corr:.2f}).")


# ─────────────────────────────────────────
# Comparator
# ─────────────────────────────────────────

class ReportComparator:
    """
    Compare two BehaviorReports from same-architecture models.

    Args:
        report_a: Baseline model (e.g., base)
        report_b: Altered model (e.g., instruct)
    """

    def __init__(self, report_a: BehaviorReport, report_b: BehaviorReport):
        assert report_a.num_layers == report_b.num_layers, (
            f"Layer count mismatch: {report_a.num_layers} vs {report_b.num_layers}. "
            f"Use CrossScaleAnalyzer for different architectures.")
        self.a = report_a
        self.b = report_b
        self.num_layers = report_a.num_layers

    def run(self) -> ComparisonResult:
        """Run full comparison."""
        common_tags = sorted(set(self.a.tags) & set(self.b.tags))
        assert len(common_tags) >= 1, "No common behavior tags between reports."

        probe_shift = self._compare_probes()
        intensity_shifts = {}
        for tag in common_tags:
            intensity_shifts[tag] = self._compare_intensity(tag)

        bif_shifts = self._compare_bifurcation()

        return ComparisonResult(
            model_a=self.a.model_name,
            model_b=self.b.model_name,
            num_layers=self.num_layers,
            common_tags=common_tags,
            probe_shift=probe_shift,
            intensity_shifts=intensity_shifts,
            bifurcation_shifts=bif_shifts,
        )

    def _compare_probes(self) -> ProbeShift:
        acc_a = self.a.layer_probe.probe_accuracy
        acc_b = self.b.layer_probe.probe_accuracy
        diff = acc_b - acc_a
        best_a = self.a.layer_probe.best_layer
        best_b = self.b.layer_probe.best_layer

        return ProbeShift(
            num_layers=self.num_layers,
            accuracy_a=acc_a,
            accuracy_b=acc_b,
            diff=diff,
            best_layer_a=best_a,
            best_layer_b=best_b,
            best_layer_shifted=best_a != best_b,
            shift_distance=abs(best_b - best_a),
            mean_accuracy_gain=diff.mean(),
            max_gain_layer=int(diff.argmax()),
            max_loss_layer=int(diff.argmin()),
        )

    def _compare_intensity(self, tag: str) -> IntensityShift:
        sig_a = self.a.signatures[tag]
        sig_b = self.b.signatures[tag]
        ia = sig_a.mean_intensity
        ib = sig_b.mean_intensity
        diff = ib - ia

        # Shape correlation
        if ia.std() > 1e-10 and ib.std() > 1e-10:
            corr = np.corrcoef(ia, ib)[0, 1]
        else:
            corr = 0.0

        # Amplitude ratio
        mean_a = ia.mean()
        mean_b = ib.mean()
        ratio = mean_b / (mean_a + 1e-10)

        peak_a = sig_a.mean_most_active_layer
        peak_b = sig_b.mean_most_active_layer

        return IntensityShift(
            tag=tag,
            intensity_a=ia,
            intensity_b=ib,
            diff=diff,
            peak_layer_a=peak_a,
            peak_layer_b=peak_b,
            peak_shifted=abs(peak_a - peak_b) > 1.0,
            shape_correlation=corr,
            amplitude_ratio=ratio,
        )

    def _compare_bifurcation(self) -> dict[str, BifurcationShift]:
        shifts = {}
        all_pairs = set(self.a.pairwise_bifurcation.keys()) | set(self.b.pairwise_bifurcation.keys())

        for pair in all_pairs:
            bif_a = self.a.pairwise_bifurcation.get(pair, {})
            bif_b = self.b.pairwise_bifurcation.get(pair, {})

            la = bif_a.get("bifurcation_layer")
            lb = bif_b.get("bifurcation_layer")
            cos_a = bif_a.get("cosine_per_layer")
            cos_b = bif_b.get("cosine_per_layer")

            # Determine direction
            if la is None and lb is None:
                direction = "unchanged"
            elif la is None and lb is not None:
                direction = "appeared"
            elif la is not None and lb is None:
                direction = "disappeared"
            elif lb < la:
                direction = "earlier"
            elif lb > la:
                direction = "later"
            else:
                direction = "unchanged"

            key = f"{pair[0]} vs {pair[1]}"
            shifts[key] = BifurcationShift(
                tag_pair=pair,
                cosine_a=cos_a,
                cosine_b=cos_b,
                bif_layer_a=la,
                bif_layer_b=lb,
                bif_shifted=la != lb,
                bif_direction=direction,
            )
        return shifts


# ─────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────

TAG_COLORS = {
    "harmful": "#E53935", "safe": "#43A047", "biased": "#FB8C00",
    "hallucination": "#8E24AA", "refusal": "#1E88E5", "sycophancy": "#FF7043",
    "factual": "#7CB342", "reasoning": "#5C6BC0", "creative": "#00ACC1",
}

COLOR_A = "#2196F3"
COLOR_B = "#F44336"


class CompareVisualizer:
    """Visualization suite for report comparison."""

    def __init__(self, figsize=(12, 6)):
        self.figsize = figsize
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except OSError:
            pass

    def plot_all(self, result: ComparisonResult, save_dir: str = "."):
        """Generate all comparison plots."""
        import os
        os.makedirs(save_dir, exist_ok=True)
        prefix = save_dir.rstrip("/")

        self.plot_probe_comparison(result, f"{prefix}/probe_comparison.png")
        self.plot_probe_diff(result, f"{prefix}/probe_diff.png")
        self.plot_intensity_comparison(result, f"{prefix}/intensity_comparison.png")
        self.plot_bifurcation_comparison(result, f"{prefix}/bifurcation_comparison.png")
        self.plot_summary_dashboard(result, f"{prefix}/dashboard.png")
        print(f"Saved 5 comparison plots to {save_dir}/")

    def plot_probe_comparison(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Overlaid probe accuracy curves for both models."""
        ps = result.probe_shift
        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(ps.num_layers)

        ax.plot(x, ps.accuracy_a, "o-", color=COLOR_A, lw=2, ms=4,
                label=f"A: {result.model_a}")
        ax.plot(x, ps.accuracy_b, "s-", color=COLOR_B, lw=2, ms=4,
                label=f"B: {result.model_b}")

        # Mark best layers
        ax.axvline(ps.best_layer_a, color=COLOR_A, ls="--", alpha=0.4)
        ax.axvline(ps.best_layer_b, color=COLOR_B, ls="--", alpha=0.4)
        ax.axhline(0.5, color="gray", ls=":", alpha=0.4)

        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("Probe Accuracy", fontsize=13)
        ax.set_ylim(0.35, 1.05)
        ax.legend(fontsize=10)
        ax.set_title("Probe Accuracy: A vs B", fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_probe_diff(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Bar chart of probe accuracy difference (B - A) per layer."""
        ps = result.probe_shift
        fig, ax = plt.subplots(figsize=self.figsize)
        x = np.arange(ps.num_layers)

        colors = [COLOR_B if d > 0 else COLOR_A for d in ps.diff]
        ax.bar(x, ps.diff, color=colors, alpha=0.7)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel("Layer", fontsize=13)
        ax.set_ylabel("Accuracy Δ (B − A)", fontsize=13)
        ax.set_title("Probe Accuracy Change per Layer", fontsize=14, fontweight="bold")

        # Annotate
        ax.annotate(f"B best gains: L{ps.max_gain_layer}",
                    xy=(ps.max_gain_layer, ps.diff[ps.max_gain_layer]),
                    xytext=(ps.max_gain_layer + 2, ps.diff[ps.max_gain_layer] + 0.02),
                    arrowprops=dict(arrowstyle="->"), fontsize=9)

        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_intensity_comparison(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Side-by-side intensity profiles for each behavior."""
        tags = result.common_tags
        n = len(tags)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
        if n == 1: axes = [axes]

        for ax, tag in zip(axes, tags):
            ish = result.intensity_shifts[tag]
            x = np.arange(len(ish.intensity_a))
            ax.plot(x, ish.intensity_a, "o-", color=COLOR_A, lw=2, ms=3, label="A")
            ax.plot(x, ish.intensity_b, "s-", color=COLOR_B, lw=2, ms=3, label="B")
            ax.fill_between(x, ish.intensity_a, ish.intensity_b,
                            alpha=0.1, color="gray")
            ax.set_title(f"[{tag}]\nr={ish.shape_correlation:.2f}, "
                         f"amp={ish.amplitude_ratio:.2f}x", fontsize=11)
            ax.set_xlabel("Layer transition")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Processing Intensity")
        fig.suptitle(f"Intensity: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_bifurcation_comparison(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Overlaid cosine similarity curves for each behavior pair."""
        bifs = result.bifurcation_shifts
        if not bifs:
            return plt.figure()

        n = len(bifs)
        fig, axes = plt.subplots(1, min(n, 4), figsize=(5 * min(n, 4), 5), sharey=True)
        if n == 1: axes = [axes]

        for ax, (key, bsh) in zip(axes, list(bifs.items())[:4]):
            if bsh.cosine_a is not None:
                x_a = np.arange(len(bsh.cosine_a))
                ax.plot(x_a, bsh.cosine_a, "o-", color=COLOR_A, lw=2, ms=3, label="A")
            if bsh.cosine_b is not None:
                x_b = np.arange(len(bsh.cosine_b))
                ax.plot(x_b, bsh.cosine_b, "s-", color=COLOR_B, lw=2, ms=3, label="B")

            ax.axhline(0.9, color="gray", ls=":", alpha=0.4)

            if bsh.bif_layer_a is not None:
                ax.axvline(bsh.bif_layer_a, color=COLOR_A, ls="--", alpha=0.4)
            if bsh.bif_layer_b is not None:
                ax.axvline(bsh.bif_layer_b, color=COLOR_B, ls="--", alpha=0.4)

            ax.set_title(f"{key}\n({bsh.bif_direction})", fontsize=10)
            ax.set_xlabel("Layer")
            ax.legend(fontsize=9)
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)

        axes[0].set_ylabel("Cosine Similarity")
        fig.suptitle(f"Bifurcation: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_summary_dashboard(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Single-figure dashboard summarizing all shifts."""
        fig = plt.figure(figsize=(14, 10))

        # Panel 1: Probe accuracy overlay
        ax1 = fig.add_subplot(2, 2, 1)
        ps = result.probe_shift
        x = np.arange(ps.num_layers)
        ax1.plot(x, ps.accuracy_a, "-", color=COLOR_A, lw=2, label="A")
        ax1.plot(x, ps.accuracy_b, "-", color=COLOR_B, lw=2, label="B")
        ax1.axhline(0.5, color="gray", ls=":", alpha=0.3)
        ax1.set_title("Probe accuracy", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Accuracy")
        ax1.legend(fontsize=9)
        ax1.set_ylim(0.35, 1.05)

        # Panel 2: Probe diff
        ax2 = fig.add_subplot(2, 2, 2)
        colors = [COLOR_B if d > 0 else COLOR_A for d in ps.diff]
        ax2.bar(x, ps.diff, color=colors, alpha=0.7)
        ax2.axhline(0, color="gray", lw=0.8)
        ax2.set_title("Probe Δ (B − A)", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Accuracy Δ")

        # Panel 3: Intensity shape correlation per tag
        ax3 = fig.add_subplot(2, 2, 3)
        tags = result.common_tags
        corrs = [result.intensity_shifts[t].shape_correlation for t in tags]
        tag_colors = [TAG_COLORS.get(t, "#888") for t in tags]
        bars = ax3.barh(range(len(tags)), corrs, color=tag_colors, alpha=0.7)
        ax3.set_yticks(range(len(tags)))
        ax3.set_yticklabels(tags)
        ax3.axvline(0.85, color="gray", ls="--", alpha=0.5, label="shape preserved")
        ax3.set_xlim(0, 1.05)
        ax3.set_title("Intensity shape correlation", fontsize=11, fontweight="bold")
        ax3.set_xlabel("Pearson r")
        ax3.legend(fontsize=8)

        # Panel 4: Peak layer shift per tag
        ax4 = fig.add_subplot(2, 2, 4)
        peak_a = [result.intensity_shifts[t].peak_layer_a for t in tags]
        peak_b = [result.intensity_shifts[t].peak_layer_b for t in tags]
        y = np.arange(len(tags))
        ax4.barh(y - 0.15, peak_a, height=0.3, color=COLOR_A, alpha=0.7, label="A")
        ax4.barh(y + 0.15, peak_b, height=0.3, color=COLOR_B, alpha=0.7, label="B")
        ax4.set_yticks(y)
        ax4.set_yticklabels(tags)
        ax4.set_title("Peak processing layer", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Layer")
        ax4.legend(fontsize=9)

        fig.suptitle(f"{result.model_a}  vs  {result.model_b}",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
