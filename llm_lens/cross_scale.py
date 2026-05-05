"""
Cross-Scale Analyzer — Compare behavioral dynamics across model sizes
to test the Critical Window Hypothesis.

Core hypothesis: The layer depth ratio where behavior divergence peaks is
stable across model scales. If true, we can monitor only this "critical
window" regardless of model size.

Usage:
    analyzer = CrossScaleAnalyzer()

    # Add reports from different model sizes
    analyzer.add_report(report_3b)   # from Qwen2.5-3B
    analyzer.add_report(report_7b)   # from Qwen2.5-7B
    analyzer.add_report(report_14b)  # from Qwen2.5-14B

    # Test critical window hypothesis
    result = analyzer.test_critical_window()
    result.print_summary()
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy import stats as sp_stats

from .behavior import BehaviorReport


@dataclass
class ModelScalePoint:
    """One data point: a model at a specific scale."""
    model_name: str
    num_layers: int
    param_count: Optional[float]  # in billions
    report: BehaviorReport

    # Derived: key ratios
    best_probe_depth: float       # best_layer / num_layers
    per_tag_peak_depth: dict[str, float]  # tag -> most_active_layer / num_layers


@dataclass
class CriticalWindow:
    """Result of critical window hypothesis test."""
    # Per-model Qwen_Qwen2.5-3B
    models: list[str]
    num_layers: list[int]
    param_counts: list[Optional[float]]
    best_probe_depths: list[float]

    # Stability metrics
    probe_depth_mean: float
    probe_depth_std: float
    probe_depth_cv: float         # coefficient of variation (std/mean)

    # Per-tag peak depths
    per_tag_depths: dict[str, list[float]]  # tag -> [depth_model1, depth_model2, ...]
    per_tag_stability: dict[str, float]     # tag -> cv

    # Critical window bounds (mean ± std)
    window_center: float
    window_start: float
    window_end: float
    window_width: float

    # Statistical test (if >=3 models)
    is_stable: bool               # cv < 0.15 (somewhat arbitrary threshold)
    kendall_tau: Optional[float] = None  # correlation between scale and depth
    kendall_p: Optional[float] = None

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"Critical Window Hypothesis Test")
        print(f"{'=' * 60}")

        print(f"\nModels analyzed: {len(self.models)}")
        for i, m in enumerate(self.models):
            p = f"{self.param_counts[i]:.2f}B" if self.param_counts[i] else "?"
            print(f"  {m}: {self.num_layers[i]} layers, {p} params, "
                  f"best probe depth = {self.best_probe_depths[i]:.3f}")

        print(f"\nBest discriminative depth ratio:")
        print(f"  Mean:  {self.probe_depth_mean:.3f}")
        print(f"  Std:   {self.probe_depth_std:.3f}")
        print(f"  CV:    {self.probe_depth_cv:.3f}")
        print(f"  Stable (CV < 0.15): {'YES' if self.is_stable else 'NO'}")

        print(f"\nCritical window: [{self.window_start:.2f}, {self.window_end:.2f}] "
              f"(width={self.window_width:.2f})")
        print(f"  Center: {self.window_center:.2f}")

        if self.kendall_tau is not None:
            print(f"\nScale-depth correlation (Kendall tau): {self.kendall_tau:.3f} "
                  f"(p={self.kendall_p:.3f})")
            if self.kendall_p > 0.05:
                print(f"  Not significant — depth ratio does NOT scale with model size.")
            else:
                print(f"  Significant — depth ratio changes with model size.")

        if self.per_tag_stability:
            print(f"\nPer-behavior peak depth stability:")
            for tag, cv in sorted(self.per_tag_stability.items()):
                depths = self.per_tag_depths[tag]
                print(f"  [{tag}]: mean={np.mean(depths):.3f}, "
                      f"std={np.std(depths):.3f}, cv={cv:.3f}")


@dataclass
class ScaleComparisonResult:
    """Detailed comparison of dynamics across scales."""
    tags: list[str]
    models: list[str]

    # (n_models, n_layers_normalized) — normalized to same depth resolution
    normalized_probe_accuracy: dict[str, np.ndarray]  # model -> (n_bins,)
    normalized_intensity: dict[str, dict[str, np.ndarray]]  # model -> tag -> (n_bins,)

    # Correlation matrix of probe accuracy curves between models
    probe_correlation_matrix: np.ndarray  # (n_models, n_models)


class CrossScaleAnalyzer:
    """
    Compare behavioral dynamics reports from models of different sizes.

    Tests whether the "critical window" (the depth ratio where behavior
    discrimination peaks) is stable across scales.
    """

    def __init__(self):
        self.points: list[ModelScalePoint] = []

    def add_report(self, report: BehaviorReport, param_count: Optional[float] = None):
        """Add a BehaviorReport from one model."""
        # Compute per-tag peak depth ratios
        per_tag = {}
        for tag, sig in report.signatures.items():
            per_tag[tag] = sig.mean_most_active_layer / report.num_layers

        point = ModelScalePoint(
            model_name=report.model_name,
            num_layers=report.num_layers,
            param_count=param_count,
            report=report,
            best_probe_depth=report.layer_probe.best_depth_ratio,
            per_tag_peak_depth=per_tag,
        )
        self.points.append(point)

    def test_critical_window(self) -> CriticalWindow:
        """
        Test the Critical Window Hypothesis.

        Returns a CriticalWindow with stability metrics.
        """
        assert len(self.points) >= 2, "Need >= 2 models for cross-scale comparison"

        models = [p.model_name for p in self.points]
        layers = [p.num_layers for p in self.points]
        params = [p.param_count for p in self.points]
        probe_depths = [p.best_probe_depth for p in self.points]

        # Overall stability
        mean_d = np.mean(probe_depths)
        std_d = np.std(probe_depths)
        cv = std_d / (mean_d + 1e-10)

        # Per-tag stability
        all_tags = set()
        for p in self.points:
            all_tags.update(p.per_tag_peak_depth.keys())

        per_tag_depths = {}
        per_tag_cv = {}
        for tag in all_tags:
            depths = [p.per_tag_peak_depth.get(tag) for p in self.points
                      if tag in p.per_tag_peak_depth]
            if len(depths) >= 2:
                per_tag_depths[tag] = depths
                tag_mean = np.mean(depths)
                tag_std = np.std(depths)
                per_tag_cv[tag] = tag_std / (tag_mean + 1e-10)

        # Critical window bounds
        window_center = mean_d
        window_start = max(0, mean_d - std_d)
        window_end = min(1, mean_d + std_d)

        # Kendall tau (if >=3 models with param counts)
        tau, tau_p = None, None
        valid_params = [(p, d) for p, d in zip(params, probe_depths) if p is not None]
        if len(valid_params) >= 3:
            ps, ds = zip(*sorted(valid_params))
            tau, tau_p = sp_stats.kendalltau(ps, ds)

        return CriticalWindow(
            models=models,
            num_layers=layers,
            param_counts=params,
            best_probe_depths=probe_depths,
            probe_depth_mean=mean_d,
            probe_depth_std=std_d,
            probe_depth_cv=cv,
            per_tag_depths=per_tag_depths,
            per_tag_stability=per_tag_cv,
            window_center=window_center,
            window_start=window_start,
            window_end=window_end,
            window_width=window_end - window_start,
            is_stable=cv < 0.15,
            kendall_tau=tau,
            kendall_p=tau_p,
        )

    def compare_dynamics(self, n_bins: int = 32) -> ScaleComparisonResult:
        """
        Normalize all models to the same depth resolution and compare
        probe accuracy curves and intensity profiles.
        """
        assert len(self.points) >= 2

        # Collect common tags
        common_tags = set(self.points[0].report.tags)
        for p in self.points[1:]:
            common_tags &= set(p.report.tags)
        common_tags = sorted(common_tags)

        # Normalize probe accuracy to n_bins
        norm_probes = {}
        for p in self.points:
            original = p.report.layer_probe.probe_accuracy
            norm_probes[p.model_name] = self._normalize_to_bins(original, n_bins)

        # Normalize intensity per tag
        norm_intensity = {}
        for p in self.points:
            norm_intensity[p.model_name] = {}
            for tag in common_tags:
                sig = p.report.signatures.get(tag)
                if sig is not None:
                    norm_intensity[p.model_name][tag] = self._normalize_to_bins(
                        sig.mean_intensity, n_bins)

        # Correlation matrix of probe accuracy curves
        model_names = [p.model_name for p in self.points]
        n = len(model_names)
        corr = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                a = norm_probes[model_names[i]]
                b = norm_probes[model_names[j]]
                corr[i, j] = np.corrcoef(a, b)[0, 1]

        return ScaleComparisonResult(
            tags=common_tags,
            models=model_names,
            normalized_probe_accuracy=norm_probes,
            normalized_intensity=norm_intensity,
            probe_correlation_matrix=corr,
        )

    def get_critical_layers(self, model_name: str, window: CriticalWindow) -> tuple[int, int]:
        """Convert critical window depth ratios to actual layer indices for a specific model."""
        point = next(p for p in self.points if p.model_name == model_name)
        start = int(window.window_start * point.num_layers)
        end = min(int(window.window_end * point.num_layers) + 1, point.num_layers)
        return start, end

    @staticmethod
    def _normalize_to_bins(arr: np.ndarray, n_bins: int) -> np.ndarray:
        """Resample an array of arbitrary length to n_bins via interpolation."""
        x_old = np.linspace(0, 1, len(arr))
        x_new = np.linspace(0, 1, n_bins)
        return np.interp(x_new, x_old, arr)
