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
from .attention import HeadAnalysisResult


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
    cosine_lower_a: Optional[np.ndarray]  # 2.5% percentile (bootstrap CI)
    cosine_upper_a: Optional[np.ndarray]  # 97.5% percentile
    cosine_lower_b: Optional[np.ndarray]
    cosine_upper_b: Optional[np.ndarray]
    bif_layer_a: Optional[int]
    bif_layer_b: Optional[int]
    bif_shifted: bool
    bif_direction: str              # "earlier", "later", "unchanged", "appeared", "disappeared"


@dataclass
class DirectionDynamicsShift:
    """How probe-direction stability and norm changed between two models."""
    direction_cosine_a: Optional[np.ndarray]    # (num_layers - 1,) — adjacent-layer probe direction cosines
    direction_cosine_b: Optional[np.ndarray]
    direction_cosine_diff: Optional[np.ndarray]
    direction_norm_a: Optional[np.ndarray]      # (num_layers,) — per-layer probe weight L2 norm
    direction_norm_b: Optional[np.ndarray]
    direction_norm_diff: Optional[np.ndarray]


@dataclass
class BICShift:
    """How behavior-information-concentration metrics changed."""
    effective_rank_a: Optional[np.ndarray]      # (num_layers,)
    effective_rank_b: Optional[np.ndarray]
    effective_rank_diff: Optional[np.ndarray]
    explained_ratio_a: Optional[np.ndarray]
    explained_ratio_b: Optional[np.ndarray]
    explained_ratio_diff: Optional[np.ndarray]
    inter_class_ratio_a: Optional[np.ndarray]
    inter_class_ratio_b: Optional[np.ndarray]
    inter_class_ratio_diff: Optional[np.ndarray]


@dataclass
class NormShift:
    """How per-tag activation norm profile changed."""
    tag: str
    norm_a: np.ndarray                           # (num_layers,)
    norm_b: np.ndarray
    norm_diff: np.ndarray
    shape_correlation: float                     # Pearson r between A and B profiles
    amplitude_ratio: float                       # mean(B) / mean(A)


@dataclass
class HeadShift:
    """Per-(layer, head) probing comparison.

    All matrices are (num_layers, num_heads). Top-K lists report individual heads
    with the largest accuracy gain or loss between models.
    """
    num_layers: int
    num_heads: int
    accuracy_a: np.ndarray
    accuracy_b: np.ndarray
    accuracy_diff: np.ndarray
    separation_a: np.ndarray
    separation_b: np.ndarray
    separation_diff: np.ndarray
    inter_class_a: np.ndarray
    inter_class_b: np.ndarray
    inter_class_diff: np.ndarray
    top_gain_heads: list                         # [(layer, head, accuracy_diff), ...] descending
    top_loss_heads: list                         # [(layer, head, accuracy_diff), ...] ascending
    n_top: int


@dataclass
class CentroidShift:
    """Per-layer per-class alignment between A's and B's class centroids.

    For each layer L and class c, computes cos(A.centroid[L, c], B.centroid[L, c]).
    A value near 1 means the same class direction; lower means the training
    rotated the representation for that class at that depth.
    """
    tags_order: list                             # class names in column order
    num_layers: int
    centroid_cosine: np.ndarray                  # (num_layers, n_classes)
    centroid_norm_ratio: np.ndarray              # (num_layers, n_classes) = ||B|| / ||A||
    mean_cosine_per_layer: np.ndarray            # (num_layers,)
    min_layer: int                               # layer of largest divergence (lowest mean cosine)
    min_layer_cosine: float
    behavior_direction_cosine: Optional[np.ndarray]  # (num_layers,) — top SVD direction alignment


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

    # New: optional extended shifts (None if source data unavailable)
    direction_shift: Optional[DirectionDynamicsShift] = None
    bic_shift: Optional[BICShift] = None
    norm_shifts: Optional[dict] = None              # tag -> NormShift
    head_shift: Optional[HeadShift] = None
    centroid_shift: Optional[CentroidShift] = None

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

        # Centroid alignment (the headline new comparison)
        if self.centroid_shift is not None:
            cs = self.centroid_shift
            mean_cos = float(cs.mean_cosine_per_layer.mean())
            print(f"\n── Class Centroid Alignment ──")
            print(f"  Mean cos(A.centroid, B.centroid) over all (layer, class): {mean_cos:.3f}")
            print(f"  Most divergent layer: L{cs.min_layer} (mean cos = {cs.min_layer_cosine:.3f})")
            # Per-class summary at the most divergent layer
            row = cs.centroid_cosine[cs.min_layer]
            order = np.argsort(row)
            worst_class = cs.tags_order[order[0]]
            best_class = cs.tags_order[order[-1]]
            print(f"    most rotated class @ L{cs.min_layer}:  '{worst_class}' (cos={row[order[0]]:.3f})")
            print(f"    most aligned class @ L{cs.min_layer}:  '{best_class}' (cos={row[order[-1]]:.3f})")

        # Head shift summary
        if self.head_shift is not None:
            hs = self.head_shift
            mean_acc_diff = float(hs.accuracy_diff.mean())
            print(f"\n── Per-Head Probe ──")
            print(f"  Mean head probe accuracy gain (B-A): {mean_acc_diff:+.4f}")
            print(f"  Top {min(3, len(hs.top_gain_heads))} heads where B improved:")
            for layer, head, d in hs.top_gain_heads[:3]:
                print(f"    L{layer:>2} H{head:>2}: {d:+.4f}")
            print(f"  Top {min(3, len(hs.top_loss_heads))} heads where B dropped:")
            for layer, head, d in hs.top_loss_heads[:3]:
                print(f"    L{layer:>2} H{head:>2}: {d:+.4f}")

        # BIC summary
        if self.bic_shift is not None and self.bic_shift.effective_rank_diff is not None:
            bs = self.bic_shift
            print(f"\n── Behavior Information Concentration (BIC) ──")
            print(f"  Effective rank   mean Δ: {float(bs.effective_rank_diff.mean()):+.3f}")
            print(f"  Explained ratio  mean Δ: {float(bs.explained_ratio_diff.mean()):+.4f}")
            print(f"  Inter-class ratio mean Δ: {float(bs.inter_class_ratio_diff.mean()):+.4f}")

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

        if self.centroid_shift is not None:
            mean_cos = float(self.centroid_shift.mean_cosine_per_layer.mean())
            if mean_cos > 0.95:
                print(f"  Class representations are almost identical between models "
                      f"(mean cos = {mean_cos:.3f}) — training did not rotate behavior directions.")
            elif mean_cos < 0.7:
                print(f"  Class representations are substantially rotated "
                      f"(mean cos = {mean_cos:.3f}) — training reorganized internal directions.")
            else:
                print(f"  Class representations are moderately aligned "
                      f"(mean cos = {mean_cos:.3f}).")


# ─────────────────────────────────────────
# Comparator
# ─────────────────────────────────────────

class ReportComparator:
    """
    Compare two BehaviorReports from same-architecture models.

    Args:
        report_a: Baseline model (e.g., base)
        report_b: Altered model (e.g., instruct)
        centroids_a: optional dict from report_io.load_class_centroids() — enables
                     CentroidShift analysis.
        centroids_b: optional dict from report_io.load_class_centroids().
    """

    def __init__(self, report_a: BehaviorReport, report_b: BehaviorReport,
                 centroids_a: Optional[dict] = None,
                 centroids_b: Optional[dict] = None):
        assert report_a.num_layers == report_b.num_layers, (
            f"Layer count mismatch: {report_a.num_layers} vs {report_b.num_layers}. "
            f"Use CrossScaleAnalyzer for different architectures.")
        self.a = report_a
        self.b = report_b
        self.num_layers = report_a.num_layers
        self.centroids_a = centroids_a
        self.centroids_b = centroids_b

    def run(self) -> ComparisonResult:
        """Run full comparison."""
        common_tags = sorted(set(self.a.tags) & set(self.b.tags))
        assert len(common_tags) >= 1, "No common behavior tags between reports."

        probe_shift = self._compare_probes()
        intensity_shifts = {}
        for tag in common_tags:
            intensity_shifts[tag] = self._compare_intensity(tag)
        bif_shifts = self._compare_bifurcation()

        # New extended shifts (return None if source data missing)
        direction_shift = self._compare_direction_dynamics()
        bic_shift = self._compare_bic()
        norm_shifts = self._compare_norms(common_tags)
        head_shift = self._compare_heads()
        centroid_shift = self._compare_centroids(common_tags)

        return ComparisonResult(
            model_a=self.a.model_name,
            model_b=self.b.model_name,
            num_layers=self.num_layers,
            common_tags=common_tags,
            probe_shift=probe_shift,
            intensity_shifts=intensity_shifts,
            bifurcation_shifts=bif_shifts,
            direction_shift=direction_shift,
            bic_shift=bic_shift,
            norm_shifts=norm_shifts,
            head_shift=head_shift,
            centroid_shift=centroid_shift,
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
        all_pairs = sorted(set(self.a.pairwise_bifurcation.keys()) | set(self.b.pairwise_bifurcation.keys()))

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
                cosine_lower_a=bif_a.get("cosine_lower"),
                cosine_upper_a=bif_a.get("cosine_upper"),
                cosine_lower_b=bif_b.get("cosine_lower"),
                cosine_upper_b=bif_b.get("cosine_upper"),
                bif_layer_a=la,
                bif_layer_b=lb,
                bif_shifted=la != lb,
                bif_direction=direction,
            )
        return shifts

    # ── Extended comparisons ──────────────────────────────────────────────

    def _compare_direction_dynamics(self) -> Optional[DirectionDynamicsShift]:
        lp_a, lp_b = self.a.layer_probe, self.b.layer_probe
        dc_a = lp_a.direction_cosine
        dc_b = lp_b.direction_cosine
        dn_a = lp_a.direction_norm
        dn_b = lp_b.direction_norm

        if dc_a is None and dn_a is None:
            return None

        return DirectionDynamicsShift(
            direction_cosine_a=dc_a,
            direction_cosine_b=dc_b,
            direction_cosine_diff=(dc_b - dc_a) if (dc_a is not None and dc_b is not None) else None,
            direction_norm_a=dn_a,
            direction_norm_b=dn_b,
            direction_norm_diff=(dn_b - dn_a) if (dn_a is not None and dn_b is not None) else None,
        )

    def _compare_bic(self) -> Optional[BICShift]:
        lp_a, lp_b = self.a.layer_probe, self.b.layer_probe
        if lp_a.bic_effective_rank is None or lp_b.bic_effective_rank is None:
            return None

        def _diff(x_a, x_b):
            if x_a is None or x_b is None:
                return None
            return x_b - x_a

        return BICShift(
            effective_rank_a=lp_a.bic_effective_rank,
            effective_rank_b=lp_b.bic_effective_rank,
            effective_rank_diff=_diff(lp_a.bic_effective_rank, lp_b.bic_effective_rank),
            explained_ratio_a=lp_a.bic_explained_ratio,
            explained_ratio_b=lp_b.bic_explained_ratio,
            explained_ratio_diff=_diff(lp_a.bic_explained_ratio, lp_b.bic_explained_ratio),
            inter_class_ratio_a=lp_a.bic_inter_class_ratio,
            inter_class_ratio_b=lp_b.bic_inter_class_ratio,
            inter_class_ratio_diff=_diff(lp_a.bic_inter_class_ratio, lp_b.bic_inter_class_ratio),
        )

    def _compare_norms(self, common_tags: list) -> Optional[dict]:
        out = {}
        for tag in common_tags:
            if tag not in self.a.signatures or tag not in self.b.signatures:
                continue
            na = self.a.signatures[tag].mean_norm
            nb = self.b.signatures[tag].mean_norm
            if na is None or nb is None or na.size == 0 or nb.size == 0:
                continue
            if na.sum() == 0 and nb.sum() == 0:
                continue  # legacy report — norm not captured
            diff = nb - na
            corr = float(np.corrcoef(na, nb)[0, 1]) if na.std() > 1e-10 and nb.std() > 1e-10 else 0.0
            ratio = float(nb.mean() / (na.mean() + 1e-10))
            out[tag] = NormShift(
                tag=tag,
                norm_a=na, norm_b=nb, norm_diff=diff,
                shape_correlation=corr, amplitude_ratio=ratio,
            )
        return out if out else None

    def _compare_heads(self) -> Optional[HeadShift]:
        ha_a = self.a.head_analysis
        ha_b = self.b.head_analysis
        if ha_a is None or ha_b is None:
            return None
        if ha_a.num_layers != ha_b.num_layers or ha_a.num_heads != ha_b.num_heads:
            print(f"  [compare] head shape mismatch "
                  f"({ha_a.num_layers}x{ha_a.num_heads} vs "
                  f"{ha_b.num_layers}x{ha_b.num_heads}); skipping head comparison.")
            return None

        acc_diff = ha_b.head_probe_accuracy - ha_a.head_probe_accuracy
        sep_diff = ha_b.head_separation - ha_a.head_separation
        ic_diff = ha_b.head_inter_class_ratio - ha_a.head_inter_class_ratio

        n_top = min(10, acc_diff.size)
        flat = acc_diff.flatten()
        order = np.argsort(flat)
        top_loss, top_gain = [], []
        for idx in order[:n_top]:
            l, h = np.unravel_index(int(idx), acc_diff.shape)
            top_loss.append((int(l), int(h), float(flat[idx])))
        for idx in order[-n_top:][::-1]:
            l, h = np.unravel_index(int(idx), acc_diff.shape)
            top_gain.append((int(l), int(h), float(flat[idx])))

        return HeadShift(
            num_layers=ha_a.num_layers,
            num_heads=ha_a.num_heads,
            accuracy_a=ha_a.head_probe_accuracy,
            accuracy_b=ha_b.head_probe_accuracy,
            accuracy_diff=acc_diff,
            separation_a=ha_a.head_separation,
            separation_b=ha_b.head_separation,
            separation_diff=sep_diff,
            inter_class_a=ha_a.head_inter_class_ratio,
            inter_class_b=ha_b.head_inter_class_ratio,
            inter_class_diff=ic_diff,
            top_gain_heads=top_gain,
            top_loss_heads=top_loss,
            n_top=n_top,
        )

    def _compare_centroids(self, common_tags: list) -> Optional[CentroidShift]:
        if self.centroids_a is None or self.centroids_b is None:
            return None
        ca = self.centroids_a.get("class_centroids")
        cb = self.centroids_b.get("class_centroids")
        if ca is None or cb is None:
            return None

        tags_a = list(self.centroids_a.get("tags_order") or [])
        tags_b = list(self.centroids_b.get("tags_order") or [])
        if not tags_a or not tags_b:
            print(f"  [compare] missing tags_order on centroid npz; skipping centroid comparison.")
            return None

        # Align by tag name; intersect with common_tags
        common = [t for t in tags_a if t in tags_b and t in common_tags]
        if len(common) < 1:
            print(f"  [compare] no common tags between centroid files; skipping.")
            return None

        idx_a = [tags_a.index(t) for t in common]
        idx_b = [tags_b.index(t) for t in common]

        ca_aligned = ca[:, idx_a, :]   # (L, C, D)
        cb_aligned = cb[:, idx_b, :]
        if ca_aligned.shape != cb_aligned.shape:
            print(f"  [compare] centroid shape mismatch after alignment; skipping.")
            return None

        L, C, _ = ca_aligned.shape
        norms_a = np.linalg.norm(ca_aligned, axis=-1) + 1e-10  # (L, C)
        norms_b = np.linalg.norm(cb_aligned, axis=-1) + 1e-10
        dot = (ca_aligned * cb_aligned).sum(axis=-1)            # (L, C)
        cosines = dot / (norms_a * norms_b)
        norm_ratio = norms_b / norms_a

        mean_per_layer = cosines.mean(axis=1)
        min_layer = int(np.argmin(mean_per_layer))
        min_layer_cos = float(mean_per_layer[min_layer])

        # behavior_direction (top SVD direction across class means)
        bd_a = self.centroids_a.get("behavior_direction")
        bd_b = self.centroids_b.get("behavior_direction")
        bd_cosine = None
        if bd_a is not None and bd_b is not None and bd_a.shape == bd_b.shape:
            bna = np.linalg.norm(bd_a, axis=-1) + 1e-10
            bnb = np.linalg.norm(bd_b, axis=-1) + 1e-10
            bd_dot = (bd_a * bd_b).sum(axis=-1)
            # SVD sign is arbitrary; compare absolute cosine
            bd_cosine = np.abs(bd_dot / (bna * bnb))

        return CentroidShift(
            tags_order=common,
            num_layers=L,
            centroid_cosine=cosines,
            centroid_norm_ratio=norm_ratio,
            mean_cosine_per_layer=mean_per_layer,
            min_layer=min_layer,
            min_layer_cosine=min_layer_cos,
            behavior_direction_cosine=bd_cosine,
        )


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

# Fixed display order for the common 4-tag setup (6 pairs).
FIXED_BIFURCATION_ORDER = [
    ("factual", "harmful"),
    ("factual", "reasoning"),
    ("factual", "safe"),
    ("harmful", "reasoning"),
    ("harmful", "safe"),
    ("reasoning", "safe"),
]


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

        n = 5
        self.plot_probe_comparison(result, f"{prefix}/probe_comparison.png")
        self.plot_probe_diff(result, f"{prefix}/probe_diff.png")
        self.plot_intensity_comparison(result, f"{prefix}/intensity_comparison.png")
        self.plot_bifurcation_comparison(result, f"{prefix}/bifurcation_comparison.png")
        self.plot_summary_dashboard(result, f"{prefix}/dashboard.png")

        # Extended plots — only emitted if the underlying data is present
        if result.direction_shift is not None:
            self.plot_direction_dynamics_comparison(
                result, f"{prefix}/direction_dynamics_comparison.png")
            n += 1
        if result.bic_shift is not None:
            self.plot_bic_comparison(result, f"{prefix}/bic_comparison.png")
            n += 1
        if result.norm_shifts:
            self.plot_norm_comparison(result, f"{prefix}/norm_comparison.png")
            n += 1
        if result.head_shift is not None:
            self.plot_head_diff(result, f"{prefix}/head_diff.png")
            n += 1
        if result.centroid_shift is not None:
            self.plot_centroid_alignment(result, f"{prefix}/centroid_alignment.png")
            n += 1

        print(f"Saved {n} comparison plots to {save_dir}/")

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
        """Overlaid cosine similarity curves with bootstrap CI bands."""
        bifs = result.bifurcation_shifts
        if not bifs:
            return plt.figure()

        ordered_items = []
        used = set()

        # 1) Preferred fixed order for the standard 6-pair view.
        for ta, tb in FIXED_BIFURCATION_ORDER:
            key = f"{ta} vs {tb}"
            if key in bifs:
                ordered_items.append((key, bifs[key]))
                used.add(key)

        # 2) Any extra pairs are appended in deterministic order.
        for key in sorted(bifs.keys()):
            if key not in used:
                ordered_items.append((key, bifs[key]))

        n = len(ordered_items)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.6 * rows), sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)

        for ax, (key, bsh) in zip(axes, ordered_items):
            if bsh.cosine_a is not None:
                x_a = np.arange(len(bsh.cosine_a))
                ax.plot(x_a, bsh.cosine_a, "o-", color=COLOR_A, lw=2, ms=3, label="A")
                if bsh.cosine_lower_a is not None and bsh.cosine_upper_a is not None:
                    ax.fill_between(x_a, bsh.cosine_lower_a, bsh.cosine_upper_a,
                                    color=COLOR_A, alpha=0.12)
            if bsh.cosine_b is not None:
                x_b = np.arange(len(bsh.cosine_b))
                ax.plot(x_b, bsh.cosine_b, "s-", color=COLOR_B, lw=2, ms=3, label="B")
                if bsh.cosine_lower_b is not None and bsh.cosine_upper_b is not None:
                    ax.fill_between(x_b, bsh.cosine_lower_b, bsh.cosine_upper_b,
                                    color=COLOR_B, alpha=0.12)

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

        # Hide unused subplot slots when n is not a multiple of cols.
        for ax in axes[n:]:
            ax.axis("off")

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

    # ── Extended comparison plots ──────────────────────────────────────────

    def plot_direction_dynamics_comparison(self, result: ComparisonResult,
                                            save_path=None) -> plt.Figure:
        """Two panels: probe direction stability (cosine) and weight norm, A vs B."""
        ds = result.direction_shift
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        if ds.direction_cosine_a is not None and ds.direction_cosine_b is not None:
            x = np.arange(len(ds.direction_cosine_a))
            ax.plot(x, ds.direction_cosine_a, "o-", color=COLOR_A, lw=2, ms=3, label="A")
            ax.plot(x, ds.direction_cosine_b, "s-", color=COLOR_B, lw=2, ms=3, label="B")
            ax.axhline(1.0, color="gray", ls=":", alpha=0.4)
            ax.set_xlabel("Layer transition")
            ax.set_ylabel("Direction cosine (adjacent layers)")
            ax.set_title("Probe direction stability", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no direction_cosine data", ha="center",
                    transform=ax.transAxes); ax.axis("off")

        ax = axes[1]
        if ds.direction_norm_a is not None and ds.direction_norm_b is not None:
            x = np.arange(len(ds.direction_norm_a))
            ax.plot(x, ds.direction_norm_a, "o-", color=COLOR_A, lw=2, ms=3, label="A")
            ax.plot(x, ds.direction_norm_b, "s-", color=COLOR_B, lw=2, ms=3, label="B")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Probe weight ‖w‖")
            ax.set_title("Probe direction magnitude", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "no direction_norm data", ha="center",
                    transform=ax.transAxes); ax.axis("off")

        fig.suptitle(f"Direction dynamics: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_bic_comparison(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Three panels: effective_rank / explained_ratio / inter_class_ratio, A vs B."""
        bs = result.bic_shift
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        panels = [
            (axes[0], bs.effective_rank_a, bs.effective_rank_b,
             "Effective rank", "rank"),
            (axes[1], bs.explained_ratio_a, bs.explained_ratio_b,
             "Explained variance ratio", "ratio"),
            (axes[2], bs.inter_class_ratio_a, bs.inter_class_ratio_b,
             "Inter-class variance ratio", "ratio"),
        ]
        for ax, va, vb, title, ylabel in panels:
            if va is None or vb is None:
                ax.text(0.5, 0.5, f"no {title.lower()} data", ha="center",
                        transform=ax.transAxes); ax.axis("off"); continue
            x = np.arange(len(va))
            ax.plot(x, va, "o-", color=COLOR_A, lw=2, ms=3, label="A")
            ax.plot(x, vb, "s-", color=COLOR_B, lw=2, ms=3, label="B")
            ax.set_xlabel("Layer")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"BIC: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_norm_comparison(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Per-tag activation L2 norm profiles, A vs B."""
        shifts = result.norm_shifts or {}
        tags = list(shifts.keys())
        if not tags:
            fig = plt.figure(); return fig

        n = len(tags)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), sharey=True)
        axes = np.atleast_1d(axes).reshape(-1)

        for ax, tag in zip(axes, tags):
            ns = shifts[tag]
            x = np.arange(len(ns.norm_a))
            ax.plot(x, ns.norm_a, "o-", color=COLOR_A, lw=2, ms=3, label="A")
            ax.plot(x, ns.norm_b, "s-", color=COLOR_B, lw=2, ms=3, label="B")
            ax.set_title(f"[{tag}]\nr={ns.shape_correlation:.2f}, "
                         f"amp={ns.amplitude_ratio:.2f}x", fontsize=10)
            ax.set_xlabel("Layer")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        for ax in axes[n:]:
            ax.axis("off")

        axes[0].set_ylabel("Activation ‖h‖")
        fig.suptitle(f"Activation norm: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_head_diff(self, result: ComparisonResult, save_path=None) -> plt.Figure:
        """Three heatmaps of (layer × head): accuracy diff, separation diff, inter-class diff."""
        hs = result.head_shift
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        panels = [
            (axes[0], hs.accuracy_diff,    "Probe accuracy Δ (B − A)"),
            (axes[1], hs.separation_diff,  "Centroid separation Δ"),
            (axes[2], hs.inter_class_diff, "Inter-class ratio Δ"),
        ]
        for ax, mat, title in panels:
            vmax = float(np.abs(mat).max())
            vmax = max(vmax, 1e-6)
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_xlabel("Head"); ax.set_ylabel("Layer")
            ax.set_title(title, fontsize=11, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Annotate top-3 gain heads on the accuracy panel
        ax_acc = axes[0]
        for layer, head, _d in hs.top_gain_heads[:3]:
            ax_acc.scatter(head, layer, marker="o", s=80, facecolors="none",
                            edgecolors="lime", lw=2, zorder=5)
        for layer, head, _d in hs.top_loss_heads[:3]:
            ax_acc.scatter(head, layer, marker="s", s=80, facecolors="none",
                            edgecolors="black", lw=2, zorder=5)

        fig.suptitle(f"Per-head shift: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    def plot_centroid_alignment(self, result: ComparisonResult,
                                 save_path=None) -> plt.Figure:
        """Centroid-alignment heatmap (layer × class) + per-layer mean curve."""
        cs = result.centroid_shift
        fig = plt.figure(figsize=(14, 5.5))

        ax1 = fig.add_subplot(1, 2, 1)
        im = ax1.imshow(cs.centroid_cosine, aspect="auto", cmap="viridis",
                        vmin=0.0, vmax=1.0, origin="lower")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Layer")
        ax1.set_xticks(range(len(cs.tags_order)))
        ax1.set_xticklabels(cs.tags_order, rotation=30, ha="right", fontsize=9)
        ax1.set_title("cos(A.centroid, B.centroid)", fontsize=11, fontweight="bold")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ax1.axhline(cs.min_layer, color="red", ls="--", alpha=0.6)

        ax2 = fig.add_subplot(1, 2, 2)
        x = np.arange(cs.num_layers)
        ax2.plot(x, cs.mean_cosine_per_layer, "o-", color="#5C6BC0", lw=2, ms=3,
                 label="mean over classes")
        if cs.behavior_direction_cosine is not None:
            ax2.plot(x, cs.behavior_direction_cosine, "s--", color="#E53935",
                     lw=1.5, ms=3, alpha=0.7, label="top behavior direction")
        ax2.axvline(cs.min_layer, color="red", ls="--", alpha=0.4,
                    label=f"min @ L{cs.min_layer}")
        ax2.set_xlabel("Layer")
        ax2.set_ylabel("Cosine")
        ax2.set_ylim(min(0.0, float(cs.mean_cosine_per_layer.min()) - 0.05), 1.05)
        ax2.set_title("Per-layer mean alignment", fontsize=11, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        fig.suptitle(f"Class centroid alignment: {result.model_a} vs {result.model_b}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig
