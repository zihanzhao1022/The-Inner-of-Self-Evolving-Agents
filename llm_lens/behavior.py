"""
Behavior Mapper — Tag model inputs by output behavior, extract dynamics
signatures, compare across behavior categories.

This module answers RQ1 and RQ2:
  RQ1: At which layer does behavior diverge? (bifurcation timing)
  RQ2: Do different behaviors have unique dynamics signatures?

Usage:
    mapper = BehaviorMapper(extractor)
    mapper.add("How to pick a lock", "harmful")
    mapper.add("What is photosynthesis", "safe")
    report = mapper.analyze()
    report.print_summary()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

from .extractor import ActivationExtractor, ExtractionResult
from .logit_lens import LogitLens, LogitLensResult
from .dynamics import LayerDynamics, DynamicsProfile
from .attention import HeadAnalyzer, HeadAnalysisResult


@dataclass
class Sample:
    """One labeled input with captured internals."""
    text: str
    tag: str
    extraction: ExtractionResult
    dynamics: DynamicsProfile
    logit_lens: Optional[LogitLensResult] = None


@dataclass
class BehaviorSignature:
    """Aggregated dynamics signature for one behavior category."""
    tag: str
    n_samples: int
    mean_intensity: np.ndarray       # (num_layers-1,) mean processing intensity
    std_intensity: np.ndarray        # (num_layers-1,)
    mean_norm: np.ndarray            # (num_layers,)
    mean_most_active_layer: float    # average most-active transition layer
    mean_early_change: float         # avg fraction of change in first 1/3
    mean_late_change: float          # avg fraction of change in last 1/3
    logit_lens_consensus: Optional[list[tuple[str, float]]] = None  # (layer,) (top1, agreement%)


@dataclass
class LayerProbeResult:
    """Per-layer behavior analysis. Built entirely from activation matrices.

    Probe accuracy is still computed via LogisticRegression to measure linear
    separability, but the *direction* and *concentration* metrics no longer
    use the probe's weight vector — they are derived directly from the
    activation matrix X (n_samples, hidden_dim) and the class labels y at
    each layer.
    """
    num_layers: int
    probe_accuracy: np.ndarray       # (num_layers,)
    mean_separation: np.ndarray      # (num_layers,) cosine distance between category means
    best_layer: int
    best_accuracy: float
    best_depth_ratio: float          # best_layer / num_layers

    # ── Class structure in activation space (replaces probe_coefs) ──
    class_centroids: Optional[np.ndarray] = None      # (num_layers, n_classes, hidden_dim)
    behavior_direction: Optional[np.ndarray] = None   # (num_layers, hidden_dim) — top SVD dir of centered class means
    tags_order: Optional[list[str]] = None            # class label ordering for centroids

    # ── Cross-layer direction dynamics (activation-based) ──
    # direction_cosine[l] = |cos(behavior_direction[l], behavior_direction[l+1])|
    # direction_norm[l]   = sqrt(tr(Σ_b)) at layer l — magnitude of inter-class signal
    direction_cosine: Optional[np.ndarray] = None     # (num_layers-1,)
    direction_norm: Optional[np.ndarray] = None       # (num_layers,)

    # ── Behavioral Information Concentration (activation-based) ──
    # All computed from class centroid SVD and X covariances, no probe weights.
    bic_effective_rank: Optional[np.ndarray] = None   # exp(entropy) of centered class-mean SVD spectrum
    bic_explained_ratio: Optional[np.ndarray] = None  # var in top (n_classes-1) inter-class directions / total var
    bic_inter_class_ratio: Optional[np.ndarray] = None  # tr(Σ_b) / tr(Σ_t) — direct LDA trace ratio


@dataclass
class BehaviorReport:
    """Full analysis report for one model."""
    model_name: str
    num_layers: int
    tags: list[str]
    sample_counts: dict[str, int]
    layer_probe: LayerProbeResult
    signatures: dict[str, BehaviorSignature]

    # Cross-behavior bifurcation (pairwise)
    pairwise_bifurcation: dict[tuple[str, str], dict] = field(default_factory=dict)

    # Optional per-attention-head analysis (populated when extractor was created
    # with capture_heads=True).
    head_analysis: Optional[HeadAnalysisResult] = None

    def print_summary(self):
        print(f"\n{'=' * 60}")
        print(f"Behavioral Dynamics Report: {self.model_name}")
        print(f"{'=' * 60}")
        print(f"Layers: {self.num_layers}")
        print(f"Categories: {self.tags}")
        for t, c in self.sample_counts.items():
            print(f"  [{t}]: {c} samples")

        probe = self.layer_probe
        print(f"\nBest discriminative layer: {probe.best_layer} "
              f"(depth ratio: {probe.best_depth_ratio:.2f}, "
              f"accuracy: {probe.best_accuracy:.3f})")

        print(f"\nTop 5 layers by probe accuracy:")
        top5 = np.argsort(probe.probe_accuracy)[::-1][:5]
        for l in top5:
            ratio = l / self.num_layers
            print(f"  Layer {l:3d} (depth {ratio:.2f}): acc={probe.probe_accuracy[l]:.3f}")

        print(f"\nPer-category dynamics:")
        for tag, sig in self.signatures.items():
            print(f"  [{tag}]:")
            print(f"    Most active layer: {sig.mean_most_active_layer:.1f} "
                  f"(depth ratio: {sig.mean_most_active_layer / self.num_layers:.2f})")
            print(f"    Early/late change: {sig.mean_early_change:.3f} / {sig.mean_late_change:.3f}")

        if self.pairwise_bifurcation:
            print(f"\nPairwise bifurcation analysis:")
            for (ta, tb), bif in self.pairwise_bifurcation.items():
                if bif["bifurcation_layer"] is not None:
                    print(f"  [{ta}] vs [{tb}]: bifurcation at layer {bif['bifurcation_layer']} "
                          f"(depth ratio: {bif['bifurcation_depth_ratio']:.2f})")
                else:
                    print(f"  [{ta}] vs [{tb}]: no clear bifurcation (threshold not crossed)")

        if self.head_analysis is not None:
            self.head_analysis.print_summary(k=10)


class BehaviorMapper:
    """
    Tag inputs by behavior, extract dynamics, compare categories.

    Workflow:
        1. mapper.add(text, tag) — add labeled samples
        2. mapper.analyze() — compute full report
        3. Use report for visualization or cross-scale comparison
    """

    def __init__(self, extractor: ActivationExtractor, lens: Optional[LogitLens] = None):
        self.extractor = extractor
        self.lens = lens or LogitLens(extractor)
        self.samples: list[Sample] = []
        self._tags: set[str] = set()

    def add(self, text: str, tag: str) -> Sample:
        """Add a labeled sample, extract immediately."""
        extraction = self.extractor.run(text)
        dynamics = LayerDynamics.compute(extraction)
        logit_lens = self.lens.decode(extraction)

        sample = Sample(
            text=text, tag=tag, extraction=extraction,
            dynamics=dynamics, logit_lens=logit_lens,
        )
        self.samples.append(sample)
        self._tags.add(tag)
        return sample

    def add_batch(self, texts: list[str], tags: list[str]):
        """Add multiple samples."""
        for text, tag in zip(texts, tags):
            self.add(text, tag)

    def analyze(self, min_per_tag: int = 3,
                with_head_analysis: bool = True,
                with_head_centroids: bool = True) -> BehaviorReport:
        """Run full analysis.

        Per-head attention analysis runs automatically if the samples carry
        head outputs (i.e., the extractor was created with capture_heads=True).
        Set with_head_analysis=False to skip it explicitly.
        """
        counts = self._tag_counts()
        valid_tags = sorted([t for t, c in counts.items() if c >= min_per_tag])
        assert len(valid_tags) >= 2, f"Need >= 2 tags with >= {min_per_tag} samples. Have: {counts}"

        num_layers = self.samples[0].extraction.num_layers

        # 1. Layer probe (activation-matrix-based)
        probe = self._compute_probe(valid_tags)

        # 2. Behavior signatures
        sigs = {}
        for tag in valid_tags:
            sigs[tag] = self._compute_signature(tag)

        # 3. Pairwise bifurcation (mean representation per tag)
        pairwise = {}
        for i, ta in enumerate(valid_tags):
            for tb in valid_tags[i + 1:]:
                bif = self._compute_pairwise_bifurcation(ta, tb)
                pairwise[(ta, tb)] = bif

        # 4. Per-attention-head analysis (optional, requires capture_heads=True)
        head_result = None
        first_ext = self.samples[0].extraction
        if with_head_analysis and first_ext.head_outputs:
            valid_samples = [s for s in self.samples if s.tag in valid_tags]
            head_result = HeadAnalyzer().analyze(
                valid_samples, valid_tags,
                min_per_tag=min_per_tag,
                with_centroids=with_head_centroids,
            )

        return BehaviorReport(
            model_name=self.extractor.model_name,
            num_layers=num_layers,
            tags=valid_tags,
            sample_counts=counts,
            layer_probe=probe,
            signatures=sigs,
            pairwise_bifurcation=pairwise,
            head_analysis=head_result,
        )

    def get_samples_by_tag(self, tag: str) -> list[Sample]:
        return [s for s in self.samples if s.tag == tag]

    # ─── Internal ───

    def _compute_probe(self, tags: list[str]) -> LayerProbeResult:
        """Per-layer behavior analysis built from the activation matrix.

        Linear probe accuracy still uses LogisticRegression — that is a
        property of the activations (whether classes are linearly separable),
        not of the probe. But every *direction* and *concentration* metric
        below is computed directly from X (n_samples, hidden_dim) and the
        class label structure, with no probe weight vector involved.
        """
        tag_samples = {t: self.get_samples_by_tag(t) for t in tags}
        num_layers = self.samples[0].extraction.num_layers
        hidden_dim = self.samples[0].extraction.hidden_dim
        n_classes = len(tags)

        accuracies = []
        separations = []
        class_centroids = np.zeros((num_layers, n_classes, hidden_dim))
        behavior_direction = np.zeros((num_layers, hidden_dim))

        direction_norm = np.zeros(num_layers)
        bic_effective_rank = np.zeros(num_layers)
        bic_explained_ratio = np.zeros(num_layers)
        bic_inter_class_ratio = np.zeros(num_layers)

        class_counts = np.array([len(tag_samples[t]) for t in tags], dtype=np.int64)
        n_total = int(class_counts.sum())

        for layer in range(num_layers):
            X_parts, y_parts, class_means = [], [], []
            for tag_idx, tag in enumerate(tags):
                vecs = torch.stack([s.extraction.get_residuals(-1)[layer]
                                    for s in tag_samples[tag]]).numpy()
                X_parts.append(vecs)
                y_parts.extend([tag_idx] * len(vecs))
                class_means.append(vecs.mean(0))

            X = np.concatenate(X_parts, axis=0)
            y = np.array(y_parts)
            class_means = np.stack(class_means)  # (n_classes, hidden_dim)
            class_centroids[layer] = class_means

            # ── Linear probe accuracy (LR weights are NOT kept) ──
            min_class = int(class_counts.min())
            cv = min(5, min_class)
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, C=1.0),
            )
            if cv >= 2 and len(X) >= 6:
                try:
                    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
                    acc = float(scores.mean())
                except Exception:
                    clf.fit(X, y)
                    acc = float(clf.score(X, y))
            else:
                clf.fit(X, y)
                acc = float(clf.score(X, y))
            accuracies.append(acc)

            # ── Activation-based behavior direction ──
            # Top singular direction of the centered class-mean matrix.
            grand_mean = (class_means * class_counts[:, None] / n_total).sum(0)
            centered_means = class_means - grand_mean  # (n_classes, hidden_dim)

            U, S, Vt = np.linalg.svd(centered_means, full_matrices=False)
            behavior_direction[layer] = Vt[0]  # top inter-class direction

            # ── BIC (Behavioral Information Concentration), activation-based ──
            # 1. Effective rank: Shannon entropy of normalized SVD spectrum of
            #    centered class means → exp(entropy). Bounded above by min(C-1, d).
            S_norm = S / (S.sum() + 1e-12)
            entropy = -float(np.sum(S_norm * np.log(S_norm + 1e-12)))
            bic_effective_rank[layer] = float(np.exp(entropy))

            # 2. Trace ratios (LDA criterion, computed directly on activations).
            #    Σ_b: between-class scatter, Σ_t: total scatter.
            trace_b = float(np.sum(class_counts * np.sum(centered_means ** 2, axis=1)) / n_total)
            trace_t = float(np.var(X, axis=0).sum())
            bic_inter_class_ratio[layer] = trace_b / (trace_t + 1e-12)
            direction_norm[layer] = float(np.sqrt(max(trace_b, 0.0)))

            # 3. Explained ratio: variance captured by projecting X onto the
            #    top (n_classes - 1) inter-class directions vs total variance.
            n_dirs = max(1, min(n_classes - 1, len(S)))
            behavior_basis = Vt[:n_dirs]  # (n_dirs, hidden_dim)
            X_centered = X - grand_mean
            X_proj = X_centered @ behavior_basis.T
            explained_var = float(np.var(X_proj, axis=0).sum())
            bic_explained_ratio[layer] = explained_var / (trace_t + 1e-12)

            # ── Mean pairwise centroid cosine distance ──
            seps = []
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    a = torch.from_numpy(class_means[i])
                    b = torch.from_numpy(class_means[j])
                    sep = 1 - F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
                    seps.append(sep)
            separations.append(float(np.mean(seps)) if seps else 0.0)

        # ── Cross-layer direction stability ──
        # SVD direction is sign-ambiguous, so take |cos| to ignore sign flips.
        direction_cosine = np.zeros(num_layers - 1)
        for layer in range(num_layers - 1):
            v1 = behavior_direction[layer]
            v2 = behavior_direction[layer + 1]
            n1 = float(np.linalg.norm(v1))
            n2 = float(np.linalg.norm(v2))
            if n1 > 1e-10 and n2 > 1e-10:
                direction_cosine[layer] = abs(float(np.dot(v1, v2)) / (n1 * n2))
            else:
                direction_cosine[layer] = 0.0

        acc_arr = np.array(accuracies)
        best = int(acc_arr.argmax())

        return LayerProbeResult(
            num_layers=num_layers,
            probe_accuracy=acc_arr,
            mean_separation=np.array(separations),
            best_layer=best,
            best_accuracy=float(acc_arr.max()),
            best_depth_ratio=best / num_layers,
            class_centroids=class_centroids,
            behavior_direction=behavior_direction,
            tags_order=list(tags),
            direction_cosine=direction_cosine,
            direction_norm=direction_norm,
            bic_effective_rank=bic_effective_rank,
            bic_explained_ratio=bic_explained_ratio,
            bic_inter_class_ratio=bic_inter_class_ratio,
        )

    def _compute_signature(self, tag: str) -> BehaviorSignature:
        """Aggregate dynamics for one behavior category."""
        samples = self.get_samples_by_tag(tag)

        intensities = np.stack([s.dynamics.processing_intensity for s in samples])
        norms = np.stack([s.dynamics.norm_per_layer for s in samples])

        # Logit lens consensus
        consensus = None
        if samples[0].logit_lens is not None:
            from collections import Counter
            consensus = []
            nl = samples[0].logit_lens.num_layers
            for l in range(nl):
                top1s = [s.logit_lens.top_tokens[l][0][0] for s in samples
                         if s.logit_lens is not None]
                c = Counter(top1s)
                tok, cnt = c.most_common(1)[0]
                consensus.append((tok, cnt / len(top1s)))

        return BehaviorSignature(
            tag=tag,
            n_samples=len(samples),
            mean_intensity=intensities.mean(0),
            std_intensity=intensities.std(0),
            mean_norm=norms.mean(0),
            mean_most_active_layer=np.mean([s.dynamics.most_active_transition for s in samples]),
            mean_early_change=np.mean([s.dynamics.early_change_ratio for s in samples]),
            mean_late_change=np.mean([s.dynamics.late_change_ratio for s in samples]),
            logit_lens_consensus=consensus,
        )

    def _compute_pairwise_bifurcation(self, tag_a: str, tag_b: str,
                                      n_bootstrap: int = 100) -> dict:
        """Compute mean-representation bifurcation with bootstrap CI."""
        sa = self.get_samples_by_tag(tag_a)
        sb = self.get_samples_by_tag(tag_b)
        n = self.samples[0].extraction.num_layers

        # Pre-extract all vectors: (n_samples, num_layers, hidden_dim)
        vecs_a = torch.stack([s.extraction.get_residuals(-1) for s in sa])
        vecs_b = torch.stack([s.extraction.get_residuals(-1) for s in sb])

        # Point estimate (mean of all samples)
        cosines = []
        for layer in range(n):
            mean_a = vecs_a[:, layer, :].mean(0)
            mean_b = vecs_b[:, layer, :].mean(0)
            cos = F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()
            cosines.append(cos)
        cos_arr = np.array(cosines)

        # Bootstrap
        rng = np.random.default_rng(42)
        boot_curves = np.zeros((n_bootstrap, n))
        na, nb = len(sa), len(sb)
        for b in range(n_bootstrap):
            idx_a = rng.choice(na, size=na, replace=True)
            idx_b = rng.choice(nb, size=nb, replace=True)
            for layer in range(n):
                ma = vecs_a[idx_a, layer, :].mean(0)
                mb = vecs_b[idx_b, layer, :].mean(0)
                boot_curves[b, layer] = F.cosine_similarity(
                    ma.unsqueeze(0), mb.unsqueeze(0)).item()

        cos_lower = np.percentile(boot_curves, 2.5, axis=0)
        cos_upper = np.percentile(boot_curves, 97.5, axis=0)

        # Bifurcation detection on point estimate
        drops = -np.diff(cos_arr)
        sharpest = int(drops.argmax()) if len(drops) > 0 else 0

        bif = None
        for i, c in enumerate(cosines):
            if c < 0.9:
                bif = i
                break

        return {
            "cosine_per_layer": cos_arr,
            "cosine_lower": cos_lower,
            "cosine_upper": cos_upper,
            "bifurcation_layer": bif,
            "bifurcation_depth_ratio": bif / n if bif is not None else None,
            "sharpest_drop_layer": sharpest,
            "sharpest_drop_depth_ratio": sharpest / n,
        }

    def _tag_counts(self) -> dict[str, int]:
        c = defaultdict(int)
        for s in self.samples:
            c[s.tag] += 1
        return dict(c)
