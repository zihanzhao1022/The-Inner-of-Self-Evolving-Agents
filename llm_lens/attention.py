"""
Attention Head Analyzer — per-attention-head behavior probing.

Identifies which attention heads carry behavior-discriminative information
by training a small linear probe on each head's pre-projection output
(the head_dim-sized vector before o_proj mixes heads back together).

Requires the extractor to be created with `capture_heads=True` so that
ExtractionResult carries `head_outputs`.

Pipeline (per layer × head):
  1. Stack the head's last-token output across all samples → (n_samples, head_dim)
  2. Train standardized LogisticRegression with cross-val → probe accuracy
  3. Compute class centroids in head_dim space → centroid separation
  4. Compute LDA-style trace ratio tr(Σ_b)/tr(Σ_t) — direct from activations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score


@dataclass
class HeadAnalysisResult:
    """Per-(layer, head) behavior analysis."""
    num_layers: int
    num_heads: int
    head_dim: int
    tags_order: list[str]

    # Core matrices, all (num_layers, num_heads)
    head_probe_accuracy: np.ndarray       # cross-val LR accuracy per head
    head_separation: np.ndarray           # mean pairwise class-centroid cosine distance
    head_inter_class_ratio: np.ndarray    # tr(Σ_b)/tr(Σ_t) per head — activation-based

    # Heavy: (num_layers, num_heads, n_classes, head_dim)
    head_centroids: Optional[np.ndarray] = None

    def get_top_heads(self, k: int = 10) -> list[tuple[int, int, float]]:
        """Top-K (layer, head, accuracy) globally."""
        flat = self.head_probe_accuracy.flatten()
        top_idx = np.argsort(flat)[::-1][:k]
        out = []
        for idx in top_idx:
            l = int(idx // self.num_heads)
            h = int(idx % self.num_heads)
            out.append((l, h, float(flat[idx])))
        return out

    def get_top_heads_per_layer(self) -> list[tuple[int, int, float]]:
        """Best head in each layer."""
        out = []
        for l in range(self.num_layers):
            h = int(self.head_probe_accuracy[l].argmax())
            out.append((l, h, float(self.head_probe_accuracy[l, h])))
        return out

    def print_summary(self, k: int = 10):
        print(f"\n── Per-Head Analysis ──")
        print(f"  {self.num_layers} layers × {self.num_heads} heads (head_dim={self.head_dim})")
        print(f"  Head accuracy: mean={self.head_probe_accuracy.mean():.3f}, "
              f"max={self.head_probe_accuracy.max():.3f}, "
              f"min={self.head_probe_accuracy.min():.3f}")
        print(f"\n  Top-{k} most discriminative heads:")
        for l, h, acc in self.get_top_heads(k):
            ratio = self.head_inter_class_ratio[l, h]
            print(f"    L{l:2d}H{h:2d}: acc={acc:.3f}, inter/total={ratio:.3f}")


class HeadAnalyzer:
    """Train per-head linear probes to find behavior-discriminative heads."""

    def analyze(self, samples, tags: list[str], min_per_tag: int = 3,
                with_centroids: bool = True) -> HeadAnalysisResult:
        """
        Args:
            samples: iterable of objects exposing `.tag` and `.extraction`
                     where extraction.head_outputs is populated
            tags: behavior tags to analyze (already filtered by min_per_tag upstream)
            with_centroids: if True, store the (n_layers, n_heads, n_classes, head_dim)
                            centroid tensor (~ MBs for typical sizes)
        """
        tag_samples = {t: [s for s in samples if s.tag == t] for t in tags}

        first = samples[0]
        ext = first.extraction
        if not ext.head_outputs:
            raise ValueError(
                "Sample extractions have no head outputs. "
                "Re-create ActivationExtractor with capture_heads=True.")

        num_layers = ext.num_layers
        num_heads = ext.num_heads
        head_dim = ext.head_dim
        n_classes = len(tags)

        head_probe_acc = np.zeros((num_layers, num_heads))
        head_separation = np.zeros((num_layers, num_heads))
        head_inter_class = np.zeros((num_layers, num_heads))
        head_centroids = (np.zeros((num_layers, num_heads, n_classes, head_dim))
                          if with_centroids else None)

        # Pre-stack: per-class tensor of shape (n_class_samples, num_layers, num_heads, head_dim)
        all_vecs_per_class = {}
        for tag in tags:
            stacked = []
            for s in tag_samples[tag]:
                hvec = s.extraction.get_head_outputs(-1)  # (num_layers, num_heads, head_dim)
                stacked.append(hvec.numpy())
            if stacked:
                all_vecs_per_class[tag] = np.stack(stacked)
            else:
                all_vecs_per_class[tag] = np.zeros((0, num_layers, num_heads, head_dim))

        class_counts = np.array([len(tag_samples[t]) for t in tags], dtype=np.int64)
        n_total = int(class_counts.sum())

        for layer in range(num_layers):
            for head in range(num_heads):
                X_parts, y_parts = [], []
                class_means = []
                for c, tag in enumerate(tags):
                    vecs = all_vecs_per_class[tag][:, layer, head, :]  # (n_c, head_dim)
                    X_parts.append(vecs)
                    y_parts.extend([c] * len(vecs))
                    class_means.append(vecs.mean(0))

                X = np.concatenate(X_parts, axis=0)
                y = np.array(y_parts)
                class_means = np.stack(class_means)  # (n_classes, head_dim)
                if with_centroids:
                    head_centroids[layer, head] = class_means

                # ── Probe ──
                min_class = int(class_counts.min())
                cv = min(5, min_class)
                clf = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, C=1.0),
                )
                if cv >= 2 and len(X) >= 6:
                    try:
                        scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
                        head_probe_acc[layer, head] = scores.mean()
                    except Exception:
                        clf.fit(X, y)
                        head_probe_acc[layer, head] = clf.score(X, y)
                else:
                    clf.fit(X, y)
                    head_probe_acc[layer, head] = clf.score(X, y)

                # ── LDA-style trace ratio: tr(Σ_b) / tr(Σ_t) ──
                grand_mean = (class_means * class_counts[:, None] / n_total).sum(0)
                centered = class_means - grand_mean
                trace_b = float(np.sum(class_counts * np.sum(centered ** 2, axis=1)) / n_total)
                trace_t = float(np.var(X, axis=0).sum())
                head_inter_class[layer, head] = trace_b / (trace_t + 1e-12)

                # ── Mean pairwise centroid cosine distance ──
                seps = []
                for i in range(n_classes):
                    for j in range(i + 1, n_classes):
                        a, b = class_means[i], class_means[j]
                        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
                        cos = float(np.dot(a, b) / denom)
                        seps.append(1.0 - cos)
                head_separation[layer, head] = float(np.mean(seps)) if seps else 0.0

        return HeadAnalysisResult(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            tags_order=list(tags),
            head_probe_accuracy=head_probe_acc,
            head_separation=head_separation,
            head_inter_class_ratio=head_inter_class,
            head_centroids=head_centroids,
        )
