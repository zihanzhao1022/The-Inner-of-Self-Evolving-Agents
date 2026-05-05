"""
Layer Dynamics — Compute layer-wise representation dynamics for a single forward pass.

This is the analytical core. Given an ExtractionResult, it computes:
1. Processing intensity profile — how much each layer transforms its input
2. Trajectory geometry — PCA projection, curvature, drift
3. Norm profile — L2 norm evolution across layers
4. Inter-layer similarity matrix — full layer-to-layer cosine similarity

These are per-sample measurements. Aggregation across samples and comparison
across behavior categories happens in behavior.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .extractor import ExtractionResult


@dataclass
class DynamicsProfile:
    """Layer-wise dynamics for a single input, single token position."""

    num_layers: int
    token_idx: int

    # Core dynamics (all arrays indexed by layer)
    inter_layer_cosine: np.ndarray     # (num_layers-1,) cos_sim(layer_i, layer_{i+1})
    inter_layer_l2: np.ndarray         # (num_layers-1,) L2 distance
    processing_intensity: np.ndarray   # (num_layers-1,) = 1 - cosine (higher = more change)
    norm_per_layer: np.ndarray         # (num_layers,) L2 norm of hidden state

    # Derived features
    most_active_transition: int        # layer index with max processing intensity
    total_change: float                # sum of processing intensity
    early_change_ratio: float          # fraction of total change in first 1/3
    late_change_ratio: float           # fraction of total change in last 1/3

    # Trajectory geometry (from PCA)
    trajectory_2d: Optional[np.ndarray] = None  # (num_layers, 2) PCA projection
    cumulative_path_length: Optional[np.ndarray] = None  # (num_layers,)

    @property
    def normalized_intensity(self) -> np.ndarray:
        """Processing intensity normalized to [0,1] within this sample."""
        pi = self.processing_intensity
        mn, mx = pi.min(), pi.max()
        if mx - mn < 1e-10:
            return np.zeros_like(pi)
        return (pi - mn) / (mx - mn)


class LayerDynamics:
    """
    Compute layer-wise dynamics from extraction Qwen_Qwen2.5-3B.

    Usage:
        dynamics = LayerDynamics()
        profile = dynamics.compute(extraction_result, token_idx=-1)
        print(f"Most active: layer {profile.most_active_transition}")
    """

    @staticmethod
    def compute(result: ExtractionResult, token_idx: int = -1) -> DynamicsProfile:
        """Compute full dynamics profile for one token position."""
        residuals = result.get_residuals(token_idx)  # (num_layers, hidden_dim)
        num_layers = residuals.shape[0]

        # Inter-layer metrics
        cosines, l2s = [], []
        for i in range(num_layers - 1):
            a, b = residuals[i], residuals[i + 1]
            cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            l2 = torch.norm(a - b).item()
            cosines.append(cos)
            l2s.append(l2)

        cosine_arr = np.array(cosines)
        l2_arr = np.array(l2s)
        intensity = 1.0 - cosine_arr  # higher = more change

        # Norms
        norms = torch.norm(residuals, dim=-1).numpy()

        # Derived features
        most_active = int(intensity.argmax())
        total = intensity.sum()

        third = len(intensity) // 3
        early = intensity[:third].sum() / (total + 1e-10)
        late = intensity[-third:].sum() / (total + 1e-10)

        # Trajectory geometry (PCA to 2D)
        traj_2d, path_len = LayerDynamics._compute_trajectory(residuals)

        return DynamicsProfile(
            num_layers=num_layers,
            token_idx=token_idx,
            inter_layer_cosine=cosine_arr,
            inter_layer_l2=l2_arr,
            processing_intensity=intensity,
            norm_per_layer=norms,
            most_active_transition=most_active,
            total_change=total,
            early_change_ratio=early,
            late_change_ratio=late,
            trajectory_2d=traj_2d,
            cumulative_path_length=path_len,
        )

    @staticmethod
    def compute_bifurcation(
        result_a: ExtractionResult,
        result_b: ExtractionResult,
        token_idx: int = -1,
        threshold: float = 0.9,
    ) -> dict:
        """
        Compare two extraction Qwen_Qwen2.5-3B (e.g., harmful vs safe on same prompt).
        Find the bifurcation point where they diverge.

        Returns dict with:
            cosine_per_layer: (num_layers,) cosine similarity at each layer
            bifurcation_layer: first layer where cos_sim < threshold (or None)
            bifurcation_sharpness: rate of cosine drop at bifurcation point
        """
        n = min(result_a.num_layers, result_b.num_layers)
        cosines = []
        for i in range(n):
            va = result_a.residual_stream[f"layer_{i}"][0, token_idx, :]
            vb = result_b.residual_stream[f"layer_{i}"][0, token_idx, :]
            cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
            cosines.append(cos)

        cosine_arr = np.array(cosines)

        # Find bifurcation
        bif_layer = None
        for i, c in enumerate(cosines):
            if c < threshold:
                bif_layer = i
                break

        # Sharpness: max drop in cosine between consecutive layers
        drops = -np.diff(cosine_arr)
        sharpness = drops.max() if len(drops) > 0 else 0.0
        sharpest_layer = int(drops.argmax()) if len(drops) > 0 else 0

        return {
            "cosine_per_layer": cosine_arr,
            "bifurcation_layer": bif_layer,
            "bifurcation_depth_ratio": bif_layer / n if bif_layer is not None else None,
            "sharpest_drop_layer": sharpest_layer,
            "sharpest_drop_depth_ratio": sharpest_layer / n,
            "bifurcation_sharpness": sharpness,
        }

    @staticmethod
    def _compute_trajectory(residuals: torch.Tensor) -> tuple:
        """PCA projection to 2D and cumulative path length."""
        from sklearn.decomposition import PCA

        vecs = residuals.numpy()
        pca = PCA(n_components=2)
        proj = pca.fit_transform(vecs)

        # Cumulative path length
        dists = [0.0]
        for i in range(1, len(proj)):
            d = np.linalg.norm(proj[i] - proj[i - 1])
            dists.append(dists[-1] + d)
        path_len = np.array(dists)

        return proj, path_len
