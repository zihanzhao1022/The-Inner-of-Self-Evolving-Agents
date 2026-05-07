"""
Steering vector extraction, cross-model Procrustes transfer, and
hook-based residual-stream injection.

The pipeline answers: "After Procrustes-aligning class centroids between two
same-architecture models, does a steering vector trained on model A still
produce its intended effect when injected into model B?"

Workflow:
    1. compute_steering_vector(centroids, layer, target, ref) — class-mean
       difference at one layer (DiM steering).
    2. procrustes_rotation_matrix(src_layer, dst_layer) — optimal rotation
       from src centroid frame to dst centroid frame.
    3. transfer_vector(vec, src_layer, dst_layer) — apply that rotation.
    4. SteeringContext — context manager that attaches a forward hook on a
       chosen decoder layer, adds (strength * vector) to its output.
    5. score_activations_against_centroids — measures how much a steered
       activation moved toward a target class centroid (cosine basis).
"""

from __future__ import annotations

import numpy as np
import torch
from contextlib import contextmanager
from scipy.linalg import orthogonal_procrustes


# ── Steering vector construction ────────────────────────────────────────────

def compute_steering_vector(
    centroids: np.ndarray,
    layer: int,
    target_idx: int,
    ref_idx: int,
    normalize: bool = False,
    target_norm: float | None = None,
) -> np.ndarray:
    """Difference of class means at `layer`: centroid[target] - centroid[ref].

    Args:
        centroids: shape (num_layers, num_classes, hidden_dim).
        layer: which layer to extract.
        target_idx, ref_idx: integer class indices.
        normalize: if True, return unit-norm vector. Default False — use the
            natural inter-class distance as the vector magnitude, so a
            steering strength of 1.0 corresponds to "add one natural
            class-mean difference" of perturbation.
        target_norm: if given, scale the returned vector to this Euclidean
            norm. Useful for cross-model fairness — pass the native model's
            class-difference norm so a transferred vector has matched
            magnitude.

    Returns:
        (hidden_dim,) numpy array.
    """
    v = centroids[layer, target_idx] - centroids[layer, ref_idx]
    if normalize:
        n = np.linalg.norm(v)
        v = v / (n + 1e-10)
    if target_norm is not None:
        n = np.linalg.norm(v)
        v = v / (n + 1e-10) * target_norm
    return v.astype(np.float32)


# ── Procrustes alignment between two centroid frames ────────────────────────

def procrustes_rotation_matrix(
    src_centroids_layer: np.ndarray,
    dst_centroids_layer: np.ndarray,
) -> np.ndarray:
    """Orthogonal matrix R that best maps src centroids onto dst centroids.

    Both inputs: shape (num_classes, hidden_dim). Each is centered in-function
    (the rotation only — translation handled separately by user if needed).

    Returns: (hidden_dim, hidden_dim) orthogonal matrix.
    Solves min_R ||A_centered @ R − B_centered||_F where R is orthogonal.
    """
    A = src_centroids_layer - src_centroids_layer.mean(axis=0, keepdims=True)
    B = dst_centroids_layer - dst_centroids_layer.mean(axis=0, keepdims=True)
    R, _ = orthogonal_procrustes(A, B)
    return R


def transfer_vector(
    vec: np.ndarray,
    src_centroids_layer: np.ndarray,
    dst_centroids_layer: np.ndarray,
) -> np.ndarray:
    """Rotate a steering vector from src model's frame into dst model's frame
    using Procrustes alignment of the class centroids at the same layer.

    Steering vectors are class-mean differences, so translation cancels and
    only the rotation matters here.
    """
    R = procrustes_rotation_matrix(src_centroids_layer, dst_centroids_layer)
    return vec @ R


# ── Hook-based injection ────────────────────────────────────────────────────

def _make_steering_hook(vector_tensor: torch.Tensor, strength: float):
    """Forward hook that adds (strength * vector) to a decoder layer's output."""
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            v = vector_tensor.to(device=hidden.device, dtype=hidden.dtype)
            return (hidden + strength * v,) + output[1:]
        v = vector_tensor.to(device=output.device, dtype=output.dtype)
        return output + strength * v
    return hook


@contextmanager
def steering_context(
    layer_module: torch.nn.Module,
    vector: np.ndarray | torch.Tensor | None,
    strength: float,
):
    """Attach a steering hook for the duration of the with-block.

    Pass strength=0 or vector=None to turn off injection (no hook attached —
    used to share code between baseline and steered runs).
    """
    if vector is None or strength == 0.0:
        yield
        return
    if isinstance(vector, np.ndarray):
        vector = torch.from_numpy(vector)
    handle = layer_module.register_forward_hook(_make_steering_hook(vector, strength))
    try:
        yield
    finally:
        handle.remove()


# ── Forward + capture (for scoring) ─────────────────────────────────────────

def forward_capture_last_layer(
    model,
    tokenizer,
    prompts: list[str],
    capture_layer_module: torch.nn.Module,
    inject_layer_module: torch.nn.Module | None = None,
    steering_vec: np.ndarray | None = None,
    strength: float = 0.0,
    device: str | None = None,
    token_idx: int = -1,
) -> np.ndarray:
    """Run prompts through `model` with optional steering; return captured
    activations from `capture_layer_module` at `token_idx`.

    Returns:
        (n_prompts, hidden_dim) np.ndarray (float32).
    """
    if device is None:
        device = next(model.parameters()).device

    captured: list[torch.Tensor] = []

    def cap_hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        captured.append(h.detach().clone())

    activations: list[np.ndarray] = []

    # Register steering BEFORE capture so the capture hook sees the steered
    # output. Otherwise, when inject_layer == capture_layer (a common case for
    # late-layer experiments), capture would fire first and never see the
    # steering modification.
    ctx = (steering_context(inject_layer_module, steering_vec, strength)
           if inject_layer_module is not None and steering_vec is not None
              and strength != 0.0
           else _null_context())
    with ctx:
        cap_handle = capture_layer_module.register_forward_hook(cap_hook)
        try:
            for p in prompts:
                inputs = tokenizer(p, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs)
                h = captured.pop()           # (1, seq_len, D)
                vec = h[0, token_idx, :].float().cpu().numpy()
                activations.append(vec)
                captured.clear()
        finally:
            cap_handle.remove()

    return np.stack(activations).astype(np.float32)


@contextmanager
def _null_context():
    yield


# ── Scoring ─────────────────────────────────────────────────────────────────

def cosine_to_centroids(
    activations: np.ndarray,
    centroids_layer: np.ndarray,
) -> np.ndarray:
    """Cosine of each activation against each centroid.

    Args:
        activations: (n_prompts, D)
        centroids_layer: (n_classes, D)
    Returns:
        (n_prompts, n_classes)
    """
    a = activations / (np.linalg.norm(activations, axis=1, keepdims=True) + 1e-10)
    c = centroids_layer / (np.linalg.norm(centroids_layer, axis=1, keepdims=True) + 1e-10)
    return a @ c.T


def transfer_score(
    activations: np.ndarray,
    centroids_layer: np.ndarray,
    target_idx: int,
    ref_idx: int,
) -> np.ndarray:
    """Per-prompt steering effect:  cos(a, target_centroid) − cos(a, ref_centroid).

    Positive = activation pulled toward target class. Returns (n_prompts,).
    """
    cos = cosine_to_centroids(activations, centroids_layer)
    return cos[:, target_idx] - cos[:, ref_idx]
