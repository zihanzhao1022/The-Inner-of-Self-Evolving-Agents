"""
Arditi-style refusal-direction extraction (Difference-in-Means).

Aligned with the open-source reference implementation
https://github.com/andyrdt/refusal_direction (Arditi et al. 2024,
"Refusal in LLMs Is Mediated by a Single Direction", NeurIPS 2024
spotlight).

Key alignment points to that reference:
  * **Token positions**: the full end-of-instruction range is captured —
    for our forced minimal Qwen ChatML wrapping
    `<|im_end|>\\n<|im_start|>assistant\\n`, that's the last 5 tokens
    `[<|im_end|>, \\n, <|im_start|>, assistant, \\n]` (positions -5..-1).
    Matches Arditi's `positions = list(range(-len(eoi_toks), 0))`.
  * **DiM accumulation in float64** (Arditi: "to avoid numerical errors").
  * **Output shape (n_pos, n_layer, d_model)** — full candidate-direction
    tensor, ready for downstream best-position/layer search or ablation
    intervention.
  * **Layer pruning helper**: `arditi_layer_prune_mask` drops the last 20 %
    of layers from "best layer" candidacy (Arditi default
    `prune_layer_percentage=0.2`), since refusal mediation in those layers
    overlaps with raw readout and confounds the causal interpretation.

Concept disclaimer:
  - Arditi's "best layer" is selected by **causal ablation** — the layer
    whose direction, when removed, most reduces the refusal score.
  - This module's `find_probe_emergence_layer` uses **observational**
    binary probe accuracy. It is correlated but NOT identical: probe acc
    high ⇒ direction is linearly readable, but does not prove causal
    mediation. A future intervention experiment is needed to reproduce
    Arditi's best-layer selection.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Sequence

from .extractor import ActivationExtractor


# Number of trailing positions to capture, matching Arditi's eoi_toks.
# For our forced minimal Qwen ChatML template, the eoi suffix
# '<|im_end|>\n<|im_start|>assistant\n' tokenises to exactly 5 tokens.
DEFAULT_EOI_LEN = 5


# ── Per-prompt activation capture across the eoi token range ────────────────

def capture_eoi_residuals(
    extractor: ActivationExtractor,
    prompts: Sequence[str],
    apply_chat_template: bool = True,
    eoi_len: int = DEFAULT_EOI_LEN,
    show_progress: bool = True,
) -> np.ndarray:
    """Capture residual stream at the last `eoi_len` token positions.

    Args:
        extractor: ActivationExtractor wrapping the target model.
        prompts: list of raw user prompts (will be wrapped if
            apply_chat_template=True).
        apply_chat_template: if True, use the forced minimal Qwen ChatML
            template; the last `eoi_len` tokens are the post-instruction
            template tokens — Arditi's eoi_toks range.
        eoi_len: how many trailing positions to capture per prompt.
            Default 5 (Qwen ChatML eoi length).

    Returns:
        np.ndarray of shape (n_prompts, eoi_len, num_layers, hidden_dim),
        float32. Position axis is ordered [-eoi_len, -eoi_len+1, ..., -1].
    """
    n = len(prompts)
    L = extractor.num_layers
    D = extractor.hidden_dim
    out = np.empty((n, eoi_len, L, D), dtype=np.float32)

    for i, p in enumerate(prompts):
        result = extractor.run(p, apply_chat_template=apply_chat_template)
        # ExtractionResult.get_residuals(token_idx=k) returns (num_layers, D);
        # iterate over the eoi range to build (eoi_len, L, D) per prompt.
        for j, tok_idx in enumerate(range(-eoi_len, 0)):
            v = result.get_residuals(token_idx=tok_idx).cpu().numpy()
            out[i, j] = v.astype(np.float32)
        if show_progress and (i + 1) % 25 == 0:
            print(f"    [refusal] captured {i + 1}/{n}")

    return out


# Backward-compat shim so older code using single-position capture still works.
def capture_post_instruction_residuals(
    extractor: ActivationExtractor,
    prompts: Sequence[str],
    apply_chat_template: bool = True,
    token_idx: int = -1,
    show_progress: bool = True,
) -> np.ndarray:
    """Capture residual stream at one token position. Returns (n, L, D).

    Kept for backward compatibility with the v1 single-position path. New
    code should use `capture_eoi_residuals` for the full Arditi-style
    multi-position tensor.
    """
    n = len(prompts)
    L = extractor.num_layers
    D = extractor.hidden_dim
    out = np.empty((n, L, D), dtype=np.float32)

    for i, p in enumerate(prompts):
        result = extractor.run(p, apply_chat_template=apply_chat_template)
        v = result.get_residuals(token_idx=token_idx).cpu().numpy()
        out[i] = v.astype(np.float32)
        if show_progress and (i + 1) % 25 == 0:
            print(f"    [refusal] captured {i + 1}/{n}")
    return out


# ── Difference-in-Means refusal direction (Arditi-style) ───────────────────

def extract_candidate_directions(
    harmful_residuals: np.ndarray,
    harmless_residuals: np.ndarray,
) -> np.ndarray:
    """Per-(position, layer) Difference-in-Means.

    Args:
        harmful_residuals:  (n_harmful,  n_pos, L, D)
        harmless_residuals: (n_harmless, n_pos, L, D)
    Returns:
        directions: (n_pos, L, D), float32.
            r[p, l] = mean(h_harmful[:, p, l, :]) - mean(h_harmless[:, p, l, :])
        Computed via float64 accumulation to avoid numerical drift on the
        (~3e8-element-per-layer) sums. Matches Arditi's float64 cache.
    """
    if harmful_residuals.ndim != 4 or harmless_residuals.ndim != 4:
        raise ValueError(
            f"expected (N, n_pos, L, D); got harmful={harmful_residuals.shape}, "
            f"harmless={harmless_residuals.shape}")

    # fp64 accumulation, downcast to fp32 at the end for storage parity.
    mu_harm = harmful_residuals.astype(np.float64).mean(axis=0)   # (n_pos, L, D)
    mu_safe = harmless_residuals.astype(np.float64).mean(axis=0)  # (n_pos, L, D)
    return (mu_harm - mu_safe).astype(np.float32)


# Backward-compat shim for single-position v1 inputs.
def extract_per_layer_directions(
    harmful_residuals: np.ndarray,
    harmless_residuals: np.ndarray,
) -> np.ndarray:
    """Single-position DiM. Kept for v1 callers.

    Args:
        harmful_residuals:  (n_harmful,  L, D)
        harmless_residuals: (n_harmless, L, D)
    Returns:
        (L, D) float32.
    """
    if harmful_residuals.ndim != 3 or harmless_residuals.ndim != 3:
        raise ValueError(
            f"expected (N, L, D); got harmful={harmful_residuals.shape}, "
            f"harmless={harmless_residuals.shape}")
    mu_harm = harmful_residuals.astype(np.float64).mean(axis=0)
    mu_safe = harmless_residuals.astype(np.float64).mean(axis=0)
    return (mu_harm - mu_safe).astype(np.float32)


def candidate_norms(directions: np.ndarray) -> np.ndarray:
    """L2 norm of every (pos, layer) direction. Shape: (n_pos, L)."""
    return np.linalg.norm(directions, axis=-1).astype(np.float32)


def cosine_per_position_layer(
    dirs_a: np.ndarray, dirs_b: np.ndarray) -> np.ndarray:
    """Per-(pos, layer) cosine between two models' candidate directions.

    Args:
        dirs_a, dirs_b: (n_pos, L, D)
    Returns:
        (n_pos, L) float32.
    """
    if dirs_a.shape != dirs_b.shape:
        raise ValueError(f"shape mismatch: {dirs_a.shape} vs {dirs_b.shape}")
    a = dirs_a.astype(np.float64)
    b = dirs_b.astype(np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=-1).astype(np.float32)


# Backward-compat: per-layer cosine for single-position inputs.
def cosine_per_layer(dirs_a: np.ndarray, dirs_b: np.ndarray) -> np.ndarray:
    if dirs_a.shape != dirs_b.shape:
        raise ValueError(f"shape mismatch: {dirs_a.shape} vs {dirs_b.shape}")
    a = dirs_a.astype(np.float64)
    b = dirs_b.astype(np.float64)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
    return (a * b).sum(axis=-1).astype(np.float32)


def per_layer_norms(directions: np.ndarray) -> np.ndarray:
    """Single-position norm. Kept for v1 callers."""
    return np.linalg.norm(directions, axis=-1).astype(np.float32)


# ── Per-(position, layer) linear probe ──────────────────────────────────────

def binary_probe_per_position_layer(
    harmful_residuals: np.ndarray,
    harmless_residuals: np.ndarray,
    test_size: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Train a logistic-regression probe at every (pos, layer), return test accs.

    Args:
        harmful_residuals:  (N_h, n_pos, L, D)
        harmless_residuals: (N_s, n_pos, L, D)

    Returns:
        (n_pos, L) float32 accuracy in [0, 1].
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    n_pos = harmful_residuals.shape[1]
    L     = harmful_residuals.shape[2]
    X_all = np.concatenate([harmless_residuals, harmful_residuals], axis=0)  # (N, n_pos, L, D)
    y     = np.concatenate([
        np.zeros(len(harmless_residuals), dtype=np.int32),
        np.ones (len(harmful_residuals),  dtype=np.int32),
    ])

    accs = np.empty((n_pos, L), dtype=np.float32)
    for p in range(n_pos):
        for l in range(L):
            X_pl = X_all[:, p, l, :]
            Xtr, Xte, ytr, yte = train_test_split(
                X_pl, y, test_size=test_size, random_state=seed, stratify=y)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, ytr)
            accs[p, l] = float(clf.score(Xte, yte))
    return accs


# Backward-compat: single-position probe.
def binary_probe_per_layer(
    harmful_residuals: np.ndarray,
    harmless_residuals: np.ndarray,
    test_size: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """Per-layer probe for v1 callers.

    Args:
        harmful_residuals:  (N_h, L, D)
        harmless_residuals: (N_s, L, D)
    Returns:
        (L,) float32.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    L = harmful_residuals.shape[1]
    X = np.concatenate([harmless_residuals, harmful_residuals], axis=0)
    y = np.concatenate([
        np.zeros(len(harmless_residuals), dtype=np.int32),
        np.ones (len(harmful_residuals),  dtype=np.int32),
    ])
    accs = np.empty(L, dtype=np.float32)
    for l in range(L):
        Xtr, Xte, ytr, yte = train_test_split(
            X[:, l, :], y, test_size=test_size, random_state=seed, stratify=y)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        accs[l] = float(clf.score(Xte, yte))
    return accs


# ── Layer pruning (Arditi default 20 % of last layers excluded) ─────────────

def arditi_layer_prune_mask(
    n_layers: int, prune_pct: float = 0.2) -> np.ndarray:
    """Boolean mask of shape (n_layers,). True = layer is a *candidate*
    (i.e. NOT pruned). False = layer is pruned (last `prune_pct` fraction).

    Matches the logic in `pipeline/submodules/select_direction.py:filter_fn`:
        layer >= int(n_layer * (1.0 - prune_layer_percentage))  → reject
    For 36 layers and 0.2 prune, this rejects layers 29..35 (the last 7).
    """
    cutoff = int(n_layers * (1.0 - prune_pct))
    mask = np.ones(n_layers, dtype=bool)
    mask[cutoff:] = False
    return mask


# ── Probe-based "emergence layer" — observational, NOT Arditi's causal best ─

def find_probe_emergence_layer(
    probe_accs: np.ndarray,
    rel_threshold: float = 0.98,
    apply_arditi_prune: bool = True,
    prune_pct: float = 0.2,
) -> int:
    """Shallowest layer that reaches `rel_threshold * max(probe_accs)`.

    Args:
        probe_accs: shape (L,) per-layer accuracy (must already have a
            position dimension flattened away — caller can `probe_accs[p].max(axis=0)`
            or pass a single position).
        apply_arditi_prune: if True, only consider layers in the
            unpruned region (first 80 %) when computing the peak threshold
            and selecting the shallowest crossing layer.

    NOTE: this is an OBSERVATIONAL emergence layer — based on linear
    probe accuracy. Arditi's "best layer" uses **causal ablation refusal
    score**. The two are correlated but not identical. Use this as
    a candidate / sanity check only, never report it as "the refusal
    mediator layer" without an intervention experiment.
    """
    if probe_accs.ndim != 1:
        raise ValueError(f"expected 1D probe_accs (n_layers,); got {probe_accs.shape}")
    L = len(probe_accs)
    mask = arditi_layer_prune_mask(L, prune_pct) if apply_arditi_prune else np.ones(L, dtype=bool)
    if not mask.any():
        return int(probe_accs.argmax())
    target = rel_threshold * probe_accs[mask].max()
    idxs = np.where((probe_accs >= target) & mask)[0]
    return int(idxs[0]) if len(idxs) > 0 else int(np.argmax(probe_accs[mask]))


# Backward-compat alias retained but with a deprecation note baked in.
def find_emergence_layer(
    probe_accs: np.ndarray,
    rel_threshold: float = 0.98,
) -> int:
    """Deprecated alias for `find_probe_emergence_layer`. Kept for v1 callers
    that didn't apply Arditi pruning. New code should use the explicit name."""
    return find_probe_emergence_layer(
        probe_accs, rel_threshold=rel_threshold, apply_arditi_prune=False)
