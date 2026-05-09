"""Single source of truth for the (base / instruct / self-evolved) model trios
at each size class, plus dtype helpers for the runners.

Usage:
    from llm_lens.model_zoo import (
        MODEL_SETS, get_models, get_compare_pairs,
        parse_dtype, DEFAULT_MODEL_SET, DEFAULT_DTYPE,
    )

    models = get_models("7B")          # [("Qwen/Qwen2.5-7B", "Qwen2.5-7B"), ...]
    pairs  = get_compare_pairs("7B")   # [("Qwen2.5-7B", "Qwen2.5-7B-Instruct"), ...]
"""

from __future__ import annotations

import torch


# Each set: (base, instruct, coder, self-evolved) — ORDER MATTERS.
# Index 0 = base (source for cross-model alignment / steering).
# Index 2 = Qwen2.5-Coder-3B is the *true* base of AZR-Coder-3B (verified via
# embedding cosine 1.000000), NOT Qwen2.5-3B. So three distinct training axes
# can now be isolated:
#   base  → instruct  : RLHF effect
#   base  → coder     : continued-pretraining (code domain) effect
#   coder → AZR-Coder : pure self-evolving RL effect
# Downstream code that hard-codes "trio" indexing (e.g. base/inst/azr) needs
# updating; legacy callers using get_compare_pairs() get all 6 cross-pairs.
MODEL_SETS: dict[str, list[tuple[str, str]]] = {
    "3B": [
        ("Qwen/Qwen2.5-3B",                            "Qwen2.5-3B"),
        ("Qwen/Qwen2.5-3B-Instruct",                   "Qwen2.5-3B-Instruct"),
        ("Qwen/Qwen2.5-Coder-3B",                      "Qwen2.5-Coder-3B"),
        ("andrewzh/Absolute_Zero_Reasoner-Coder-3b",   "AZR-Coder-3B"),
    ],
    # 7B has a 5-member quintet because andrewzh2 published an additional
    # AZR-Base-7B variant (presumably trained on plain Qwen2.5-7B without
    # the Coder continued-pretraining step). This lets us directly test
    # the "Coder pretraining provides robustness against self-evolving
    # drift" hypothesis: compare base→AZR-Base vs Coder→AZR-Coder.
    "7B": [
        ("Qwen/Qwen2.5-7B",                            "Qwen2.5-7B"),
        ("Qwen/Qwen2.5-7B-Instruct",                   "Qwen2.5-7B-Instruct"),
        ("Qwen/Qwen2.5-Coder-7B",                      "Qwen2.5-Coder-7B"),
        ("andrewzh2/Absolute_Zero_Reasoner-Base-7b",   "AZR-Base-7B"),
        ("andrewzh/Absolute_Zero_Reasoner-Coder-7b",   "AZR-Coder-7B"),
    ],
    "14B": [
        ("Qwen/Qwen2.5-14B",                           "Qwen2.5-14B"),
        ("Qwen/Qwen2.5-14B-Instruct",                  "Qwen2.5-14B-Instruct"),
        ("Qwen/Qwen2.5-Coder-14B",                     "Qwen2.5-Coder-14B"),
        ("andrewzh/Absolute_Zero_Reasoner-Coder-14b",  "AZR-Coder-14B"),
    ],
}

DEFAULT_MODEL_SET = "3B"
DEFAULT_DTYPE     = "float32"

DTYPE_CHOICES = ("float32", "bfloat16", "float16")


def get_models(model_set: str = DEFAULT_MODEL_SET) -> list[tuple[str, str]]:
    if model_set not in MODEL_SETS:
        raise ValueError(
            f"Unknown model_set {model_set!r}. Choose from {list(MODEL_SETS)}.")
    return list(MODEL_SETS[model_set])


def get_compare_pairs(model_set: str = DEFAULT_MODEL_SET) -> list[tuple[str, str]]:
    """Cross-pairs of all models in the set (n choose 2). For the 4-member
    quartet (base / instruct / coder / AZR), this yields 6 pairs. Downstream
    code interested in the three "axis" pairs should pick out:
        (base, instruct)  — RLHF axis
        (base, coder)     — continued-pretraining axis
        (coder, AZR)      — self-evolving axis
    """
    models = get_models(model_set)
    pairs = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pairs.append((models[i][1], models[j][1]))
    return pairs


def get_axis_pairs(model_set: str = DEFAULT_MODEL_SET) -> dict[str, tuple[str, str]]:
    """Return the named training-axis pairs by short label.

    For a 4-member quartet (3B/14B): [base, Instruct, Coder, AZR-Coder]
        RLHF                : base ↔ Instruct
        domain              : base ↔ Coder
        self_evolving       : Coder ↔ AZR-Coder

    For a 5-member quintet (7B): [base, Instruct, Coder, AZR-Base, AZR-Coder]
        RLHF                : base ↔ Instruct
        domain              : base ↔ Coder
        self_evolving_direct: base ↔ AZR-Base
        self_evolving_via_coder: Coder ↔ AZR-Coder
        +cross_AZR (optional): AZR-Base ↔ AZR-Coder (do the two RL paths
            converge?)
    """
    models = get_models(model_set)
    n = len(models)
    if n < 4:
        return {}
    if n == 4:
        base, instruct, coder, azr = (m[1] for m in models[:4])
        return {
            "RLHF":           (base, instruct),
            "domain":         (base, coder),
            "self_evolving":  (coder, azr),
        }
    if n == 5:
        base, instruct, coder, azr_base, azr_coder = (m[1] for m in models[:5])
        return {
            "RLHF":                    (base, instruct),
            "domain":                  (base, coder),
            "self_evolving_direct":    (base, azr_base),
            "self_evolving_via_coder": (coder, azr_coder),
            "cross_AZR":               (azr_base, azr_coder),
        }
    # Unknown N — return only RLHF / domain since they're well-defined.
    base, instruct, coder = (m[1] for m in models[:3])
    return {"RLHF": (base, instruct), "domain": (base, coder)}


def parse_dtype(name: str) -> torch.dtype:
    """Map a CLI string ('float32'/'bfloat16'/'float16') to a torch dtype."""
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        # Friendly aliases
        "fp32":   torch.float32,
        "bf16":   torch.bfloat16,
        "fp16":   torch.float16,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown dtype {name!r}. Choose from {sorted(set(DTYPE_CHOICES))}.")
    return mapping[name]
