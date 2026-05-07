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


# Each set: (base, instruct, self-evolved) — ORDER MATTERS, downstream code
# treats index 0 as the "source" model in cross-model alignment / steering.
MODEL_SETS: dict[str, list[tuple[str, str]]] = {
    "3B": [
        ("Qwen/Qwen2.5-3B",                            "Qwen2.5-3B"),
        ("Qwen/Qwen2.5-3B-Instruct",                   "Qwen2.5-3B-Instruct"),
        ("andrewzh/Absolute_Zero_Reasoner-Coder-3b",   "AZR-Coder-3B"),
    ],
    "7B": [
        ("Qwen/Qwen2.5-7B",                            "Qwen2.5-7B"),
        ("Qwen/Qwen2.5-7B-Instruct",                   "Qwen2.5-7B-Instruct"),
        ("andrewzh/Absolute_Zero_Reasoner-Coder-7b",   "AZR-Coder-7B"),
    ],
    "14B": [
        ("Qwen/Qwen2.5-14B",                           "Qwen2.5-14B"),
        ("Qwen/Qwen2.5-14B-Instruct",                  "Qwen2.5-14B-Instruct"),
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
    """Standard three pairs: base↔instruct, base↔azr, instruct↔azr (short labels)."""
    models = get_models(model_set)
    if len(models) < 2:
        return []
    if len(models) == 2:
        return [(models[0][1], models[1][1])]
    base, instruct, azr = models[0][1], models[1][1], models[2][1]
    return [(base, instruct), (base, azr), (instruct, azr)]


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
