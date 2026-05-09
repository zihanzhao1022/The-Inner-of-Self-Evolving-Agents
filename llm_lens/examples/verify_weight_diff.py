#!/usr/bin/env python3
"""
Weight-level diff scan across the (base / Instruct / AZR-Coder) trio.

Loads each model's state_dict on CPU, computes flatten-cosine of every
shared weight matrix between pairs (base–inst, base–azr, inst–azr).

Output:
    results/weight_diff_<TS>/
        weight_cosines.npz                   # numerical heatmaps + top-level
        top_cosines.json                     # readable summary
        weight_cosines_heatmap.png           # 3-panel layer × component heatmap
        run_meta.json

Usage:
    python -m llm_lens.examples.verify_weight_diff
    python -m llm_lens.examples.verify_weight_diff --model-set 7B
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.model_zoo import (
    MODEL_SETS, get_models, parse_dtype,
    DEFAULT_MODEL_SET, DEFAULT_DTYPE, DTYPE_CHOICES,
)


# Components inside each transformer layer (Qwen2/Llama-style decoder block).
# Order here = column order in the heatmap.
LAYER_COMPONENTS = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]
LAYER_COMPONENT_SHORT = {
    "input_layernorm.weight":          "in_LN",
    "self_attn.q_proj.weight":         "q",
    "self_attn.k_proj.weight":         "k",
    "self_attn.v_proj.weight":         "v",
    "self_attn.o_proj.weight":         "o",
    "post_attention_layernorm.weight": "out_LN",
    "mlp.gate_proj.weight":            "gate",
    "mlp.up_proj.weight":              "up",
    "mlp.down_proj.weight":            "down",
}


def matrix_cosine(A: torch.Tensor, B: torch.Tensor) -> float:
    """Flatten cosine, computed in float64.

    Why fp64: when matrices have ~3e8 elements (e.g. embed 151936 x 2048)
    and cos ≈ 1, fp32 accumulation of (a @ b) and ||a|| * ||b|| diverges
    by ~10% — verified empirically that fp32 yields cos > 1 for
    base–inst pairs, breaking the metric. fp64's 53-bit mantissa gives
    ~1e-8 cumulative error over 3e8 elements, which is well below the
    cosine signal scale we care about (down to 1e-3).
    """
    a = A.flatten().double()
    b = B.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def matrix_diff_norm(A: torch.Tensor, B: torch.Tensor) -> float:
    return float((A.double() - B.double()).norm())


def extract_weights(model, num_layers: int,
                    store_dtype: torch.dtype = torch.bfloat16
                    ) -> dict[str, torch.Tensor]:
    """Pull all weights of interest into a flat dict of CPU tensors.

    `store_dtype=bfloat16` halves RAM for the held weights (3B fp32 dict
    is ~12 GB, bf16 ~6 GB). Cosine math always re-upcasts per call.
    """
    def _grab(t: torch.Tensor) -> torch.Tensor:
        return t.detach().to(store_dtype).clone()

    out: dict[str, torch.Tensor] = {
        "embed":      _grab(model.model.embed_tokens.weight),
        "final_norm": _grab(model.model.norm.weight),
    }
    if not getattr(model.config, "tie_word_embeddings", False):
        out["lm_head"] = _grab(model.lm_head.weight)
    for l_idx in range(num_layers):
        layer = model.model.layers[l_idx]
        for comp in LAYER_COMPONENTS:
            obj = layer
            for part in comp.split("."):
                obj = getattr(obj, part)
            out[f"L{l_idx}__{comp}"] = _grab(obj)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Weight-level cosine diff scan across model trio")
    p.add_argument("--model-set", default=DEFAULT_MODEL_SET,
                   choices=list(MODEL_SETS),
                   help="Which trio to compare")
    p.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_CHOICES),
                   help="Load dtype. Default bf16 to fit two trio members in "
                        "RAM simultaneously on Windows (bf16 cosines are still "
                        "computed via fp32 upcast inside matrix_cosine — no "
                        "precision loss for the per-matrix flatten cosine).")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output-suffix", default=None)
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    out_ts = args.output_suffix or datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.results_root) / f"weight_diff_{out_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[verify_weight_diff] output → {out_dir}")

    from transformers import AutoModelForCausalLM
    models = get_models(args.model_set)
    if len(models) < 4:
        raise ValueError(
            f"need 4 models in quartet (base/inst/coder/AZR); got {len(models)}. "
            f"Update model_zoo.MODEL_SETS for {args.model_set!r}.")

    # The three TRAINING-AXIS pairs we care about. Restricting to these (not
    # all 6 cross-pairs) keeps RAM constant across the run: base+inst, then
    # free inst, base+coder, free base, coder+azr.
    base_full,  base_short  = models[0]
    inst_full,  inst_short  = models[1]
    coder_full, coder_short = models[2]
    azr_full,   azr_short   = models[3]
    print(f"  quartet: base={base_short}, inst={inst_short}, "
          f"coder={coder_short}, azr={azr_short}")
    print(f"  axis pairs:")
    print(f"    RLHF:           {base_short:<22} ↔ {inst_short}")
    print(f"    domain:         {base_short:<22} ↔ {coder_short}")
    print(f"    self_evolving:  {coder_short:<22} ↔ {azr_short}")

    n_comp = len(LAYER_COMPONENTS)
    pairs = [
        ("base",  "inst",  base_full,  inst_full),    # RLHF axis
        ("base",  "coder", base_full,  coder_full),   # domain axis
        ("coder", "azr",   coder_full, azr_full),     # self-evolving axis
    ]
    heatmaps: dict[str, np.ndarray] = {f"{a}_{b}": None for a, b, *_ in pairs}
    diff_norms: dict[str, np.ndarray] = {f"{a}_{b}": None for a, b, *_ in pairs}
    top_cosines: dict[str, dict[str, float]] = {f"{a}_{b}": {} for a, b, *_ in pairs}

    # Cache loaded weights so each model is loaded at most once across the
    # three pairs (base appears twice; coder appears twice). RAM peak with
    # 3 simultaneously held bf16 dicts is ~18 GB at fp32 storage cost
    # equivalents, so we hold at most 2 at once.
    loaded: dict[str, dict[str, torch.Tensor]] = {}
    num_layers = None
    tie = None

    def _ensure_loaded(short: str, full: str) -> dict[str, torch.Tensor]:
        nonlocal num_layers, tie
        if short in loaded:
            return loaded[short]
        print(f"\n  Loading {short} ({full})...")
        m = AutoModelForCausalLM.from_pretrained(full, dtype=dtype, device_map="cpu")
        if num_layers is None:
            num_layers = m.config.num_hidden_layers
            tie = bool(getattr(m.config, "tie_word_embeddings", False))
            print(f"    num_layers={num_layers}  tie_word_embeddings={tie}")
        loaded[short] = extract_weights(m, num_layers)
        del m
        gc.collect()
        print(f"    extracted {len(loaded[short])} weight tensors (bf16)")
        return loaded[short]

    def _evict(short: str) -> None:
        if short in loaded:
            del loaded[short]
            gc.collect()

    # ── Process each axis pair, evicting models between to keep RAM low ────
    for (a, b, a_full, b_full) in pairs:
        wa = _ensure_loaded(a, a_full)
        wb = _ensure_loaded(b, b_full)

        if num_layers is None or tie is None:
            raise RuntimeError("model meta not initialised")

        if not top_cosines.get("__top_keys"):
            top_cosines["__top_keys"] = ["embed", "final_norm"] + (["lm_head"] if not tie else [])
        top_keys = top_cosines["__top_keys"]

        pname = f"{a}_{b}"
        for tk in top_keys:
            top_cosines[pname][tk] = matrix_cosine(wa[tk], wb[tk])

        h = np.zeros((num_layers, n_comp), dtype=np.float32)
        d = np.zeros((num_layers, n_comp), dtype=np.float32)
        for l_idx in range(num_layers):
            for c_idx, comp in enumerate(LAYER_COMPONENTS):
                key = f"L{l_idx}__{comp}"
                h[l_idx, c_idx] = matrix_cosine(wa[key], wb[key])
                d[l_idx, c_idx] = matrix_diff_norm(wa[key], wb[key])
        heatmaps[pname]  = h
        diff_norms[pname] = d
        print(f"    {pname}: done")

        # Eviction: after RLHF (base, inst) we can drop inst; after domain
        # (base, coder) we can drop base; coder stays for self-evolving.
        if (a, b) == ("base", "inst"):
            _evict("inst")
        elif (a, b) == ("base", "coder"):
            _evict("base")
        elif (a, b) == ("coder", "azr"):
            _evict("coder")
            _evict("azr")

    top_keys = top_cosines.pop("__top_keys")

    # ── Save numerical artefacts ───────────────────────────────────────────
    np.savez(
        out_dir / "weight_cosines.npz",
        **{f"cos_{p}":  h for p, h in heatmaps.items()},
        **{f"diff_{p}": h for p, h in diff_norms.items()},
    )
    with open(out_dir / "top_cosines.json", "w", encoding="utf-8") as f:
        json.dump({
            "top_cosines": top_cosines,
            "tie_word_embeddings": tie,
            "components_order": LAYER_COMPONENTS,
        }, f, indent=2)
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": out_ts, "model_set": args.model_set, "dtype": args.dtype,
            "num_layers": int(num_layers), "tie": bool(tie),
            "models": {
                "base": base_full, "inst": inst_full,
                "coder": coder_full, "azr": azr_full,
            },
            "axis_pairs": {
                "RLHF":          [base_short, inst_short],
                "domain":        [base_short, coder_short],
                "self_evolving": [coder_short, azr_short],
            },
            "written_at": datetime.now().isoformat(),
        }, f, indent=2)

    # ── Plot heatmaps ──────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        n_pairs = len(pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs + 1, 10))
        if n_pairs == 1:
            axes = [axes]
        # Fixed [0, 1] range so panels are directly comparable. Cosines
        # in this experiment are normally >= 0.5; a value below 0.5 is a
        # strong "this matrix was significantly retrained" signal.
        for ax, (a, b, *_rest) in zip(axes, pairs):
            pname = f"{a}_{b}"
            h = heatmaps[pname]
            im = ax.imshow(h, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
            ax.set_xticks(range(n_comp))
            ax.set_xticklabels([LAYER_COMPONENT_SHORT[c] for c in LAYER_COMPONENTS],
                               rotation=45, ha="right", fontsize=9)
            ax.set_yticks(range(num_layers))
            ax.set_yticklabels([f"L{i}" for i in range(num_layers)], fontsize=6)
            ax.set_title(f"cos({a}, {b})")
            for l_idx in range(num_layers):
                for c_idx in range(n_comp):
                    v = h[l_idx, c_idx]
                    ax.text(c_idx, l_idx, f"{v:.2f}", ha="center", va="center",
                            fontsize=4,
                            color="white" if v < 0.5 else "black")
            plt.colorbar(im, ax=ax, fraction=0.05)
        for ax, (a, b, *_rest) in zip(axes, pairs):
            tc = top_cosines[f"{a}_{b}"]
            txt = "  ".join(f"{k}={v:.4f}" for k, v in tc.items())
            ax.text(0.5, -0.08, f"top: {txt}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)
        plt.suptitle(
            "Per-(layer, component) weight cosine — three training axes\n"
            "RLHF (base↔inst)   Domain (base↔coder)   Self-evolving (coder↔AZR)\n"
            "Green ≈ unchanged; Red = significantly retrained")
        plt.tight_layout()
        plt.savefig(out_dir / "weight_cosines_heatmap.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARN] plotting failed: {e}")

    # ── Console summary ────────────────────────────────────────────────────
    pair_names = [f"{a}_{b}" for a, b, *_rest in pairs]
    print("\n" + "=" * 68)
    print("  TOP-LEVEL COMPONENTS (cosine)")
    print("=" * 68)
    header = "  " + f"{'component':<14}" + "".join(f"{p.replace('_','–'):>14}" for p in pair_names)
    print(header)
    for tk in top_keys:
        row = "  " + f"{tk:<14}" + "".join(f"{top_cosines[p][tk]:>14.6f}" for p in pair_names)
        print(row)

    print("\n" + "=" * 68)
    print("  PER-LAYER MEAN COSINE (averaged over 9 components)")
    print("=" * 68)
    print("  " + f"{'layer':<6}" + "".join(f"{p.replace('_','–'):>14}" for p in pair_names))
    for l_idx in range(num_layers):
        row = "  " + f"L{l_idx:>3}".ljust(6) + "".join(
            f"{heatmaps[p][l_idx].mean():>14.6f}" for p in pair_names)
        print(row)

    print("\n" + "=" * 68)
    print("  MOST-CHANGED (lowest cos) PER PAIR")
    print("=" * 68)
    for a, b, *_rest in pairs:
        pname = f"{a}_{b}"
        h = heatmaps[pname]
        l_min, c_min = np.unravel_index(h.argmin(), h.shape)
        print(f"  {a}–{b:<5}  min={h[l_min, c_min]:.4f}  at L{l_min} {LAYER_COMPONENTS[c_min]}")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
