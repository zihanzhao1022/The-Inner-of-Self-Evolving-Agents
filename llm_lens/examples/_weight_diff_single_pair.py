#!/usr/bin/env python3
"""
Worker subprocess for 7B weight-diff: load 2 models on CPU (bf16),
compute per-matrix cosines on GPU (fp64), save per-pair artifacts,
exit.

Started by `run_weight_diff_subprocess.py`. Each invocation is a fresh
Python process, so RAM peaks reset between pairs — works around the
HF from_pretrained transient peak that OOMed 7B in the in-process
v1–v5 attempts.

Usage:
    python -m llm_lens.examples._weight_diff_single_pair \\
        --a-full Qwen/Qwen2.5-7B --a-short Qwen2.5-7B \\
        --b-full Qwen/Qwen2.5-Coder-7B --b-short Qwen2.5-Coder-7B \\
        --out-dir results/weight_diff_20260512-7B/ \\
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.model_zoo import parse_dtype


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


def gpu_cosine_fp64(A_cpu: torch.Tensor, B_cpu: torch.Tensor,
                    use_gpu: bool = True) -> tuple[float, float]:
    """Flatten cosine + l2-diff-norm in fp64, optionally on GPU.

    Falls back to CPU if GPU OOMs on this matrix (only embed risks this:
    152064*3584*8 bytes = 4.4 GB each = 8.8 GB pair, fits in 11 GB).
    """
    if use_gpu and torch.cuda.is_available():
        try:
            a = A_cpu.flatten().to("cuda", dtype=torch.float64, non_blocking=True)
            b = B_cpu.flatten().to("cuda", dtype=torch.float64, non_blocking=True)
            cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
            diff = float((a - b).norm())
            del a, b
            torch.cuda.empty_cache()
            return cos, diff
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    # CPU fallback
    a = A_cpu.flatten().to(torch.float64)
    b = B_cpu.flatten().to(torch.float64)
    cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
    diff = float((a - b).norm())
    return cos, diff


def extract_weights(model, num_layers: int,
                    store_dtype: torch.dtype = torch.bfloat16
                    ) -> dict[str, torch.Tensor]:
    """Pull all weights of interest into a flat dict of CPU tensors."""
    def _grab(t):
        return t.detach().to(store_dtype).clone()

    out = {
        "embed": _grab(model.model.embed_tokens.weight),
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a-full", required=True)
    ap.add_argument("--a-short", required=True)
    ap.add_argument("--b-full", required=True)
    ap.add_argument("--b-short", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--no-gpu", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[worker] {args.a_short}  ↔  {args.b_short}", flush=True)
    print(f"[worker] dtype={args.dtype}  gpu={'on' if (not args.no_gpu and torch.cuda.is_available()) else 'off'}",
          flush=True)

    dtype = parse_dtype(args.dtype)

    # Load A
    from transformers import AutoModelForCausalLM
    print(f"[worker] loading A: {args.a_full}", flush=True)
    mA = AutoModelForCausalLM.from_pretrained(
        args.a_full, dtype=dtype, device_map="cpu", low_cpu_mem_usage=True)
    num_layers = mA.config.num_hidden_layers
    tie = bool(getattr(mA.config, "tie_word_embeddings", False))
    print(f"[worker]   num_layers={num_layers}, tie={tie}", flush=True)
    wA = extract_weights(mA, num_layers)
    del mA
    gc.collect()
    print(f"[worker]   A extracted ({len(wA)} tensors)", flush=True)

    # Load B
    print(f"[worker] loading B: {args.b_full}", flush=True)
    mB = AutoModelForCausalLM.from_pretrained(
        args.b_full, dtype=dtype, device_map="cpu", low_cpu_mem_usage=True)
    if mB.config.num_hidden_layers != num_layers:
        raise ValueError(f"layer count mismatch: A={num_layers}, B={mB.config.num_hidden_layers}")
    wB = extract_weights(mB, num_layers)
    del mB
    gc.collect()
    print(f"[worker]   B extracted ({len(wB)} tensors)", flush=True)

    use_gpu = not args.no_gpu

    # Top-level keys
    top_keys = ["embed", "final_norm"] + (["lm_head"] if not tie else [])
    top_cosines = {}
    print(f"[worker] computing top cosines: {top_keys}", flush=True)
    for tk in top_keys:
        cos, diff = gpu_cosine_fp64(wA[tk], wB[tk], use_gpu=use_gpu)
        top_cosines[tk] = {"cosine": cos, "diff_norm": diff}
        print(f"[worker]   {tk}: cos={cos:.10f}, diff={diff:.4f}", flush=True)

    # Layer × component matrix
    n_comp = len(LAYER_COMPONENTS)
    cos_mat = np.zeros((num_layers, n_comp), dtype=np.float32)
    diff_mat = np.zeros((num_layers, n_comp), dtype=np.float32)
    print(f"[worker] computing {num_layers} × {n_comp} = {num_layers*n_comp} layer cosines...", flush=True)
    import time
    t0 = time.time()
    for l_idx in range(num_layers):
        for c_idx, comp in enumerate(LAYER_COMPONENTS):
            key = f"L{l_idx}__{comp}"
            cos, diff = gpu_cosine_fp64(wA[key], wB[key], use_gpu=use_gpu)
            cos_mat[l_idx, c_idx] = cos
            diff_mat[l_idx, c_idx] = diff
        if (l_idx + 1) % 7 == 0 or (l_idx + 1) == num_layers:
            elapsed = time.time() - t0
            rate = (l_idx + 1) / max(elapsed, 1e-6)
            eta = (num_layers - l_idx - 1) / max(rate, 1e-6)
            print(f"[worker]   layer {l_idx + 1}/{num_layers}  rate={rate:.2f} L/s  eta={eta:.0f}s", flush=True)

    # Save per-pair artifacts
    pair_tag = f"{args.a_short}__{args.b_short}"
    npz_path = out_dir / f"pair_{pair_tag}.npz"
    json_path = out_dir / f"top_cosines_{pair_tag}.json"
    np.savez(npz_path,
             cos=cos_mat, diff=diff_mat,
             a_short=args.a_short, b_short=args.b_short,
             num_layers=num_layers, tie=tie)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "a_short": args.a_short,
            "b_short": args.b_short,
            "a_full": args.a_full,
            "b_full": args.b_full,
            "tie_word_embeddings": tie,
            "num_layers": num_layers,
            "top_cosines": top_cosines,
            "layer_cos_min": float(cos_mat.min()),
            "layer_cos_max": float(cos_mat.max()),
            "layer_cos_mean": float(cos_mat.mean()),
        }, fh, indent=2)
    print(f"[worker] saved {npz_path} + {json_path}", flush=True)
    print(f"[worker] done — top cos: " + ", ".join(
        f"{k}={v['cosine']:.6f}" for k, v in top_cosines.items()), flush=True)


if __name__ == "__main__":
    main()
