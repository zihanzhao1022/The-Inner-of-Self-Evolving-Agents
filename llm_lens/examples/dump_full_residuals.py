#!/usr/bin/env python3
"""
Dump ALL tokens × ALL layers raw residual stream for every prompt.

For every prompt in harmbench200_alpaca200 (200 harmful + 200 harmless),
runs the model once, captures the residual stream after every transformer
block at every token position, and writes the lot to a single compressed
NPZ as fp16 named arrays.

This is the GROUND-TRUTH raw activation cache. All downstream analyses
(centroid, refusal direction, behavior direction, probe accuracy,
within-class spread, etc.) should be derivable from this file.

The output is too big for git (1.5–3 GB per model). Push it to
HuggingFace Datasets instead.

Output layout:
  results/full_residuals_<TS>/<SHORT>.npz
    harmful_<i>:        (n_layer, seq_len_i, hidden_dim) fp16, i = 0..n_harmful-1
    harmless_<i>:       (n_layer, seq_len_i, hidden_dim) fp16
    harmful_source_idx: (n_harmful,)   int32   — HarmBench source idx
    harmless_source_idx:(n_harmless,)  int32   — Alpaca source idx
    harmful_token_ids_<i>:  (seq_len_i,) int32  — token id sequence
    harmless_token_ids_<i>: (seq_len_i,) int32
    harmful_seq_lens:   (n_harmful,)   int32
    harmless_seq_lens:  (n_harmless,)  int32
    meta:               json-encoded string  — model name, dataset, dtype, etc.

Usage:
    python -m llm_lens.examples.dump_full_residuals \
        --model Qwen/Qwen2.5-3B \
        --short Qwen2.5-3B \
        --template-mode auto \
        --output-dir results/full_residuals_20260518-0020 \
        --dataset harmbench200_alpaca200 \
        --n-per-class 200 \
        --dtype float32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.datasets import load_prompts
from llm_lens.extractor import ActivationExtractor
from llm_lens.model_zoo import parse_dtype, get_native_template
from llm_lens.templates import wrap_prompt


def dump_for_model(model_name: str,
                   short: str,
                   template_mode: str,
                   dataset: str,
                   n_per_class: int,
                   dtype: torch.dtype,
                   out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{short}.npz")

    prompts = load_prompts(dataset, max_per_class=n_per_class)
    harmful  = prompts["harmful"]
    harmless = prompts["harmless"]

    if template_mode == "auto":
        template_mode = get_native_template(short)
    print(f"[dump] model={model_name} short={short} template={template_mode} "
          f"n_harmful={len(harmful)} n_harmless={len(harmless)} dtype={dtype}")

    extractor = ActivationExtractor(model_name, dtype=dtype, capture_heads=False)
    n_layers   = extractor.num_layers
    hidden_dim = extractor.hidden_dim
    print(f"[dump] loaded: n_layers={n_layers} hidden_dim={hidden_dim}")

    payload: dict[str, np.ndarray] = {}
    src_idx_h, src_idx_s = [], []
    seq_lens_h, seq_lens_s = [], []

    def _run_one(prompt_text: str) -> tuple[np.ndarray, np.ndarray]:
        # Apply the model's native template before feeding to extractor.
        wrapped = wrap_prompt(prompt_text, template_mode,
                              tokenizer=extractor.tokenizer)
        # Bypass the extractor's built-in ChatML wrapping by passing wrapped text
        # directly; apply_chat_template=False keeps the wrapped string verbatim.
        result = extractor.run(wrapped, apply_chat_template=False)
        # Shape: (n_layer, seq_len, hidden_dim); convert to fp16 on CPU.
        all_resids = result.get_all_token_residuals().cpu().to(torch.float16).numpy()
        token_ids = result.input_ids[0].cpu().numpy().astype(np.int32)
        return all_resids, token_ids

    t0 = time.time()
    for tag, pool, src_list, sl_list in [
        ("harmful",  harmful,  src_idx_h, seq_lens_h),
        ("harmless", harmless, src_idx_s, seq_lens_s),
    ]:
        for i, prompt_text in enumerate(pool):
            arr, tok_ids = _run_one(prompt_text)
            payload[f"{tag}_{i}"] = arr                          # (n_layer, seq_len, D) fp16
            payload[f"{tag}_token_ids_{i}"] = tok_ids            # (seq_len,) int32
            src_list.append(i)            # source idx = position in our paired set
            sl_list.append(arr.shape[1])
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                est_total = elapsed / ((i + 1) + (len(pool) if tag == "harmless" else 0))
                print(f"  [{tag}] {i+1}/{len(pool)}  elapsed={elapsed:.1f}s")

    payload["harmful_source_idx"]  = np.array(src_idx_h, dtype=np.int32)
    payload["harmless_source_idx"] = np.array(src_idx_s, dtype=np.int32)
    payload["harmful_seq_lens"]    = np.array(seq_lens_h, dtype=np.int32)
    payload["harmless_seq_lens"]   = np.array(seq_lens_s, dtype=np.int32)

    meta = {
        "model_name":   model_name,
        "short":        short,
        "template_mode": template_mode,
        "dataset":      dataset,
        "n_layers":     int(n_layers),
        "hidden_dim":   int(hidden_dim),
        "n_harmful":    len(harmful),
        "n_harmless":   len(harmless),
        "dtype_storage": "float16",
        "dtype_inference": str(dtype),
        "written_at":   datetime.now().isoformat(),
        "tokenizer_name": getattr(extractor.tokenizer, "name_or_path", model_name),
    }
    payload["meta"] = np.array(json.dumps(meta, indent=2))

    elapsed = time.time() - t0
    print(f"[dump] all {len(harmful) + len(harmless)} prompts captured in {elapsed:.1f}s")
    print(f"[dump] saving {out_path} (this may take a minute for compression)...")
    t_save = time.time()
    np.savez_compressed(out_path, **payload)
    print(f"[dump] saved in {time.time() - t_save:.1f}s, "
          f"size={os.path.getsize(out_path) / 2**30:.2f} GB")

    # Free GPU memory before returning so caller can do another model in same process
    del extractor
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(
        description="Dump all-tokens × all-layers raw residuals for one model")
    p.add_argument("--model", required=True, help="HF full name, e.g. Qwen/Qwen2.5-3B")
    p.add_argument("--short", required=True, help="Short label, e.g. Qwen2.5-3B")
    p.add_argument("--template-mode", default="auto",
                   choices=["auto", "raw", "qwen_chatml", "azr_r1"],
                   help="auto = look up in model_zoo.NATIVE_TEMPLATE")
    p.add_argument("--dataset", default="harmbench200_alpaca200")
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--output-dir", default="results/full_residuals")
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    dump_for_model(args.model, args.short, args.template_mode,
                   args.dataset, args.n_per_class, dtype, args.output_dir)


if __name__ == "__main__":
    main()
