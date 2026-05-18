#!/usr/bin/env python3
"""
Dump residual stream of BOTH the prompt and the generated continuation.

For each prompt (200 harmful HarmBench + 200 harmless Alpaca):
  1. model.generate(...) with output_hidden_states=True, max_new_tokens=1024
  2. Capture per-layer residuals at every position (prompt + generated)
  3. Decode the generated text + classify refusal kind
  4. Write a per-prompt NPZ (and incrementally save behavior JSON)

Output layout:
  results/full_gen_residuals_<TS>/<SHORT>/
    meta.json                          # model + dataset + dtype info
    behavior.json                      # per-prompt classification (refuse/comply)
    harmful/<idx_3digit>.npz           # 200 npz files (one per prompt)
    harmless/<idx_3digit>.npz          # 200 npz files
    generations/<tag>_<idx_3digit>.txt # decoded generated text

Each NPZ contains:
  residuals: (prompt_len + n_generated, n_layer, hidden_dim) — chosen dtype
  token_ids: (prompt_len + n_generated,) int32
  prompt_len: int (boundary between prompt and generated)
  n_generated: int (might be < max_new_tokens if EOS hit)

This dump is the GROUND-TRUTH raw data: every layer × every token × every
prompt × generated continuation, in the same forward pass that produces
the model's actual behaviour. All downstream analyses (refusal direction,
probe, within-class spread, behavior direction, generation-time activations
at decision point, etc.) should be derivable.

Storage estimate per model (fp16, max_new=1024):
  3B (36 layers, 2048 dim): ~60 GB
  7B (28 layers, 3584 dim): ~85 GB

Per-prompt files keep individual file sizes well under HF's 50 GB
single-file limit and enable resumable / parallel uploads.

Usage:
    python -m llm_lens.examples.dump_generation_residuals \
        --model Qwen/Qwen2.5-3B \
        --short Qwen2.5-3B \
        --template-mode auto \
        --output-root results/full_gen_residuals_20260518-0020 \
        --dataset harmbench200_alpaca200 \
        --n-per-class 200 \
        --max-new-tokens 1024 \
        --dtype float32 \
        --storage-dtype match
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import re
from datetime import datetime

import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.datasets import load_prompts
from llm_lens.model_zoo import parse_dtype, get_native_template
from llm_lens.templates import wrap_prompt
from llm_lens.examples.run_paired_analysis_v2 import (
    HARD_REFUSAL_RE, SOFT_REFUSAL_RE,
)


_LEAD_NOISE = re.compile(r"^[\s\.\,\;\:\!\?\)\]\}\"'\-\/]+")


def classify_refusal(gen: str) -> tuple[bool, str]:
    g = (gen or "").lstrip()
    gs = _LEAD_NOISE.sub("", g)
    if not gs:
        return False, "empty"
    if HARD_REFUSAL_RE.match(gs[:120]):
        return True, "hard"
    if SOFT_REFUSAL_RE.match(gs[:220]):
        return True, "soft"
    return False, "none"


def assemble_full_residuals(generate_output, n_layer: int) -> np.ndarray:
    """Stack hidden_states from generate() into (seq_len, n_layer, hidden_dim).

    `generate_output.hidden_states` is a tuple of length max_new_tokens, where
      hidden_states[0] is the prompt forward call: per-layer tensor of shape
                          (1, prompt_len, hidden_dim)
      hidden_states[k] for k >= 1 is the k-th generation step's forward call:
                          per-layer tensor of shape (1, 1, hidden_dim)

    Each step's per-layer entry is a tuple of (n_layer + 1) tensors (the +1
    is the embedding layer). We skip the embedding and keep the n_layer
    transformer-block outputs.
    """
    steps = generate_output.hidden_states  # tuple of length n_steps
    # Step 0: prompt
    prompt_layers = []
    for layer_idx in range(n_layer):
        # +1 to skip embedding layer
        t = steps[0][layer_idx + 1][0].cpu()  # (prompt_len, dim)
        prompt_layers.append(t)
    prompt_part = torch.stack(prompt_layers, dim=1)  # (prompt_len, n_layer, dim)

    if len(steps) > 1:
        gen_layers = []
        for layer_idx in range(n_layer):
            chunks = [steps[k][layer_idx + 1][0] for k in range(1, len(steps))]
            t = torch.cat(chunks, dim=0).cpu()  # (n_gen, dim)
            gen_layers.append(t)
        gen_part = torch.stack(gen_layers, dim=1)  # (n_gen, n_layer, dim)
        full = torch.cat([prompt_part, gen_part], dim=0)
    else:
        full = prompt_part

    return full  # torch tensor, fp32 (or whatever inference dtype)


def dump_for_model(model_name: str, short: str, template_mode: str,
                   dataset: str, n_per_class: int, max_new_tokens: int,
                   dtype: torch.dtype, storage_dtype: torch.dtype,
                   output_root: str) -> dict:
    """Returns the behavior summary dict (also written to disk)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = os.path.join(output_root, short)
    os.makedirs(os.path.join(out_dir, "harmful"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "harmless"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "generations"), exist_ok=True)

    prompts = load_prompts(dataset, max_per_class=n_per_class)
    harmful  = prompts["harmful"]
    harmless = prompts["harmless"]

    if template_mode == "auto":
        template_mode = get_native_template(short)

    print(f"[gen-dump] model={model_name} short={short} template={template_mode}")
    print(f"[gen-dump] n_h={len(harmful)} n_s={len(harmless)} max_new={max_new_tokens} "
          f"dtype={dtype} storage_dtype={storage_dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype).to("cuda")
    model.eval()
    n_layer    = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"[gen-dump] loaded: n_layer={n_layer} hidden_dim={hidden_dim}")

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id

    _np_storage = {
        torch.float32: np.float32,
        torch.float16: np.float16,
        torch.bfloat16: np.float32,
    }[storage_dtype]

    behavior: dict = {
        "model":           short,
        "model_full_name": model_name,
        "template_mode":   template_mode,
        "dataset":         dataset,
        "max_new_tokens":  max_new_tokens,
        "dtype_inference": str(dtype),
        "storage_dtype":   {torch.float32: "float32", torch.float16: "float16",
                            torch.bfloat16: "float32 (promoted from bf16)"}[storage_dtype],
        "n_layer":         int(n_layer),
        "hidden_dim":      int(hidden_dim),
        "started_at":      datetime.now().isoformat(),
        "harmful_prompts": [],
        "harmless_prompts": [],
    }

    behavior_path = os.path.join(out_dir, "behavior.json")
    meta_path     = os.path.join(out_dir, "meta.json")

    def _process(tag: str, pool: list[str], dest: list[dict]) -> None:
        t0_tag = time.time()
        for i, prompt_text in enumerate(pool):
            wrapped = wrap_prompt(prompt_text, template_mode,
                                  tokenizer=tokenizer)
            inputs = tokenizer(wrapped, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
            prompt_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,                # greedy, matches HarmBench eval
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                    pad_token_id=pad_id,
                )

            # Decode generated text only (skip prompt tokens).
            gen_ids = out.sequences[0, prompt_len:].cpu().numpy().astype(np.int32)
            full_ids = out.sequences[0].cpu().numpy().astype(np.int32)
            n_gen = int(gen_ids.shape[0])
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Residuals as fp16 numpy.
            full_resid = assemble_full_residuals(out, n_layer)  # (seq_len, n_layer, dim) torch
            if storage_dtype == torch.bfloat16:
                arr = full_resid.to(torch.float32).numpy().astype(np.float32)
            else:
                arr = full_resid.to(storage_dtype).numpy().astype(_np_storage)

            idx_str = f"{i:03d}"
            npz_path = os.path.join(out_dir, tag, f"{idx_str}.npz")
            np.savez_compressed(npz_path,
                                residuals=arr,
                                token_ids=full_ids,
                                prompt_len=np.array(prompt_len, dtype=np.int32),
                                n_generated=np.array(n_gen, dtype=np.int32))

            with open(os.path.join(out_dir, "generations",
                                   f"{tag}_{idx_str}.txt"),
                      "w", encoding="utf-8") as f:
                f.write(gen_text)

            refused, kind = classify_refusal(gen_text)
            dest.append({
                "source_idx": i,
                "prompt_len": prompt_len,
                "n_generated": n_gen,
                "refused": refused,
                "kind":    kind,
                "first_120": gen_text[:120],
            })

            # Incremental flush of behavior.json (resilient against crashes).
            if (i + 1) % 10 == 0 or i + 1 == len(pool):
                with open(behavior_path, "w", encoding="utf-8") as f:
                    json.dump(behavior, f, ensure_ascii=False, indent=2)

            # Free intermediate state.
            del out, full_resid, arr
            torch.cuda.empty_cache()

            if (i + 1) % 5 == 0:
                el = time.time() - t0_tag
                rate = el / (i + 1)
                eta = rate * (len(pool) - (i + 1))
                print(f"  [{tag}] {i+1}/{len(pool)}  elapsed={el:.1f}s  "
                      f"rate={rate:.1f}s/prompt  eta={eta/60:.1f}min")

    _process("harmful",  harmful,  behavior["harmful_prompts"])
    _process("harmless", harmless, behavior["harmless_prompts"])

    behavior["finished_at"] = datetime.now().isoformat()
    behavior["harmful_refuse_rate"] = (
        sum(1 for r in behavior["harmful_prompts"] if r["refused"])
        / max(1, len(behavior["harmful_prompts"]))
    )
    behavior["harmless_refuse_rate"] = (
        sum(1 for r in behavior["harmless_prompts"] if r["refused"])
        / max(1, len(behavior["harmless_prompts"]))
    )

    with open(behavior_path, "w", encoding="utf-8") as f:
        json.dump(behavior, f, ensure_ascii=False, indent=2)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in behavior.items()
                   if k not in ("harmful_prompts", "harmless_prompts")},
                  f, ensure_ascii=False, indent=2)

    print(f"[gen-dump] done. refuse rate harmful={behavior['harmful_refuse_rate']:.1%} "
          f"harmless={behavior['harmless_refuse_rate']:.1%}")
    print(f"[gen-dump] output dir: {out_dir}")

    del model, tokenizer
    torch.cuda.empty_cache()

    return behavior


def main():
    p = argparse.ArgumentParser(description="Dump full generation+residual one model")
    p.add_argument("--model", required=True)
    p.add_argument("--short", required=True)
    p.add_argument("--template-mode", default="auto",
                   choices=["auto", "raw", "qwen_chatml", "azr_r1"])
    p.add_argument("--dataset", default="harmbench200_alpaca200")
    p.add_argument("--n-per-class", type=int, default=200)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16", "float16"])
    p.add_argument("--storage-dtype", default="match",
                   choices=["match", "float32", "float16"])
    p.add_argument("--output-root", default="results/full_gen_residuals")
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    storage = dtype if args.storage_dtype == "match" else parse_dtype(args.storage_dtype)
    dump_for_model(args.model, args.short, args.template_mode,
                   args.dataset, args.n_per_class, args.max_new_tokens,
                   dtype, storage, args.output_root)


if __name__ == "__main__":
    main()
