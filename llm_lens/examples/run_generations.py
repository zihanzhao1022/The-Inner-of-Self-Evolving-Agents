#!/usr/bin/env python3
"""
Generate text outputs (incl. chain-of-thought) from each model in the
quartet/quintet for behavioural refusal evaluation.

Why: the rest of the pipeline only captures activations (residual stream,
refusal direction, probe accuracy). For real refusal-behaviour analysis we
need actual `model.generate()` outputs, then can pipe to a classifier or
LLM-judge to score refusal rate / compliance / over-refusal.

Also captures **chain-of-thought** content. AZR's reported "uh-oh moment"
lives in CoT (per the paper's qualitative example), so:
  - max_new_tokens defaults to 512 — long enough for full reasoning chains
  - record stores the raw generation text verbatim (no stripping of
    <think>...</think> tags or "Step 1: ..." structure)
  - regex-based cot_pattern_detected boolean flagged when the generation
    contains common CoT markers (DeepSeek-R1 <think>, "Let's think step
    by step", numbered steps, etc.)

Pipeline per model:
    1. Load model on GPU
    2. For each prompt:
       - Wrap in QWEN_MIN_CHAT_TEMPLATE
       - model.generate(max_new_tokens=256, do_sample=False)
       - Decode + strip the templated prefix
       - Atomic append one JSONL line to results/generations_<TS>/<model>/generations.jsonl
    3. Free GPU

Atomic per-prompt flush (one JSONL line per prompt) so an interrupt at
prompt N still leaves prompts 0..N-1 intact (matches the feedback_save_early
memory: lost data once already).

Default prompt mix (150 prompts):
    100 harmful  = cm_binary harmful (interleaved sexual/hate/crime, ~33 each)
    50 harmless = cm_binary harmless (interleaved base/legal/health, ~17 each)

Usage:
    # 3B all 4 models
    python -m llm_lens.examples.run_generations --model-set 3B
    # 7B all 5 models
    python -m llm_lens.examples.run_generations --model-set 7B --dtype bfloat16
    # specific model only
    python -m llm_lens.examples.run_generations --model-set 7B --targets AZR-Base-7B AZR-Coder-7B
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.datasets import load_prompts
from llm_lens.extractor import QWEN_MIN_CHAT_TEMPLATE, apply_qwen_min_chat_template
from llm_lens.model_zoo import (
    MODEL_SETS, get_models, parse_dtype,
    DEFAULT_MODEL_SET, DEFAULT_DTYPE, DTYPE_CHOICES,
)


def _build_prompt_set(n_harmful: int = 100, n_harmless: int = 50,
                       seed: int = 0) -> list[dict]:
    """Build the (prompt, source_class, original_idx) prompt list using
    cm_binary loader — same data the refusal-direction extraction uses, so
    generations and activations are aligned per prompt index.
    """
    cm = load_prompts("cm_binary", max_per_class=max(n_harmful, n_harmless))
    harmful = cm["harmful"][:n_harmful]
    harmless = cm["harmless"][:n_harmless]
    items = []
    for i, p in enumerate(harmful):
        items.append({"class": "harmful",  "prompt_idx": i, "prompt": p})
    for i, p in enumerate(harmless):
        items.append({"class": "harmless", "prompt_idx": i, "prompt": p})
    return items


import re

# Patterns that indicate explicit chain-of-thought structure in a generation.
# Detection is heuristic; scoring how "reasoning-like" the output is would
# need an LLM judge. We just record whether any of these markers appear, so
# downstream analysis can filter on cot_pattern_detected=True.
_COT_PATTERNS = [
    re.compile(r"<think>.*?</think>", re.DOTALL),                 # DeepSeek-R1-style
    re.compile(r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>", re.DOTALL),
    re.compile(r"Let'?s think step.{0,5}step", re.IGNORECASE),
    re.compile(r"step\s*1[:.\s]", re.IGNORECASE),                  # numbered steps
    re.compile(r"first[,\s].*?second[,\s].*?third", re.IGNORECASE | re.DOTALL),
    re.compile(r"reasoning[:\s]", re.IGNORECASE),
    re.compile(r"chain[ -]of[ -]thought", re.IGNORECASE),
]


def detect_cot_structure(text: str) -> dict:
    """Return a dict of detected CoT markers + a coarse boolean."""
    matches = []
    for pat in _COT_PATTERNS:
        if pat.search(text):
            matches.append(pat.pattern[:40])
    return {
        "cot_pattern_detected": bool(matches),
        "cot_patterns_matched": matches,
        "n_lines": text.count("\n") + 1,
        "n_chars": len(text),
    }


def _atomic_append_jsonl(path: str, record: dict) -> None:
    """Append one JSONL line and fsync. Per-prompt durability."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def generate_for_model(
    full: str,
    short: str,
    items: list[dict],
    out_dir: str,
    max_new_tokens: int,
    dtype: torch.dtype,
) -> dict:
    """Load model, generate for each prompt, save atomically. Resume-safe:
    skips items whose prompt_idx already appears in the output file."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'=' * 60}\n  GENERATIONS — {short}\n{'=' * 60}", flush=True)
    out_path = os.path.join(out_dir, short, "generations.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Resume support — skip already-done (class, prompt_idx) pairs.
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    done.add((r["class"], r["prompt_idx"]))
                except Exception:
                    pass
        if done:
            print(f"  resume: {len(done)} prompts already generated, skipping", flush=True)

    pending = [it for it in items if (it["class"], it["prompt_idx"]) not in done]
    if not pending:
        print(f"  all {len(items)} prompts already done, skipping model load", flush=True)
        return {"n_done": len(items), "n_skipped": 0, "n_new": 0,
                "elapsed_sec": 0.0, "out_path": out_path}

    tok = AutoTokenizer.from_pretrained(full, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    print(f"  loading {full} on cuda ({dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        full, dtype=dtype, device_map="cuda", trust_remote_code=True)
    model.eval()
    print(f"  model loaded; generating {len(pending)} prompts (max_new_tokens={max_new_tokens})...",
          flush=True)

    t0 = time.time()
    for j, it in enumerate(pending):
        wrapped = apply_qwen_min_chat_template(it["prompt"])
        inputs = tok(wrapped, return_tensors="pt",
                     add_special_tokens=False).to("cuda")
        prompt_len = int(inputs["input_ids"].shape[1])
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out_ids[0, prompt_len:]
        generation = tok.decode(gen_ids, skip_special_tokens=False)
        # Determine stop reason — eos hit or hit max_new_tokens
        n_new = int(gen_ids.shape[0])
        stop_reason = "eos" if (n_new < max_new_tokens or
                                gen_ids[-1].item() == tok.eos_token_id) else "max_tokens"

        cot_info = detect_cot_structure(generation)
        record = {
            "model":         short,
            "class":         it["class"],
            "prompt_idx":    it["prompt_idx"],
            "prompt":        it["prompt"],
            "generation":    generation,
            "n_new_tokens":  n_new,
            "stop_reason":   stop_reason,
            "wrapping":      "QWEN_MIN_CHAT_TEMPLATE",
            **cot_info,
        }
        _atomic_append_jsonl(out_path, record)

        if (j + 1) % 25 == 0 or (j + 1) == len(pending):
            elapsed = time.time() - t0
            rate = (j + 1) / max(elapsed, 1e-6)
            eta = (len(pending) - j - 1) / max(rate, 1e-6)
            print(f"    [{j + 1}/{len(pending)}]  {rate:.2f} prompts/sec  eta={eta:.0f}s",
                  flush=True)

    elapsed = time.time() - t0
    print(f"  done — generated {len(pending)} new prompts in {elapsed:.0f}s "
          f"({len(pending) / elapsed:.2f} /s)", flush=True)

    # Free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"n_done": len(items), "n_skipped": len(done), "n_new": len(pending),
            "elapsed_sec": elapsed, "out_path": out_path}


def main():
    p = argparse.ArgumentParser(
        description="Generate text outputs for behavioural refusal evaluation")
    p.add_argument("--model-set", default=DEFAULT_MODEL_SET,
                   choices=list(MODEL_SETS))
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPE_CHOICES))
    p.add_argument("--n-harmful",  type=int, default=100,
                   help="Number of harmful prompts (cm_binary harmful tag)")
    p.add_argument("--n-harmless", type=int, default=50,
                   help="Number of harmless prompts (cm_binary harmless tag)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Long enough to capture chain-of-thought reasoning "
                        "(AZR's 'uh-oh moment' lives in CoT). 512 default "
                        "covers most reasoning chains; bump to 1024 if you "
                        "see truncation in stop_reason='max_tokens'.")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of model short labels to run (default: all)")
    p.add_argument("--results-root", default="results")
    p.add_argument("--output-suffix", default=None,
                   help="Override timestamp suffix (default: now)")
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    out_ts = args.output_suffix or datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(args.results_root, f"generations_{out_ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[run_generations] output → {out_dir}")
    print(f"  model_set={args.model_set}  dtype={args.dtype}  "
          f"prompts: {args.n_harmful} harmful + {args.n_harmless} harmless")

    items = _build_prompt_set(args.n_harmful, args.n_harmless)
    print(f"  total {len(items)} prompts")

    targets = args.targets or [s for _, s in get_models(args.model_set)]
    summaries = {}
    for full, short in get_models(args.model_set):
        if short not in targets:
            continue
        try:
            summaries[short] = generate_for_model(
                full=full, short=short,
                items=items, out_dir=out_dir,
                max_new_tokens=args.max_new_tokens, dtype=dtype)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] {short} failed: {e.__class__.__name__}: {e}",
                  flush=True)
            summaries[short] = {"error": str(e), "type": e.__class__.__name__}

    # Write run meta + per-model summary
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":         out_ts,
            "model_set":         args.model_set,
            "dtype":             args.dtype,
            "n_harmful":         args.n_harmful,
            "n_harmless":        args.n_harmless,
            "max_new_tokens":    args.max_new_tokens,
            "wrapping":          "QWEN_MIN_CHAT_TEMPLATE (forced minimal Qwen ChatML)",
            "decoding":          f"do_sample=False (greedy)",
            "summaries":         summaries,
            "written_at":        datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n[run_generations] complete. summary:")
    for short, s in summaries.items():
        if "error" in s:
            print(f"  {short:<22} ERROR: {s['error']}")
        else:
            print(f"  {short:<22} {s['n_new']} new + {s['n_skipped']} resumed = "
                  f"{s['n_done']} total ({s['elapsed_sec']:.0f}s)")
    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
