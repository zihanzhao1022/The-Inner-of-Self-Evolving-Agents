#!/usr/bin/env python3
"""
Build a paired HarmBench + Alpaca prompts file for representation-layer
re-runs.

The representation experiments (phase-1 trajectory, refusal direction,
displacement, Procrustes, paired analysis) were originally run on the
IBM cm_binary dataset. To make them comparable with the HarmBench-200
behavioural eval, we need the same prompt distribution on both sides.
This builder produces a single JSON with:

  - 200 HarmBench harmful prompts (standard split, head order — same
    source_idx range 0..199 as harmbench_eval_20260515-1512/prompts.json,
    so cross-referencing with the behavioural data is 1-to-1)
  - 200 Alpaca harmless prompts (no_input subset, deterministic first-200
    sample — same loader as Arditi paper)

Schema matches build_harmbench_eval_prompts.py / build_azr_eval_prompts.py,
so run_generations.py and the refusal-direction scripts pick it up
unchanged.

Output is saved INSIDE the repo (data/) so it commits with the codebase
and is reproducible across machines.

Usage:
    python -m llm_lens.examples.build_harmbench_paired_prompts \
        --n 200 \
        --out data/harmbench200_alpaca200.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.datasets import load_prompts


def build(n: int) -> tuple[list[dict], dict]:
    hb = load_prompts("harmbench", max_per_class=0)
    al = load_prompts("alpaca",    max_per_class=0)

    hb_pool = hb["harmful"]
    al_pool = al["harmless"]

    if n > len(hb_pool):
        raise ValueError(f"requested n={n} but HarmBench pool only has "
                         f"{len(hb_pool)} after filtering")
    if n > len(al_pool):
        raise ValueError(f"requested n={n} but Alpaca pool only has "
                         f"{len(al_pool)} (no_input filter)")

    harmful = [
        {"class": "harmful",  "source_idx": i, "prompt": hb_pool[i]}
        for i in range(n)
    ]
    harmless = [
        {"class": "harmless", "source_idx": i, "prompt": al_pool[i]}
        for i in range(n)
    ]

    prompts = harmful + harmless
    pool_sizes = {
        "pool_size_harmful":  len(hb_pool),
        "pool_size_harmless": len(al_pool),
    }
    return prompts, pool_sizes


def main():
    p = argparse.ArgumentParser(
        description="Build paired HarmBench + Alpaca prompts (representation layer)"
    )
    p.add_argument("--n", type=int, default=200,
                   help="Number of prompts per side (default 200, paper setting).")
    p.add_argument("--out", required=True,
                   help="Output JSON path (parents auto-created).")
    args = p.parse_args()

    prompts, pool_sizes = build(args.n)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "meta": {
            "source_harmful":  "harmbench",
            "source_harmless": "alpaca",
            "sample_mode":     "head",
            "filter":          ("harmful=FunctionalCategory=standard, "
                                "ContextString=empty; "
                                "harmless=no_input only"),
            "n_harmful":       args.n,
            "n_harmless":      args.n,
            **pool_sizes,
            "written_at":      datetime.now().isoformat(),
        },
        "prompts": prompts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    n_harm  = sum(1 for r in prompts if r["class"] == "harmful")
    n_safe  = sum(1 for r in prompts if r["class"] == "harmless")
    print(f"wrote {len(prompts)} prompts to {args.out}")
    print(f"  harmful (HarmBench):  {n_harm}")
    print(f"  harmless (Alpaca):    {n_safe}")
    print(f"  first harmful:  {prompts[0]['prompt']!r}")
    print(f"  first harmless: {prompts[n_harm]['prompt']!r}")


if __name__ == "__main__":
    main()
