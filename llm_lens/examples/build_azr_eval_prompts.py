#!/usr/bin/env python3
"""
Sample a balanced harmful / harmless prompt pool from cm_binary and
persist it to JSON so all 9 evaluation models read the EXACT same prompts.

Why a separate sampling step:
  - Previous runs used `cm_binary` top-N (deterministic but not random),
    which over-weights the early sub-categories.
  - For the AZR-native-template eval we want a *random* sample for
    objectivity, but it must be reproducible AND identical across the
    9 models so that per-model results are 1-to-1 comparable.
  - Solution: sample once with a fixed seed, write to JSON, then every
    runner reads from that JSON.

Output JSON schema:
    {
        "meta": {
            "seed": int,
            "n_harmful": int, "n_harmless": int,
            "pool": "cm_binary",
            "pool_size_harmful": int, "pool_size_harmless": int,
            "written_at": ISO timestamp,
        },
        "prompts": [
            {"class": "harmful" | "harmless",
             "source_idx": int,      # index in the cm_binary class pool
             "prompt": str},
            ...
        ]
    }

Usage:
    python -m llm_lens.examples.build_azr_eval_prompts \
        --n-harmful 25 --n-harmless 25 --seed 42 \
        --out results/azr_native_eval_<TS>/prompts.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.datasets import load_prompts


def sample(n_harmful: int, n_harmless: int, seed: int) -> tuple[list[dict], dict]:
    """Random sample (without replacement) from cm_binary harmful + harmless.

    Returns (prompts list, pool size dict). `source_idx` is the index in the
    cm_binary class pool so the sample can be reproduced from the dataset
    alone (load cm_binary → pick these indices in this order).
    """
    cm = load_prompts("cm_binary", max_per_class=0)   # 0 ⇒ unlimited
    pool_harmful  = cm["harmful"]
    pool_harmless = cm["harmless"]

    if n_harmful > len(pool_harmful):
        raise ValueError(f"requested {n_harmful} harmful but pool has only "
                         f"{len(pool_harmful)}")
    if n_harmless > len(pool_harmless):
        raise ValueError(f"requested {n_harmless} harmless but pool has only "
                         f"{len(pool_harmless)}")

    rng = np.random.RandomState(seed)
    idx_harmful  = sorted(rng.choice(len(pool_harmful),  n_harmful,  replace=False).tolist())
    idx_harmless = sorted(rng.choice(len(pool_harmless), n_harmless, replace=False).tolist())

    prompts: list[dict] = []
    for i in idx_harmful:
        prompts.append({"class": "harmful",  "source_idx": int(i), "prompt": pool_harmful[i]})
    for i in idx_harmless:
        prompts.append({"class": "harmless", "source_idx": int(i), "prompt": pool_harmless[i]})

    return prompts, {"pool_size_harmful": len(pool_harmful),
                     "pool_size_harmless": len(pool_harmless)}


def main():
    p = argparse.ArgumentParser(description="Sample evaluation prompts")
    p.add_argument("--n-harmful",  type=int, default=25)
    p.add_argument("--n-harmless", type=int, default=25)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        required=True,
                   help="Output JSON path (parents auto-created)")
    args = p.parse_args()

    prompts, pool_sizes = sample(args.n_harmful, args.n_harmless, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "meta": {
            "seed":              args.seed,
            "n_harmful":         args.n_harmful,
            "n_harmless":        args.n_harmless,
            "pool":              "cm_binary",
            **pool_sizes,
            "written_at":        datetime.now().isoformat(),
        },
        "prompts": prompts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(prompts)} prompts to {args.out}")
    print(f"  {args.n_harmful} harmful + {args.n_harmless} harmless, "
          f"seed={args.seed}, pool=cm_binary "
          f"({pool_sizes['pool_size_harmful']} h / "
          f"{pool_sizes['pool_size_harmless']} hl)")


if __name__ == "__main__":
    main()
