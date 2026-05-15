#!/usr/bin/env python3
"""
Build a HarmBench-based evaluation prompts.json for run_generations.

HarmBench (centerforaisafety/HarmBench, behaviors_text_all.csv) is a real
jailbreak benchmark — far more L2-severe than the cm_binary topic-boundary
prompts. The default filter keeps only:
  - FunctionalCategory == "standard"  (no multi-turn / image / context-string)
  - empty ContextString               (self-contained imperatives)
yielding ~200 prompts. We take the FIRST N (CSV order, deterministic).

Schema matches build_azr_eval_prompts.py output, so run_generations.py with
--prompts-file picks it up unchanged. Since HarmBench has no harmless
counterpart, n_harmless = 0 and prompt_idx within the harmful class runs
0..N-1.

Usage:
    python -m llm_lens.examples.build_harmbench_eval_prompts \
        --n-harmful 50 \
        --out results/harmbench_eval_<TS>/prompts.json
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


def sample_slice(n_harmful: int, start_idx: int = 0) -> tuple[list[dict], dict]:
    """Take prompts[start_idx : start_idx + n_harmful] from HarmBench (CSV order).

    Deterministic: same start_idx + n always returns the same slice. `source_idx`
    in each record is the absolute index in the HarmBench pool, so subsequent
    runs (e.g. start_idx=50 then start_idx=100) line up 1-to-1 with this one.
    """
    hb = load_prompts("harmbench", max_per_class=0)
    pool = hb["harmful"]
    end_idx = start_idx + n_harmful
    if end_idx > len(pool):
        raise ValueError(f"requested prompts[{start_idx}:{end_idx}] but HarmBench "
                         f"pool only has {len(pool)} after filtering")
    selected = pool[start_idx:end_idx]
    prompts = [
        {"class": "harmful", "source_idx": start_idx + i, "prompt": p}
        for i, p in enumerate(selected)
    ]
    return prompts, {"pool_size_harmful": len(pool),
                     "start_idx": start_idx, "end_idx_excl": end_idx}


def main():
    p = argparse.ArgumentParser(description="Build HarmBench eval prompts")
    p.add_argument("--n-harmful", type=int, default=50,
                   help="Number of HarmBench prompts to take (deterministic, CSV order).")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Slice offset into HarmBench pool. Use 0 for first 50, "
                        "50 for prompts 50-99, etc. — lets you chain runs without "
                        "duplicating prompts.")
    p.add_argument("--out", required=True,
                   help="Output JSON path (parents auto-created)")
    args = p.parse_args()

    prompts, pool_sizes = sample_slice(args.n_harmful, args.start_idx)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = {
        "meta": {
            "source":            "harmbench",
            "sample_mode":       "slice",
            "filter":            "FunctionalCategory=standard, ContextString=empty",
            "n_harmful":         args.n_harmful,
            "n_harmless":        0,
            **pool_sizes,
            "written_at":        datetime.now().isoformat(),
        },
        "prompts": prompts,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"wrote {len(prompts)} HarmBench prompts to {args.out}")
    print(f"  pool size after filter: {pool_sizes['pool_size_harmful']}")
    print(f"  first prompt: {prompts[0]['prompt']!r}")


if __name__ == "__main__":
    main()
