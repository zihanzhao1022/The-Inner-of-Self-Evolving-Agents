#!/usr/bin/env python3
"""
Subprocess-per-axis-pair weight-diff coordinator. Workaround for the
HF from_pretrained transient memory peak that OOMed v1-v5 in-process
7B attempts on Windows 32 GB.

For each named axis pair (RLHF / domain / self_evolving_* / cross_AZR):
  1. Spawn `_weight_diff_single_pair.py` as a fresh Python subprocess
     with the pair's two full HF model IDs.
  2. The worker loads both, computes top + per-layer cosines, saves
     per-pair `pair_<a>__<b>.npz` + `top_cosines_<a>__<b>.json` to
     out_dir.
  3. Worker exits → all RAM released by OS.
  4. Repeat for next pair.

After all pairs done, aggregate into the standard outputs:
  weight_cosines.npz        # cos/diff matrices for all pairs
  top_cosines.json          # readable summary
  weight_cosines_heatmap.png
  run_meta.json

Usage:
    python -m llm_lens.examples.run_weight_diff_subprocess --model-set 7B
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.model_zoo import get_models, get_axis_pairs


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
COMP_ORDER = list(LAYER_COMPONENT_SHORT.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-set", default="7B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-gpu", action="store_true",
                    help="Disable GPU cosine acceleration (use CPU fp64)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip an axis pair whose pair_<...>.npz already exists")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = Path(args.out_dir or f"results/weight_diff_{ts}_subprocess_{args.model_set}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[coordinator] out_dir = {out_dir}")

    models = get_models(args.model_set)
    short_to_full = {short: full for full, short in models}
    axis_pairs_named = get_axis_pairs(args.model_set)
    if not axis_pairs_named:
        raise ValueError(f"No axis pairs for {args.model_set}")

    print(f"[coordinator] axis pairs ({len(axis_pairs_named)}):")
    for axis, (sa, sb) in axis_pairs_named.items():
        print(f"  {axis:<28} {sa:<22} ↔ {sb}")

    # Spawn one worker per pair
    pair_results = {}
    for axis, (a_short, b_short) in axis_pairs_named.items():
        a_full = short_to_full[a_short]
        b_full = short_to_full[b_short]
        pair_tag = f"{a_short}__{b_short}"
        npz_path = out_dir / f"pair_{pair_tag}.npz"

        if args.skip_existing and npz_path.exists():
            print(f"\n[coordinator] {axis}: pair_{pair_tag}.npz exists, skipping")
            continue

        print(f"\n[coordinator] ====== axis={axis} =====================")
        print(f"[coordinator] spawning worker for {pair_tag}")

        cmd = [
            sys.executable, "-m", "llm_lens.examples._weight_diff_single_pair",
            "--a-full", a_full,
            "--a-short", a_short,
            "--b-full", b_full,
            "--b-short", b_short,
            "--out-dir", str(out_dir),
            "--dtype", args.dtype,
        ]
        if args.no_gpu:
            cmd.append("--no-gpu")

        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"[coordinator] ⚠ worker {pair_tag} returned exit code {rc}")
            pair_results[axis] = {"status": "error", "exit_code": rc, "pair_tag": pair_tag}
            continue
        if not npz_path.exists():
            print(f"[coordinator] ⚠ worker {pair_tag} finished but no npz produced")
            pair_results[axis] = {"status": "missing", "pair_tag": pair_tag}
            continue
        pair_results[axis] = {"status": "ok", "pair_tag": pair_tag}
        print(f"[coordinator] {axis}: ✓ saved {npz_path.name}")

    # Aggregate
    print("\n[coordinator] aggregating per-pair results...")
    aggregated_cos = {}
    aggregated_diff = {}
    aggregated_top = {}
    num_layers = None
    tie = None

    for axis, (a_short, b_short) in axis_pairs_named.items():
        pair_tag = f"{a_short}__{b_short}"
        npz_path = out_dir / f"pair_{pair_tag}.npz"
        json_path = out_dir / f"top_cosines_{pair_tag}.json"
        if not npz_path.exists() or not json_path.exists():
            print(f"  {axis} ({pair_tag}): missing — skipping in aggregate")
            continue
        npz = np.load(npz_path, allow_pickle=True)
        aggregated_cos[pair_tag] = npz["cos"]
        aggregated_diff[pair_tag] = npz["diff"]
        if num_layers is None:
            num_layers = int(npz["num_layers"])
            tie = bool(npz["tie"])
        with open(json_path) as fh:
            top = json.load(fh)
        aggregated_top[pair_tag] = {k: v["cosine"] for k, v in top["top_cosines"].items()}

    # Save aggregated outputs
    np.savez(out_dir / "weight_cosines.npz",
             **{f"cos_{k}": v for k, v in aggregated_cos.items()},
             **{f"diff_{k}": v for k, v in aggregated_diff.items()})
    with open(out_dir / "top_cosines.json", "w") as fh:
        json.dump({
            "top_cosines": aggregated_top,
            "tie_word_embeddings": tie,
            "components_order": COMP_ORDER,
            "axis_pairs": {axis: list(pair) for axis, pair in axis_pairs_named.items()},
        }, fh, indent=2)
    with open(out_dir / "run_meta.json", "w") as fh:
        json.dump({
            "timestamp": ts,
            "model_set": args.model_set,
            "dtype": args.dtype,
            "num_layers": num_layers,
            "tie": tie,
            "models": {short: full for full, short in models},
            "axis_pairs": {axis: list(pair) for axis, pair in axis_pairs_named.items()},
            "pair_results": pair_results,
            "method": "subprocess-per-axis-pair (workaround for 7B in-process OOM)",
            "written_at": datetime.now().isoformat(),
        }, fh, indent=2)

    # Heatmap
    try:
        import matplotlib.pyplot as plt
        pair_tags = list(aggregated_cos.keys())
        n_pairs = len(pair_tags)
        if n_pairs > 0:
            fig, axes = plt.subplots(1, n_pairs, figsize=(7 * n_pairs + 1, 10))
            if n_pairs == 1:
                axes = [axes]
            for ax, pair_tag in zip(axes, pair_tags):
                h = aggregated_cos[pair_tag]
                im = ax.imshow(h, aspect="auto", vmin=0.0, vmax=1.0, cmap="RdYlGn")
                ax.set_xticks(range(len(COMP_ORDER)))
                ax.set_xticklabels([LAYER_COMPONENT_SHORT[c] for c in COMP_ORDER],
                                   rotation=45, ha="right", fontsize=9)
                ax.set_ylabel("layer index")
                ax.set_title(pair_tag.replace("__", "\nvs "), fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.suptitle(f"{args.model_set} weight cosines (fp64 per-matrix, subprocess workaround)",
                         fontsize=13, fontweight="bold", y=1.02)
            fig.tight_layout()
            fig.savefig(out_dir / "weight_cosines_heatmap.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            print(f"[coordinator] saved heatmap: {out_dir}/weight_cosines_heatmap.png")
    except Exception as e:
        print(f"[coordinator] heatmap failed: {e}")

    # Summary
    print("\n[coordinator] === FINAL ===")
    print(f"axis pairs completed: {len(aggregated_cos)} / {len(axis_pairs_named)}")
    print("\ntop cosines by pair:")
    for pair_tag, top in aggregated_top.items():
        items = ", ".join(f"{k}={v:.6f}" for k, v in top.items())
        print(f"  {pair_tag}:  {items}")
    print(f"\noutput: {out_dir}/")


if __name__ == "__main__":
    main()
