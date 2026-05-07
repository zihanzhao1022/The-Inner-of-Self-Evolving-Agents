#!/usr/bin/env python3
"""
Steering transfer experiment.

Tests whether a class-mean-difference steering vector trained on Qwen2.5-3B
(base) at a chosen layer can be Procrustes-rotated into AZR-Coder-3B's
representation frame and still produce its intended effect inside AZR.

Pipeline (per (target_model, inject_layer, strength)):
  for each test prompt p:
    forward p through target_model with steering hook on inject_layer
    capture activation at L35
    score = cos(activation, target_centroid_dst) − cos(activation, ref_centroid_dst)
  report mean score

Conditions per target model:
  C0 baseline           — no injection
  C1 native             — inject the model's *own* hate vector
  C2 raw transfer       — inject base's hate vector unrotated (only meaningful for non-base targets)
  C3 procrustes transfer — inject base's hate vector rotated via Procrustes(base→target)

Output:
  results/steering_transfer_<TS>/results.json
  results/steering_transfer_<TS>/steering_transfer_<layer>.png  (one per inject layer)
  results/steering_transfer_<TS>/steering_transfer_summary.png  (combined)

Usage:
    python -m llm_lens.examples.run_steering_transfer --ts 20260506-0000
    python -m llm_lens.examples.run_steering_transfer --ts 20260506-0000 \
        --layers 25 33 35 --strengths -3 -1 0 1 3 --n-prompts 100
"""

from __future__ import annotations

import os, sys, json, argparse, time
from datetime import datetime
import numpy as np
import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.extractor import ActivationExtractor
from llm_lens.report_io import find_artifacts_for_report, load_class_centroids
from llm_lens.steering import (
    compute_steering_vector,
    transfer_vector,
    forward_capture_last_layer,
    transfer_score,
)
from llm_lens.datasets import _IBM_DIR
from llm_lens.model_zoo import (
    MODEL_SETS, get_models, parse_dtype,
    DEFAULT_MODEL_SET, DEFAULT_DTYPE, DTYPE_CHOICES,
)


def _models_for_set(model_set: str):
    """Return [(full, short, report_file), ...] for the given size class.

    The report filename pattern in run_experiment.phase1_single_model is
    'report_<safe_name>.json' where safe_name = full.replace('/', '_').
    """
    out = []
    for full, short in get_models(model_set):
        report_fname = f"report_{full.replace('/', '_')}.json"
        out.append((full, short, report_fname))
    return out


SOURCE_MODEL_BY_SET = {            # base = first entry of each set
    s: get_models(s)[0][1] for s in MODEL_SETS
}

DEFAULT_TARGET_TAG = "hate_speech"
DEFAULT_REF_TAG    = "base"
DEFAULT_PROMPT_TAG = "base"   # eval on neutral prompts → see if steering pulls them to hate
# Layer defaults are 3B-specific (36 layers; inject in the last few + capture at L35).
# For 7B (28 layers) or 14B (48 layers) you must override via --layers and
# --capture-layer; warning printed at runtime if values look out of range.
DEFAULT_LAYERS     = (25, 33, 35)
# Strengths are multiples of the target model's NATIVE class-mean-difference
# magnitude at the inject layer. So strength=1 means "add one natural
# inter-class distance", which is what the model itself uses to separate
# classes.  ±2 covers a strong-but-not-destructive sweep.
DEFAULT_STRENGTHS  = (-2.0, -1.0, 0.0, 1.0, 2.0)
CAPTURE_LAYER      = 35


# ── Eval prompts ────────────────────────────────────────────────────────────

def load_eval_prompts(tag: str, n: int, split: str = "test") -> list[str]:
    """Pull `n` prompts of `tag` from condition_multiple's `split`."""
    path = os.path.join(_IBM_DIR, "condition_multiple.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data[split]
    out = []
    for it in items:
        if tag in it and it[tag]:
            out.append(it[tag].strip())
            if len(out) >= n:
                break
    return out


# ── Per-target evaluation ───────────────────────────────────────────────────

def evaluate_one_target(
    target_full: str,
    target_short: str,
    target_centroids: dict,
    src_centroids: dict,         # base's centroids (for vector + rotation)
    eval_prompts: list[str],
    layers: list[int],
    strengths: list[float],
    target_idx: int,
    ref_idx: int,
    capture_layer: int,
    source_model_short: str,
    dtype: torch.dtype = torch.float32,
):
    """Load target model, run all (layer × strength × condition) combinations.

    Returns: dict with shape  results[layer][strength][condition] = list of float scores.
    """
    print(f"\n{'=' * 60}\n  TARGET = {target_short}  ({target_full})\n{'=' * 60}")
    ext = ActivationExtractor(target_full, dtype=dtype, capture_heads=False)
    capture_module = ext._layers[capture_layer]
    target_cents_full = target_centroids["class_centroids"]   # (L, C, D)

    out: dict = {}
    for L in layers:
        out[L] = {}
        inject_module = ext._layers[L]

        # Pre-compute the candidate vectors for this layer.
        # All three are scaled to the TARGET MODEL's native class-difference
        # magnitude so that strength=1 = "one natural inter-class distance"
        # for any condition. This isolates direction differences from
        # incidental magnitude differences across models.
        vec_native_unscaled = compute_steering_vector(
            target_cents_full, L, target_idx, ref_idx)
        native_norm = float(np.linalg.norm(vec_native_unscaled))

        vec_native = vec_native_unscaled            # already at native magnitude

        vec_src = compute_steering_vector(
            src_centroids["class_centroids"], L, target_idx, ref_idx,
            target_norm=native_norm)
        vec_raw  = vec_src                                                    # unrotated, native-magnitude
        vec_proc_unscaled = transfer_vector(
            vec_src,
            src_centroids["class_centroids"][L],
            target_cents_full[L],
        )
        # Re-scale (Procrustes preserves magnitude, so this is mostly a no-op,
        # but defensive for numerical drift)
        vec_proc = vec_proc_unscaled / (np.linalg.norm(vec_proc_unscaled) + 1e-10) * native_norm

        # Diagnostic line so we can spot weird vector geometry from logs
        print(f"  L{L}  native_norm={native_norm:.2f}  "
              f"cos(raw, native)={float(vec_raw @ vec_native) / (native_norm**2 + 1e-10):.3f}  "
              f"cos(proc, native)={float(vec_proc @ vec_native) / (native_norm**2 + 1e-10):.3f}")

        def _evaluate(condition_name: str, vec: np.ndarray | None, strength: float):
            t0 = time.time()
            acts = forward_capture_last_layer(
                model=ext.model,
                tokenizer=ext.tokenizer,
                prompts=eval_prompts,
                capture_layer_module=capture_module,
                inject_layer_module=inject_module if vec is not None else None,
                steering_vec=vec,
                strength=strength,
            )
            scores = transfer_score(acts, target_cents_full[capture_layer],
                                    target_idx, ref_idx)
            dt = time.time() - t0
            print(f"    L{L:>2}  s={strength:+.1f}  {condition_name:>14s}  "
                  f"mean={scores.mean():+.4f}  std={scores.std():.4f}  ({dt:.1f}s)")
            return scores.tolist()

        for s in strengths:
            cell: dict = {}
            cell["baseline"] = _evaluate("baseline", None, 0.0) if s == 0.0 else None
            if s != 0.0:
                cell["native"] = _evaluate("native", vec_native, s)
                if target_short != source_model_short:
                    cell["raw"]      = _evaluate("raw_xfer",  vec_raw,  s)
                    cell["procrust"] = _evaluate("procrust",  vec_proc, s)
            out[L][float(s)] = cell

    # Free GPU
    del ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Cross-model steering transfer experiment")
    p.add_argument("--ts", required=True,
                   help="Timestamp suffix of the existing phase-1 run "
                        "(reports + class_centroids must already be on disk)")
    p.add_argument("--results-root", default="results",
                   help="Where the phase-1 outputs live and where to write our results.")
    p.add_argument("--layers", nargs="+", type=int, default=list(DEFAULT_LAYERS))
    p.add_argument("--strengths", nargs="+", type=float, default=list(DEFAULT_STRENGTHS))
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--target-tag", default=DEFAULT_TARGET_TAG,
                   help="Class to steer toward")
    p.add_argument("--ref-tag", default=DEFAULT_REF_TAG,
                   help="Class to steer away from (DiM reference)")
    p.add_argument("--prompt-tag", default=DEFAULT_PROMPT_TAG,
                   help="Which class of test prompts to evaluate on")
    p.add_argument("--capture-layer", type=int, default=CAPTURE_LAYER)
    p.add_argument("--output-suffix", default=None,
                   help="Override the auto-generated output dir suffix")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of model short labels (default: all 3)")
    p.add_argument("--model-set", default=DEFAULT_MODEL_SET, choices=list(MODEL_SETS),
                   help="Which (base/instruct/self-evolved) trio to use. "
                        "Default '3B'. For 7B/14B you usually also want to "
                        "override --layers / --capture-layer (3B is 36 layers, "
                        "7B is 28, 14B is 48).")
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPE_CHOICES),
                   help="Model loading dtype.")
    args = p.parse_args()

    MODELS = _models_for_set(args.model_set)
    SOURCE_MODEL = SOURCE_MODEL_BY_SET[args.model_set]
    dtype = parse_dtype(args.dtype)

    if 0.0 not in args.strengths:
        args.strengths = [0.0] + list(args.strengths)
    args.strengths = sorted(set(args.strengths))

    out_ts = args.output_suffix or datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(args.results_root, f"steering_transfer_{out_ts}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n[run_steering_transfer] output → {out_dir}")
    print(f"  layers={args.layers}, strengths={args.strengths}, "
          f"n_prompts={args.n_prompts}, target='{args.target_tag}', ref='{args.ref_tag}'")

    # ── Load all centroid npz files ─────────────────────────────────────────
    centroids_by_short = {}
    for full, short, fname in MODELS:
        rep_path = os.path.join(args.results_root, short, args.ts, fname)
        art = find_artifacts_for_report(rep_path)
        if "centroids" not in art:
            raise FileNotFoundError(
                f"No class_centroids npz under {os.path.dirname(rep_path)}")
        centroids_by_short[short] = load_class_centroids(art["centroids"])

    src = centroids_by_short[SOURCE_MODEL]
    tags_order = list(src["tags_order"])
    if args.target_tag not in tags_order or args.ref_tag not in tags_order:
        raise ValueError(f"target/ref tag not in {tags_order}")
    target_idx = tags_order.index(args.target_tag)
    ref_idx    = tags_order.index(args.ref_tag)

    # Sanity: every model must share the same tag order (they should — same dataset)
    for short, c in centroids_by_short.items():
        assert list(c["tags_order"]) == tags_order, f"tag order mismatch in {short}"

    # ── Load eval prompts ───────────────────────────────────────────────────
    eval_prompts = load_eval_prompts(args.prompt_tag, args.n_prompts, split="test")
    if len(eval_prompts) < args.n_prompts:
        print(f"  [WARN] only got {len(eval_prompts)}/{args.n_prompts} eval prompts")
    print(f"  eval prompts: {len(eval_prompts)} from condition_multiple.test[{args.prompt_tag}]")

    # ── Run experiment per target model ────────────────────────────────────
    targets = args.targets or [s for _, s, _ in MODELS]
    results: dict = {}
    for full, short, _ in MODELS:
        if short not in targets:
            continue
        results[short] = evaluate_one_target(
            target_full=full,
            target_short=short,
            target_centroids=centroids_by_short[short],
            src_centroids=src,
            eval_prompts=eval_prompts,
            layers=args.layers,
            strengths=args.strengths,
            target_idx=target_idx,
            ref_idx=ref_idx,
            capture_layer=args.capture_layer,
            source_model_short=SOURCE_MODEL,
            dtype=dtype,
        )

    # ── Save raw results ────────────────────────────────────────────────────
    payload = {
        "timestamp": out_ts,
        "phase1_ts": args.ts,
        "target_tag": args.target_tag,
        "ref_tag": args.ref_tag,
        "prompt_tag": args.prompt_tag,
        "tags_order": tags_order,
        "layers": args.layers,
        "strengths": args.strengths,
        "capture_layer": args.capture_layer,
        "n_eval_prompts": len(eval_prompts),
        "results": results,
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON → {json_path}")

    # ── Visualize ───────────────────────────────────────────────────────────
    try:
        from llm_lens.examples.plot_steering_transfer import plot_all
        plot_all(payload, out_dir)
    except Exception as e:
        print(f"  [WARN] plotting failed: {e}")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
