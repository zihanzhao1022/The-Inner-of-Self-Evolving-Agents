#!/usr/bin/env python3
"""
Arditi-style refusal-direction extraction for the (base / Instruct / Coder /
AZR) quartet.

Aligned with https://github.com/andyrdt/refusal_direction:
  * Captures the FULL eoi_toks range (last 5 tokens of the forced minimal
    Qwen ChatML template), not just the last token.
  * DiM in float64.
  * Output candidate-direction tensor shape (n_pos, n_layer, hidden_dim) per
    model — drop-in compatible with Arditi's `candidate_directions`.
  * Reports Arditi's prune zone (last 20 % of layers) on every figure;
    "best layer" by probe is selected from the unpruned region only.
  * Renamed metric: `probe_emergence_layer` (observational), with explicit
    docstring noting this is NOT Arditi's causal best-layer.

Output:
    results/refusal_direction_<TS>/
        candidate_directions.npz       # per-model (n_pos, L, D) tensor
        per_model_metrics.json
        cross_model_metrics.json       # (n_pos, L) cosine per axis pair
        run_meta.json
        fig_norm_heatmap.png           # (n_models)-panel ‖r‖ heatmap (n_pos × L)
        fig_probe_acc_heatmap.png      # (n_models)-panel probe-acc heatmap (n_pos × L)
        fig_probe_acc_curves.png       # per-layer probe acc, max over positions
        fig_emergence_curve.png        # probe acc with prune zone shaded
        fig_cosine_pairs.png           # last-position cos(r^A_l, r^B_l) per axis

Usage:
    python -m llm_lens.examples.run_refusal_direction --dataset cm_binary --max-per-class 128
    python -m llm_lens.examples.run_refusal_direction --dataset arditi_combined --max-per-class 128
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

from llm_lens.extractor import ActivationExtractor
from llm_lens.datasets import load_prompts
from llm_lens.refusal import (
    DEFAULT_EOI_LEN,
    capture_eoi_residuals,
    extract_candidate_directions,
    candidate_norms,
    cosine_per_position_layer,
    binary_probe_per_position_layer,
    arditi_layer_prune_mask,
    find_probe_emergence_layer,
)
from llm_lens.model_zoo import (
    MODEL_SETS, get_models, get_axis_pairs, parse_dtype,
    DEFAULT_MODEL_SET, DEFAULT_DTYPE, DTYPE_CHOICES,
)


# ── Per-model extraction ────────────────────────────────────────────────────

def extract_for_model(
    full: str,
    short: str,
    harmless_prompts: list[str],
    harmful_prompts:  list[str],
    eoi_len: int,
    dtype: torch.dtype,
) -> dict:
    """Load model, capture eoi residuals, compute (n_pos, L, D) tensor + probes."""
    print(f"\n{'=' * 60}\n  MODEL = {short}\n{'=' * 60}")
    ext = ActivationExtractor(full, dtype=dtype, capture_heads=False)

    t0 = time.time()
    print(f"  capturing harmless residuals (n={len(harmless_prompts)})...")
    safe_resid = capture_eoi_residuals(
        ext, harmless_prompts, apply_chat_template=True, eoi_len=eoi_len)
    print(f"  capturing harmful residuals (n={len(harmful_prompts)})...")
    harm_resid = capture_eoi_residuals(
        ext, harmful_prompts, apply_chat_template=True, eoi_len=eoi_len)
    capture_dt = time.time() - t0
    print(f"  capture done in {capture_dt:.1f}s — "
          f"shapes harmless={safe_resid.shape}, harmful={harm_resid.shape}")

    directions = extract_candidate_directions(harm_resid, safe_resid)        # (n_pos, L, D)
    norms      = candidate_norms(directions)                                  # (n_pos, L)
    print(f"  computing per-(pos, layer) binary probe (LR)...")
    t1 = time.time()
    probe_accs = binary_probe_per_position_layer(harm_resid, safe_resid)     # (n_pos, L)
    probe_dt = time.time() - t1

    # Best-by-probe location, applying Arditi pruning
    n_layers = probe_accs.shape[1]
    prune_mask = arditi_layer_prune_mask(n_layers, prune_pct=0.2)            # (L,)
    masked = probe_accs.copy()
    masked[:, ~prune_mask] = -np.inf                                          # exclude pruned
    flat_idx = int(np.argmax(masked))
    best_pos, best_layer = np.unravel_index(flat_idx, masked.shape)
    best_acc = float(probe_accs[best_pos, best_layer])

    # Emergence layer (over best position) — max-position-collapsed curve
    max_pos_curve = probe_accs.max(axis=0)                                    # (L,)
    emerg_layer = find_probe_emergence_layer(
        max_pos_curve, rel_threshold=0.98, apply_arditi_prune=True)

    print(f"  probe done in {probe_dt:.1f}s — "
          f"best_acc={best_acc:.4f} at (pos=-{eoi_len-best_pos}, L{int(best_layer)}); "
          f"emergence (98% peak, unpruned) = L{emerg_layer}")

    # Raw residuals at best (pos, layer) — for sample-level scatter plots.
    # Cheap to keep: (N, D) per side, ~5 MB at 7B bf16. Per-prompt rather
    # than per-(pos, layer) — only the chosen slice.
    raw_safe_best  = safe_resid[:, best_pos, best_layer, :].astype(np.float32)
    raw_harm_best  = harm_resid[:, best_pos, best_layer, :].astype(np.float32)

    # Free GPU
    del ext
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "directions":     directions,          # (n_pos, L, D)
        "norms":          norms,               # (n_pos, L)
        "probe_acc":      probe_accs,          # (n_pos, L)
        "best_pos_idx":   int(best_pos),
        "best_layer":     int(best_layer),
        "best_acc":       best_acc,
        "probe_emergence_layer": int(emerg_layer),
        # Means kept for sanity / cross-checks (no full N×n_pos×L×D dump)
        "harmless_mean": safe_resid.astype(np.float64).mean(axis=0).astype(np.float32),
        "harmful_mean":  harm_resid.astype(np.float64).mean(axis=0).astype(np.float32),
        # Raw residuals at best (pos, layer) — used by single-model scatter plots.
        "raw_safe_best": raw_safe_best,        # (N_harmless, D)
        "raw_harm_best": raw_harm_best,        # (N_harmful,  D)
    }


# ── Plotting (5 figures) ────────────────────────────────────────────────────

def make_figures(per_model: dict, cos_pairs: dict, eoi_len: int, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    short_names = list(per_model.keys())
    L = per_model[short_names[0]]["norms"].shape[1]
    layers = np.arange(L)
    pos_labels = [f"-{eoi_len - i}" for i in range(eoi_len)]
    prune_mask = arditi_layer_prune_mask(L, prune_pct=0.2)
    prune_start = int(np.argmin(prune_mask))                  # first pruned layer

    color_map = {}
    palette = ["tab:gray", "tab:blue", "tab:green", "tab:red", "tab:purple"]
    for i, n in enumerate(short_names):
        color_map[n] = palette[i % len(palette)]

    # ── Fig 1 — per-model (n_pos × L) norm heatmap ──────────────────────────
    n_models = len(short_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4),
                             sharey=True)
    if n_models == 1:
        axes = [axes]
    vmax = max(per_model[n]["norms"].max() for n in short_names)
    for ax, name in zip(axes, short_names):
        norms = per_model[name]["norms"]
        im = ax.imshow(norms, aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.axvline(prune_start - 0.5, color="red", lw=1, linestyle="--",
                   alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_xticks(np.arange(0, L, 4))
        ax.set_yticks(range(eoi_len))
        ax.set_yticklabels(pos_labels)
        ax.set_ylabel("Position (relative to last token)")
        ax.set_title(f"{name}  ‖r_pos,l‖")
        plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Per-(position, layer) refusal direction norm — "
                 "red dashed = Arditi prune zone start (last 20 % of layers)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_norm_heatmap.png"), dpi=120)
    plt.close(fig)

    # ── Fig 2 — per-model (n_pos × L) probe-acc heatmap ─────────────────────
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4),
                             sharey=True)
    if n_models == 1:
        axes = [axes]
    for ax, name in zip(axes, short_names):
        accs = per_model[name]["probe_acc"]
        im = ax.imshow(accs, aspect="auto", vmin=0.5, vmax=1.0, cmap="RdYlGn")
        ax.axvline(prune_start - 0.5, color="black", lw=1, linestyle="--",
                   alpha=0.7)
        ax.set_xlabel("Layer")
        ax.set_xticks(np.arange(0, L, 4))
        ax.set_yticks(range(eoi_len))
        ax.set_yticklabels(pos_labels)
        ax.set_ylabel("Position (relative to last token)")
        ax.set_title(f"{name}  probe-acc")
        plt.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle("Per-(position, layer) binary probe accuracy — "
                 "black dashed = Arditi prune zone start")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_probe_acc_heatmap.png"), dpi=120)
    plt.close(fig)

    # ── Fig 3 — per-layer probe-acc curve (max over positions) ──────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in short_names:
        curve = per_model[name]["probe_acc"].max(axis=0)
        ax.plot(layers, curve, marker="o", ms=3, label=name,
                color=color_map[name])
    ax.axvspan(prune_start - 0.5, L - 0.5, color="red", alpha=0.1,
               label="Arditi prune zone (last 20 %)")
    ax.axhline(0.5, color="k", lw=0.5, linestyle=":", alpha=0.5)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best probe accuracy across positions")
    ax.set_title("Where in depth is refusal linearly decodable? — max-over-positions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_probe_acc_curves.png"), dpi=120)
    plt.close(fig)

    # ── Fig 4 — emergence layer overlay ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for name in short_names:
        curve = per_model[name]["probe_acc"].max(axis=0)
        ax.plot(layers, curve, marker="o", ms=3, label=name,
                color=color_map[name])
        emerg = per_model[name]["probe_emergence_layer"]
        ax.axvline(emerg, color=color_map[name], lw=1, linestyle="--",
                   alpha=0.7)
        ax.text(emerg, 0.55, f"L{emerg}", color=color_map[name],
                rotation=90, va="bottom", ha="right", fontsize=8)
    ax.axvspan(prune_start - 0.5, L - 0.5, color="red", alpha=0.1)
    ax.axhline(0.5, color="k", lw=0.5, linestyle=":", alpha=0.5)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Best probe accuracy across positions")
    ax.set_title(
        "Probe-based refusal-encoding emergence layer (98 % of unpruned peak)\n"
        "[NOT Arditi's causal best-layer — observational only]")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_emergence_curve.png"), dpi=120)
    plt.close(fig)

    # ── Fig 5 — last-position cosine across model pairs ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for pair_name, cos in cos_pairs.items():
        # cos shape (n_pos, L); show position -1 (last token) curve
        ax.plot(layers, cos[-1], marker="s", ms=3, label=pair_name)
    ax.axvspan(prune_start - 0.5, L - 0.5, color="red", alpha=0.1,
               label="Arditi prune zone")
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.axhline(1, color="k", lw=0.5, alpha=0.3, linestyle="--")
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("Layer")
    ax.set_ylabel("cos(r^A, r^B) at last-eoi position")
    ax.set_title("Cross-model alignment of refusal direction — last-eoi position")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_cosine_pairs.png"), dpi=120)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Arditi-style multi-position refusal-direction extraction")
    p.add_argument("--dataset", default="cm_binary",
                   help="Binary harmful/harmless dataset registered in datasets.py: "
                        "cm_binary | arditi_combined | (single-source ones to be combined manually)")
    p.add_argument("--max-per-class", type=int, default=128,
                   help="Cap on harmless / harmful prompt counts (each side). "
                        "Arditi paper uses 128.")
    p.add_argument("--eoi-len", type=int, default=DEFAULT_EOI_LEN,
                   help="Number of trailing positions to capture (Qwen ChatML "
                        "eoi_toks length is 5).")
    p.add_argument("--model-set", default=DEFAULT_MODEL_SET,
                   choices=list(MODEL_SETS),
                   help="Which model quartet to run.")
    p.add_argument("--dtype", default=DEFAULT_DTYPE, choices=list(DTYPE_CHOICES))
    p.add_argument("--results-root", default="results")
    p.add_argument("--output-suffix", default=None)
    p.add_argument("--targets", nargs="+", default=None,
                   help="Subset of model short labels (default: all 4)")
    args = p.parse_args()

    dtype = parse_dtype(args.dtype)
    out_ts = args.output_suffix or datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = os.path.join(args.results_root, f"refusal_direction_{out_ts}")
    print(f"\n[run_refusal_direction] output → {out_dir}")
    print(f"  dataset={args.dataset}  max_per_class={args.max_per_class}  "
          f"eoi_len={args.eoi_len}  model_set={args.model_set}  dtype={args.dtype}")

    # Load data BEFORE creating output dir.
    prompts = load_prompts(args.dataset, max_per_class=args.max_per_class)
    if not {"harmless", "harmful"}.issubset(prompts):
        raise ValueError(
            f"dataset '{args.dataset}' must yield 'harmless'+'harmful' tags; "
            f"got {list(prompts)}. Use 'arditi_combined' to mix single-source "
            f"datasets, or 'cm_binary' for the IBM split.")
    harmless = prompts["harmless"]
    harmful  = prompts["harmful"]

    os.makedirs(out_dir, exist_ok=True)

    # Per-model extraction loop
    targets = args.targets or [s for _, s in get_models(args.model_set)]
    per_model: dict[str, dict] = {}
    for full, short in get_models(args.model_set):
        if short not in targets:
            continue
        per_model[short] = extract_for_model(
            full=full, short=short,
            harmless_prompts=harmless, harmful_prompts=harmful,
            eoi_len=args.eoi_len, dtype=dtype)

    # Cross-model cosine: one (n_pos, L) per pair. We use the named axis pairs
    # from get_axis_pairs (RLHF / domain / self_evolving) PLUS all C(n,2) pairs
    # so the figure can show the full landscape if desired.
    short_names = list(per_model.keys())
    cos_pairs: dict[str, np.ndarray] = {}
    for i, a in enumerate(short_names):
        for b in short_names[i + 1:]:
            cos_pairs[f"{a} vs {b}"] = cosine_per_position_layer(
                per_model[a]["directions"], per_model[b]["directions"])

    # ── Save artefacts ─────────────────────────────────────────────────────
    np.savez(
        os.path.join(out_dir, "candidate_directions.npz"),
        **{f"{k}__directions": v["directions"] for k, v in per_model.items()},
        **{f"{k}__norms":      v["norms"]      for k, v in per_model.items()},
        **{f"{k}__probe_acc":  v["probe_acc"]  for k, v in per_model.items()},
        **{f"{k}__safe_mean":  v["harmless_mean"] for k, v in per_model.items()},
        **{f"{k}__harm_mean":  v["harmful_mean"]  for k, v in per_model.items()},
        **{f"{k}__raw_safe_best": v["raw_safe_best"] for k, v in per_model.items()},
        **{f"{k}__raw_harm_best": v["raw_harm_best"] for k, v in per_model.items()},
        **{f"{k}__best_pos":   np.array([v["best_pos_idx"]]) for k, v in per_model.items()},
        **{f"{k}__best_layer": np.array([v["best_layer"]])   for k, v in per_model.items()},
        **{f"cos__{name.replace(' vs ', '__')}": cos
           for name, cos in cos_pairs.items()},
    )

    payload_per_model = {
        k: {
            "norms":               v["norms"].tolist(),
            "probe_acc":           v["probe_acc"].tolist(),
            "best_pos_idx":        int(v["best_pos_idx"]),
            "best_layer":          int(v["best_layer"]),
            "best_acc":            float(v["best_acc"]),
            "probe_emergence_layer": int(v["probe_emergence_layer"]),
        } for k, v in per_model.items()
    }
    payload_cross = {name: cos.tolist() for name, cos in cos_pairs.items()}

    axis_pairs = get_axis_pairs(args.model_set)
    with open(os.path.join(out_dir, "per_model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload_per_model, f, indent=2)
    with open(os.path.join(out_dir, "cross_model_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload_cross, f, indent=2)
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":         out_ts,
            "dataset":           args.dataset,
            "max_per_class":     args.max_per_class,
            "n_harmless":        len(harmless),
            "n_harmful":         len(harmful),
            "eoi_len":           args.eoi_len,
            "model_set":         args.model_set,
            "dtype":             args.dtype,
            "chat_template":     "QWEN_MIN_CHAT_TEMPLATE (forced minimal Qwen ChatML)",
            "axis_pairs":        axis_pairs,
            "arditi_alignment":  {
                "token_position": f"full eoi_toks range, {args.eoi_len} positions [-{args.eoi_len}..-1]",
                "dim_dtype":      "float64 accumulation, float32 storage",
                "layer_pruning":  "last 20% of layers excluded from emergence/best-layer search",
                "best_layer_metric": "probe_acc (observational); NOT Arditi causal ablation",
            },
            "written_at":        datetime.now().isoformat(),
        }, f, indent=2)

    # ── Make figures ───────────────────────────────────────────────────────
    try:
        make_figures(per_model, cos_pairs, args.eoi_len, out_dir)
        print(f"\nFigures: fig_norm_heatmap.png, fig_probe_acc_heatmap.png, "
              f"fig_probe_acc_curves.png, fig_emergence_curve.png, fig_cosine_pairs.png")
    except Exception as e:
        print(f"  [WARN] plotting failed: {e}")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    eoi_len = args.eoi_len
    for k, v in per_model.items():
        pos_label = f"-{eoi_len - v['best_pos_idx']}"
        print(f"  {k:<22} probe best={v['best_acc']:.4f} @ "
              f"(pos={pos_label}, L{v['best_layer']:>2}); "
              f"emergence (98% unpruned peak) at L{v['probe_emergence_layer']}")

    print("\n  Pairwise refusal-direction cosine at last-eoi position, last unpruned layer:")
    L_last = per_model[short_names[0]]["directions"].shape[1] - 1
    L_prune_start = int(np.argmin(arditi_layer_prune_mask(L_last + 1)))
    L_report = max(0, L_prune_start - 1)
    for name, cos in cos_pairs.items():
        v_last_unpruned = float(cos[-1, L_report])
        v_actual_last   = float(cos[-1, L_last])
        print(f"    {name:<48}  cos[L{L_report}, last-eoi]={v_last_unpruned:+.4f}  "
              f"cos[L{L_last}, last-eoi]={v_actual_last:+.4f}")

    if axis_pairs:
        print("\n  Axis-aligned cosines @ last-eoi, last unpruned layer (L%d):" % L_report)
        for axis_name, (a, b) in axis_pairs.items():
            key = f"{a} vs {b}"
            if key not in cos_pairs:
                key = f"{b} vs {a}"
            if key not in cos_pairs:
                print(f"    {axis_name:<14} (skipped: both endpoints not run)")
                continue
            v_last_unpruned = float(cos_pairs[key][-1, L_report])
            print(f"    {axis_name:<14} {a:<22} ↔ {b:<22}  cos = {v_last_unpruned:+.4f}")

    print(f"\nAll outputs in {out_dir}/")


if __name__ == "__main__":
    main()
