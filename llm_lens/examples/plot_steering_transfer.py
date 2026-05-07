"""Visualization for steering-transfer results.

Three plots are produced into the supplied output dir:

  1. steering_transfer_curves.png — per-(target, layer) panel grid:
        x = strength, y = mean cos-shift score, lines = condition
  2. steering_transfer_summary.png — single bar chart at the largest |strength|,
        x = (target, layer) bins, bars = condition (native vs raw vs procrust)
  3. procrustes_recovery_ratio.png — line plot of Procrustes/Native ratio,
        showing how completely Procrustes-aligned transfer recovers native effect

Input format: the dict that run_steering_transfer.py serialised as results.json,
or that dict already in memory.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COND_STYLE = {
    "baseline": dict(color="#888888", marker="o", ls=":",  lw=1.4, label="baseline (no inject)"),
    "native":   dict(color="#1976D2", marker="o", ls="-",  lw=2.4, label="native (own vec)"),
    "raw":      dict(color="#FF9800", marker="s", ls="--", lw=2.0, label="raw transfer (no rotation)"),
    "procrust": dict(color="#43A047", marker="^", ls="-",  lw=2.4, label="Procrustes transfer"),
}

TARGET_ORDER = ["Qwen2.5-3B", "Qwen2.5-3B-Instruct", "AZR-Coder-3B"]


def _mean_score(cell, cond):
    if cell is None: return None
    v = cell.get(cond)
    if v is None or len(v) == 0: return None
    return float(np.mean(v))


def _se_score(cell, cond):
    if cell is None: return 0.0
    v = cell.get(cond)
    if v is None or len(v) == 0: return 0.0
    arr = np.asarray(v)
    return float(arr.std(ddof=1) / max(np.sqrt(len(arr)), 1))


def plot_all(payload: dict, out_dir: str) -> None:
    layers    = list(map(int, payload["layers"]))
    strengths = list(map(float, payload["strengths"]))
    results   = payload["results"]
    targets   = [t for t in TARGET_ORDER if t in results]
    target_tag = payload["target_tag"]
    ref_tag    = payload["ref_tag"]
    prompt_tag = payload["prompt_tag"]
    n          = payload["n_eval_prompts"]

    # baseline value at strength=0 (per target × layer) used for delta shifts
    def baseline_score(target: str, layer: int):
        cell = results[target].get(str(layer)) or results[target].get(layer)
        if cell is None: return None
        # strength=0 is keyed as either "0.0" or 0.0 depending on how json round-tripped
        for k in (0.0, "0.0", "0"):
            cz = cell.get(k)
            if cz is not None:
                bs = _mean_score(cz, "baseline")
                if bs is not None: return bs
        return None

    # ── FIG 1: per-(target, layer) curves ──────────────────────────────────
    rows = len(targets)
    cols = len(layers)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                              sharex=True, squeeze=False)
    # squeeze=False guarantees axes is always 2D, even when rows==1 or cols==1

    for ri, target in enumerate(targets):
        for ci, layer in enumerate(layers):
            ax = axes[ri, ci]
            cell_per_strength = results[target].get(str(layer)) or results[target].get(layer)
            if cell_per_strength is None:
                ax.set_visible(False); continue

            # Collect curves
            for cond, style in COND_STYLE.items():
                xs, ys, errs = [], [], []
                for s in strengths:
                    c = cell_per_strength.get(s) or cell_per_strength.get(str(s))
                    if c is None: continue
                    m = _mean_score(c, cond)
                    if m is None: continue
                    xs.append(s)
                    ys.append(m)
                    errs.append(_se_score(c, cond))
                if not xs: continue
                ax.errorbar(xs, ys, yerr=errs,
                            color=style["color"], marker=style["marker"],
                            ls=style["ls"], lw=style["lw"], ms=6, capsize=3,
                            label=style["label"])

            ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)
            ax.axvline(0.0, color="gray", lw=0.5, alpha=0.4)
            if ri == 0:
                ax.set_title(f"Inject @ L{layer}", fontsize=12, fontweight="bold")
            if ri == rows - 1:
                ax.set_xlabel("Steering strength")
            if ci == 0:
                ax.set_ylabel(f"{target}\ncos(act,{target_tag}) − cos(act,{ref_tag})",
                              fontsize=10)
            ax.grid(True, alpha=0.3)
    # one shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Steering transfer — does base's '{target_tag}-{ref_tag}' direction "
        f"work in other models?\n"
        f"(eval = {n} '{prompt_tag}' prompts, score = how far activation moves "
        f"toward '{target_tag}')",
        fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.subplots_adjust(bottom=0.10, top=0.90)
    out1 = os.path.join(out_dir, "steering_transfer_curves.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out1}")

    # ── FIG 2: bar summary at max-magnitude strength ───────────────────────
    s_max = max(strengths, key=abs)
    if s_max == 0.0:
        sorted_nonzero = [s for s in strengths if s != 0.0]
        s_max = sorted_nonzero[-1] if sorted_nonzero else None

    if s_max is not None:
        labels_x = []
        bar_data = {c: [] for c in ("native", "raw", "procrust")}
        for target in targets:
            for layer in layers:
                cell_per_strength = results[target].get(str(layer)) or results[target].get(layer)
                if cell_per_strength is None: continue
                c = cell_per_strength.get(s_max) or cell_per_strength.get(str(s_max))
                if c is None: continue
                base = baseline_score(target, layer) or 0.0
                labels_x.append(f"{target.split('-')[-1]}\nL{layer}")
                for cond in bar_data:
                    m = _mean_score(c, cond)
                    bar_data[cond].append((m - base) if m is not None else np.nan)

        if labels_x:
            x = np.arange(len(labels_x))
            width = 0.27
            fig, ax = plt.subplots(figsize=(max(11, 0.9 * len(labels_x) + 4), 5))
            ax.bar(x - width, bar_data["native"],   width=width,
                   color=COND_STYLE["native"]["color"], label="native")
            ax.bar(x,         bar_data["raw"],      width=width,
                   color=COND_STYLE["raw"]["color"], label="raw")
            ax.bar(x + width, bar_data["procrust"], width=width,
                   color=COND_STYLE["procrust"]["color"], label="Procrustes")
            ax.axhline(0, color="black", lw=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels(labels_x, fontsize=9)
            ax.set_ylabel(f"score(steered) − score(baseline)\n"
                          f"= shift toward '{target_tag}'")
            ax.set_title(f"Steering effect at strength {s_max:+g}",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=10)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out2 = os.path.join(out_dir, "steering_transfer_summary.png")
            fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  saved {out2}")

    # ── FIG 3: Procrustes / Native recovery ratio ──────────────────────────
    # For non-base targets and strength != 0:
    # ratio = procrust_shift / native_shift  (1.0 = full recovery, 0.0 = no recovery)
    fig, ax = plt.subplots(figsize=(11, 5))
    s_for_ratio = [s for s in strengths if abs(s) > 0]
    plotted_any = False
    for target in targets:
        if target == TARGET_ORDER[0]:        # skip the base (raw==procrust==native)
            continue
        for li, layer in enumerate(layers):
            cell_per_strength = results[target].get(str(layer)) or results[target].get(layer)
            if cell_per_strength is None: continue
            base = baseline_score(target, layer) or 0.0
            xs, ys_proc, ys_raw = [], [], []
            for s in s_for_ratio:
                c = cell_per_strength.get(s) or cell_per_strength.get(str(s))
                if c is None: continue
                native_m = _mean_score(c, "native")
                proc_m   = _mean_score(c, "procrust")
                raw_m    = _mean_score(c, "raw")
                if None in (native_m, proc_m, raw_m): continue
                native_shift = native_m - base
                if abs(native_shift) < 1e-6: continue
                xs.append(s)
                ys_proc.append((proc_m - base) / native_shift)
                ys_raw.append((raw_m - base) / native_shift)
            if not xs: continue
            plotted_any = True
            ax.plot(xs, ys_proc, "-",  marker="^", lw=2.2,
                    label=f"{target} L{layer} — Procrustes/native")
            ax.plot(xs, ys_raw, "--",  marker="s", lw=1.6, alpha=0.65,
                    label=f"{target} L{layer} — raw/native")

    if plotted_any:
        ax.axhline(1.0, color="green",  ls=":", alpha=0.6, label="100% recovery")
        ax.axhline(0.0, color="red",    ls=":", alpha=0.6, label="0% recovery")
        ax.set_xlabel("Steering strength")
        ax.set_ylabel("(transfer effect) / (native effect)")
        ax.set_title("Recovery ratio — how much of native steering does cross-model transfer reproduce?",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=8, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out3 = os.path.join(out_dir, "procrustes_recovery_ratio.png")
        fig.savefig(out3, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out3}")
    else:
        plt.close(fig)
