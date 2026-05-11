"""Visualization for value-benchmark results.

Three plots:
  1. value_benchmarks_summary.png — one panel per benchmark, 3 bars per panel
  2. bbq_by_category.png — BBQ accuracy broken down by social category
  3. advbench_distribution.png — per-prompt logprob histogram, 3 models overlaid
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_COLORS = {
    "Qwen2.5-3B":          "#000000",   # base (black)
    "Qwen2.5-3B-Instruct": "#1976D2",   # instruct (blue)
    "Qwen2.5-Coder-3B":    "#388E3C",   # domain (green)
    "AZR-Coder-3B":        "#D32F2F",   # azr (red)
    "Qwen2.5-7B":          "#000000",
    "Qwen2.5-7B-Instruct": "#1976D2",
    "Qwen2.5-Coder-7B":    "#388E3C",
    "AZR-Base-7B":         "#FF9800",   # self-evolved direct (orange)
    "AZR-Coder-7B":        "#D32F2F",
}
MODEL_ORDER = [
    "Qwen2.5-3B", "Qwen2.5-3B-Instruct", "Qwen2.5-Coder-3B", "AZR-Coder-3B",
    "Qwen2.5-7B", "Qwen2.5-7B-Instruct", "Qwen2.5-Coder-7B",
    "AZR-Base-7B", "AZR-Coder-7B",
]


def _present_models(results: dict) -> list[str]:
    return [m for m in MODEL_ORDER if m in results]


_SHORT_LABEL_OVERRIDES = {
    "Qwen2.5-3B":          "Base",
    "Qwen2.5-3B-Instruct": "Instruct",
    "Qwen2.5-Coder-3B":    "Coder",
    "AZR-Coder-3B":        "AZR",
    "Qwen2.5-7B":          "Base",
    "Qwen2.5-7B-Instruct": "Instruct",
    "Qwen2.5-Coder-7B":    "Coder",
    "AZR-Base-7B":         "AZR-Base",
    "AZR-Coder-7B":        "AZR-Coder",
}


def _short_label(m: str) -> str:
    return _SHORT_LABEL_OVERRIDES.get(m, m)


def plot_all(results: dict, out_dir: str) -> None:
    benchmarks = list(results.get("_meta", {}).get("benchmarks") or
                       [k for k in next(iter(results.values())).keys()
                        if not k.startswith("_")])
    models = _present_models(results)

    # ── FIG 1: per-benchmark summary ───────────────────────────────────────
    n = len(benchmarks)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)
    axes = axes[0]

    for ax, bench in zip(axes, benchmarks):
        # Pick the right primary metric per benchmark
        if bench == "advbench_compliance":
            metric_key = "mean_logprob_target"
            ylabel = "log P(compliant continuation | request)\n(higher = more willing to comply)"
            title  = "AdvBench compliance"
        elif bench == "truthfulqa_mc1":
            metric_key = "accuracy"
            ylabel = "Accuracy"
            title  = "TruthfulQA MC1 (honesty)"
        elif bench == "bbq_ambig":
            metric_key = "accuracy"
            ylabel = "Accuracy on ambiguous\n(picking the 'unknown' option)"
            title  = "BBQ ambiguous (anti-bias)"
        else:
            metric_key = "accuracy"
            ylabel = "Score"; title = bench

        values = []
        for m in models:
            r = results.get(m, {}).get(bench, {})
            v = r.get(metric_key)
            values.append(np.nan if v is None else float(v))

        x = np.arange(len(models))
        bars = ax.bar(x, values, color=[MODEL_COLORS.get(m, "#888") for m in models],
                       alpha=0.85, edgecolor="black", lw=0.6)

        # Label bars
        for xi, v in zip(x, values):
            if not np.isnan(v):
                ax.text(xi, v, f"{v:.3f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=9)

        # Reference lines
        if bench == "truthfulqa_mc1":
            ax.axhline(0.25, color="gray", ls=":", lw=1,
                       label="random (4-way)")
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.0)
        elif bench == "bbq_ambig":
            ax.axhline(1/3, color="gray", ls=":", lw=1,
                       label="random (3-way)")
            ax.legend(fontsize=8)
            ax.set_ylim(0, 1.0)

        ax.set_xticks(x)
        ax.set_xticklabels([_short_label(m) for m in models],
                           rotation=15, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Value-benchmark comparison — output-level differences across training paradigms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(); fig.subplots_adjust(top=0.86)
    out1 = os.path.join(out_dir, "value_benchmarks_summary.png")
    fig.savefig(out1, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {out1}")

    # ── FIG 2: BBQ by category ─────────────────────────────────────────────
    if "bbq_ambig" in benchmarks and any(
        results.get(m, {}).get("bbq_ambig", {}).get("by_category")
        for m in models
    ):
        # collect categories from first model that has them
        any_cats = None
        for m in models:
            cats = results.get(m, {}).get("bbq_ambig", {}).get("by_category")
            if cats:
                any_cats = list(cats.keys()); break
        if any_cats:
            n_models = len(models)
            fig, ax = plt.subplots(figsize=(max(9, 1.4 * len(any_cats) + 2), 5))
            x = np.arange(len(any_cats))
            width = 0.8 / max(n_models, 1)
            offset = (n_models - 1) / 2.0
            for i, m in enumerate(models):
                cats = results.get(m, {}).get("bbq_ambig", {}).get("by_category", {})
                vals = [cats.get(c, np.nan) for c in any_cats]
                ax.bar(x + (i - offset) * width, vals, width=width,
                       color=MODEL_COLORS.get(m, "#888"), alpha=0.85,
                       edgecolor="black", lw=0.5,
                       label=_short_label(m) + f" ({m})")
            ax.axhline(1/3, color="gray", ls=":", lw=1,
                       label="random (3-way)")
            ax.set_xticks(x)
            ax.set_xticklabels(any_cats, rotation=15, fontsize=10)
            ax.set_ylabel("Accuracy on ambiguous (gold = 'unknown')", fontsize=11)
            ax.set_title("BBQ ambiguous by category — bias incidence per axis",
                         fontsize=13, fontweight="bold")
            ax.set_ylim(0, 1.0)
            ax.legend(fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out2 = os.path.join(out_dir, "bbq_by_category.png")
            fig.savefig(out2, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"  saved {out2}")

    # ── FIG 3: AdvBench logprob distribution ───────────────────────────────
    if "advbench_compliance" in benchmarks:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        any_hist = False
        all_means = []
        for m in models:
            r = results.get(m, {}).get("advbench_compliance", {})
            per = r.get("per_item", [])
            color = MODEL_COLORS.get(m, "#888")
            if per:
                lps = np.array([row["logprob_target"] for row in per])
                ax.hist(lps, bins=40, alpha=0.40, color=color,
                        label=f"{m} (per-item, mean={lps.mean():.3f})",
                        edgecolor="black", lw=0.3)
                ax.axvline(lps.mean(), color=color, ls="--", alpha=0.85, lw=2)
                all_means.append(lps.mean())
                any_hist = True
            else:
                # Per-item not saved — only draw a mean vertical line
                mean_v = r.get("mean_logprob_target")
                if mean_v is not None:
                    ax.axvline(mean_v, color=color, ls="-", alpha=0.85, lw=2.2,
                               label=f"{m} (mean only, no per-item)")
                    all_means.append(mean_v)
        ax.set_xlabel("Per-token log P(compliant target | harmful request)\n"
                       "(less negative = more willing)")
        ax.set_ylabel("Count of prompts" if any_hist else "")
        ax.set_title("AdvBench compliance likelihood — distribution per model",
                     fontsize=13, fontweight="bold")
        if any_hist or all_means:
            ax.legend(fontsize=9, loc="upper left")
        ax.grid(True, alpha=0.3)
        if not any_hist and all_means:
            ax.set_xlim(min(all_means) - 0.5, max(all_means) + 0.5)
            ax.set_yticks([])
        fig.tight_layout()
        out3 = os.path.join(out_dir, "advbench_distribution.png")
        fig.savefig(out3, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out3}")
