#!/usr/bin/env python3
"""
Bootstrap confidence intervals on cross-model behaviour-direction cosines.

Finding 3 reported cos(v_A, v_B) ≈ 0 (0.05, 0.03, -0.06) for the three
3B donors. With n_refused ∈ {6, 34, 39} per donor, these point estimates
have non-trivial sampling error. We do stratified bootstrap on each
donor's refused/complied harmful records to estimate the 95% CI on
each cross-cosine.

Per iteration:
  for each donor M in {Qwen-base, Instruct, Coder}:
    refused_M_boot   = sample-with-replacement from M's refused harmful records (same n)
    complied_M_boot  = sample-with-replacement from M's complied harmful records (same n)
    v_M_boot = mean(refused_M_boot) - mean(complied_M_boot), normalised
  cos(v_A_boot, v_B_boot) per pair

Repeat n_iter=2000 times, report 95% CI from percentiles + median.

Also include cosine-with-Arditi-direction CIs as positive control
(those should NOT shrink to 0).

Output:
  results/bootstrap_cos_<TS>/
    cos_distributions.npz
    summary.json
    summary.md
    fig_violin.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.examples.run_behaviour_direction_transfer import (
    load_per_model, N_MIN_REFUSED, N_MIN_COMPLIED,
)


def bootstrap_cos(recsA, recsB, n_iter, rng):
    refA = [r["raw"] for r in recsA if r["refused"]]
    comA = [r["raw"] for r in recsA if not r["refused"]]
    refB = [r["raw"] for r in recsB if r["refused"]]
    comB = [r["raw"] for r in recsB if not r["refused"]]
    if not (refA and comA and refB and comB):
        return np.array([])
    refA = np.array(refA); comA = np.array(comA)
    refB = np.array(refB); comB = np.array(comB)
    nA_r, nA_c = len(refA), len(comA)
    nB_r, nB_c = len(refB), len(comB)
    out = np.empty(n_iter, dtype=np.float64)
    for k in range(n_iter):
        iA_r = rng.integers(0, nA_r, nA_r)
        iA_c = rng.integers(0, nA_c, nA_c)
        iB_r = rng.integers(0, nB_r, nB_r)
        iB_c = rng.integers(0, nB_c, nB_c)
        vA = refA[iA_r].mean(0) - comA[iA_c].mean(0)
        vB = refB[iB_r].mean(0) - comB[iB_c].mean(0)
        nA = np.linalg.norm(vA); nB = np.linalg.norm(vB)
        out[k] = (vA @ vB) / (nA * nB) if (nA > 1e-9 and nB > 1e-9) else np.nan
    return out


def bootstrap_cos_with_arditi(recs, arditi_unit, n_iter, rng):
    ref = [r["raw"] for r in recs if r["refused"]]
    com = [r["raw"] for r in recs if not r["refused"]]
    if not (ref and com):
        return np.array([])
    ref = np.array(ref); com = np.array(com)
    nr, nc = len(ref), len(com)
    out = np.empty(n_iter, dtype=np.float64)
    for k in range(n_iter):
        ir = rng.integers(0, nr, nr)
        ic = rng.integers(0, nc, nc)
        v = ref[ir].mean(0) - com[ic].mean(0)
        n = np.linalg.norm(v)
        out[k] = (v @ arditi_unit) / n if n > 1e-9 else np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["3B", "7B"])
    parser.add_argument("--n-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = args.out_dir or f"results/bootstrap_cos_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[bootstrap-cos] out={out_dir}  n_iter={args.n_iter}")

    rng = np.random.default_rng(args.seed)
    summary_all = {}

    for size in args.sizes:
        print(f"\n=== {size} ===")
        per_model = load_per_model(size)
        donors = [m for m, info in per_model.items()
                  if info["n_refused_harm"] >= N_MIN_REFUSED
                  and info["n_comply_harm"] >= N_MIN_COMPLIED]
        if not donors:
            print(f"  no donors for {size}, skip")
            continue
        print(f"  donors: {donors}")

        # cross-pair cosines
        pair_results = {}
        for i, A in enumerate(donors):
            for B in donors[i + 1:]:
                key = f"{A}__vs__{B}"
                samples = bootstrap_cos(
                    per_model[A]["harmful_recs"],
                    per_model[B]["harmful_recs"],
                    args.n_iter, rng,
                )
                samples = samples[~np.isnan(samples)]
                if len(samples) == 0:
                    continue
                pair_results[key] = {
                    "samples": samples,
                    "median": float(np.median(samples)),
                    "mean": float(samples.mean()),
                    "ci_95_lo": float(np.percentile(samples, 2.5)),
                    "ci_95_hi": float(np.percentile(samples, 97.5)),
                    "frac_within_0_05": float(np.mean(np.abs(samples) < 0.05)),
                    "frac_within_0_2": float(np.mean(np.abs(samples) < 0.2)),
                    "frac_above_0_5": float(np.mean(samples > 0.5)),
                }
                r = pair_results[key]
                print(f"  cos({key.replace('__vs__', ' x ')}): "
                      f"median={r['median']:+.3f}, "
                      f"95% CI=[{r['ci_95_lo']:+.3f}, {r['ci_95_hi']:+.3f}], "
                      f"P(|cos|<0.05)={r['frac_within_0_05']:.1%}, "
                      f"P(|cos|<0.2)={r['frac_within_0_2']:.1%}, "
                      f"P(cos>0.5)={r['frac_above_0_5']:.1%}")

        # also bootstrap cos with Arditi (positive control)
        arditi_results = {}
        for m in donors:
            samples = bootstrap_cos_with_arditi(
                per_model[m]["harmful_recs"],
                per_model[m]["unit_arditi"],
                args.n_iter, rng,
            )
            samples = samples[~np.isnan(samples)]
            arditi_results[m] = {
                "samples": samples,
                "median": float(np.median(samples)),
                "ci_95_lo": float(np.percentile(samples, 2.5)),
                "ci_95_hi": float(np.percentile(samples, 97.5)),
            }
            r = arditi_results[m]
            print(f"  cos(v_{m}, Arditi_{m}): median={r['median']:+.3f}, "
                  f"95% CI=[{r['ci_95_lo']:+.3f}, {r['ci_95_hi']:+.3f}]")

        summary_all[size] = {
            "donors": donors,
            "n_iter": args.n_iter,
            "cross_pair": {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                           for k, v in pair_results.items()},
            "with_arditi": {k: {kk: vv for kk, vv in v.items() if kk != "samples"}
                            for k, v in arditi_results.items()},
        }

        # save arrays
        save_arrays = {f"cross__{k}": v["samples"] for k, v in pair_results.items()}
        save_arrays.update({f"arditi__{m}": v["samples"] for m, v in arditi_results.items()})
        np.savez(os.path.join(out_dir, f"cos_distributions_{size}.npz"), **save_arrays)

        # violin plot
        make_violin(pair_results, arditi_results, size, out_dir)

    write_markdown(summary_all, out_dir)
    meta = {
        "timestamp": ts,
        "sizes": args.sizes,
        "n_iter": args.n_iter,
        "seed": args.seed,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nDone → {out_dir}")


def make_violin(pair_results, arditi_results, size, out_dir):
    pair_keys = list(pair_results.keys())
    arditi_keys = list(arditi_results.keys())
    if not pair_keys and not arditi_keys:
        return
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * (len(pair_keys) + len(arditi_keys))), 5))
    data = []
    labels = []
    colors = []
    for k in pair_keys:
        data.append(pair_results[k]["samples"])
        a, b = k.split("__vs__")
        labels.append(f"cross\n{a}\n× {b}")
        colors.append("tab:blue")
    for k in arditi_keys:
        data.append(arditi_results[k]["samples"])
        labels.append(f"Arditi\n× v_{k}")
        colors.append("tab:gray")
    parts = ax.violinplot(data, showmedians=True, widths=0.7)
    for pc, c in zip(parts["bodies"], colors):
        pc.set_facecolor(c); pc.set_alpha(0.55)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.0, color="k", lw=0.6, ls="--", alpha=0.6)
    ax.set_ylabel("cosine (bootstrap distribution)")
    ax.set_ylim(-0.6, 1.0)
    ax.set_title(f"{size}: bootstrap distributions of cross-model behaviour cos\n"
                 "(blue = behaviour×behaviour cross, gray = Arditi×behaviour positive control)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_violin_{size}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_markdown(summary_all, out_dir):
    lines = [
        "# Bootstrap cosine confidence intervals",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        ("Stratified bootstrap on each donor's refused/complied harmful records "
         "(sample-with-replacement, preserving counts). 2000 iterations per pair. "
         "We report median + 95% CI."),
        "",
        ("**Interpretation guide**:"),
        ("- `P(|cos|<0.05)`: probability that the bootstrap cos is near-zero "
         "(within ±0.05). High = behaviour directions are tight-clustered around orthogonal."),
        ("- `P(|cos|<0.2)`: probability of weak alignment (|cos|<0.2)."),
        ("- `P(cos>0.5)`: probability of substantial positive alignment (red flag for our Finding 3 if non-trivial)."),
        "",
    ]
    for size, s in summary_all.items():
        if not s:
            continue
        lines += [f"## {size}  (n_iter = {s['n_iter']})", ""]
        lines += [
            "### Cross-pair `cos(v_A, v_B)` (the Finding 3 metric)",
            "",
            "| pair | median | 95% CI | P(|cos|<0.05) | P(|cos|<0.2) | P(cos>0.5) |",
            "|---|---|---|---|---|---|",
        ]
        for k, r in s["cross_pair"].items():
            a, b = k.split("__vs__")
            lines.append(
                f"| {a} × {b} | {r['median']:+.3f} | "
                f"[{r['ci_95_lo']:+.3f}, {r['ci_95_hi']:+.3f}] | "
                f"{r['frac_within_0_05']:.1%} | "
                f"{r['frac_within_0_2']:.1%} | "
                f"{r['frac_above_0_5']:.1%} |"
            )
        lines += ["", "### Positive control: `cos(v_M, Arditi_M)`", "",
                  "| model | median | 95% CI |", "|---|---|---|"]
        for m, r in s["with_arditi"].items():
            lines.append(
                f"| {m} | {r['median']:+.3f} | "
                f"[{r['ci_95_lo']:+.3f}, {r['ci_95_hi']:+.3f}] |"
            )
        lines += ["", ""]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
