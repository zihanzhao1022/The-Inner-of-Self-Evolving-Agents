#!/usr/bin/env python3
"""
Cross-model behaviour-direction transfer.

A behaviour direction is fit per-model from its own harmful prompts:
    v_M = mean(raw_M | harmful, refused) - mean(raw_M | harmful, comply)

This is the within-class refused-vs-comply axis (different from the
across-class Arditi DiM, see paired_analysis_v2). Only models with at
least N_MIN_REFUSED refused AND N_MIN_COMPLIED complied harmful prompts
can be donors. AZR-Coder-3B (n_refused=1) cannot be a donor — but it
can still be a *recipient*: we project AZR's harmful activations onto
each donor's behaviour direction and compare the distribution to the
donor's own.

Two main outputs:

1. **Cross-cosine matrix** — cos(v_A, v_B) for all donor pairs. If the
   matrix is near-uniform-high, all models share a refusal-mediator axis
   even though they were fit independently. If AZR's base (Coder) has
   significantly higher cos with self-evolving variants than RLHF
   variants, that's evidence the self-evolving step preserves the
   behaviour axis.

2. **Transfer AUC matrix** — AUC(donor's behaviour direction predicts
   recipient's refusal, within harmful only). Diagonal is in-sample
   upper bound; off-diagonal is the cross-model generalization. A
   diagonal-dominant matrix = each model has its own behaviour
   direction. A near-uniform matrix = shared direction.

3. **AZR-Coder-3B distribution shift** — even though AZR can't be a
   donor, we plot its 18 harmful projections on each donor's v alongside
   the donor's own refused/complied distributions. If AZR's mean is
   shifted off the comply mean toward the refuse mean (despite mostly
   complying), it means activation reaches the same region but generation
   no longer translates that into refusal — a downstream decoupling.
   Conversely if AZR sits firmly in the comply region the early-layer
   activation has already shifted.

Usage:
    python -m llm_lens.examples.run_behaviour_direction_transfer
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from llm_lens.examples.run_paired_analysis_v2 import (
    classify_refusal, is_degenerate_loop,
)


N_MIN_REFUSED = 3
N_MIN_COMPLIED = 3


def load_per_model(size: str) -> dict:
    refusal_dir = f"results/refusal_direction_{size}_cm_binary_n128_with_raw"
    gen_root = f"results/generations_{size}"
    d = np.load(os.path.join(refusal_dir, "candidate_directions.npz"))
    models = sorted({k.split("__")[0] for k in d.files if k.endswith("__directions")})

    out = {}
    for m in models:
        gen_path = os.path.join(gen_root, m, "generations.jsonl")
        if not os.path.exists(gen_path):
            continue
        bp = int(d[f"{m}__best_pos"][0])
        bl = int(d[f"{m}__best_layer"][0])
        direction = d[f"{m}__directions"][bp, bl].astype(np.float64)
        unit_arditi = direction / np.linalg.norm(direction)
        raw_harm = d[f"{m}__raw_harm_best"].astype(np.float64)

        harmful_recs = []
        for line in open(gen_path, encoding="utf-8"):
            rec = json.loads(line)
            if rec["class"] != "harmful":
                continue
            idx = int(rec["prompt_idx"])
            if idx >= raw_harm.shape[0]:
                continue
            if is_degenerate_loop(rec["generation"]):
                continue
            refused, _ = classify_refusal(rec["generation"])
            harmful_recs.append({
                "prompt_idx": idx,
                "raw": raw_harm[idx],
                "refused": refused,
            })

        # behaviour direction (if enough refused / complied)
        n_ref = sum(r["refused"] for r in harmful_recs)
        n_com = len(harmful_recs) - n_ref
        v_beh = None
        v_unit = None
        if n_ref >= N_MIN_REFUSED and n_com >= N_MIN_COMPLIED:
            ref_acts = np.array([r["raw"] for r in harmful_recs if r["refused"]])
            com_acts = np.array([r["raw"] for r in harmful_recs if not r["refused"]])
            v_beh = ref_acts.mean(0) - com_acts.mean(0)
            nrm = np.linalg.norm(v_beh)
            if nrm > 1e-9:
                v_unit = v_beh / nrm

        out[m] = {
            "best_pos": bp,
            "best_layer": bl,
            "unit_arditi": unit_arditi,
            "harmful_recs": harmful_recs,
            "n_refused_harm": n_ref,
            "n_comply_harm": n_com,
            "v_unit_behaviour": v_unit,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", default=["3B", "7B"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = args.out_dir or f"results/behaviour_transfer_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"[behaviour-transfer] out={out_dir}")

    summary_all = {}
    for size in args.sizes:
        print(f"\n=== {size} ===")
        per_model = load_per_model(size)
        if not per_model:
            print(f"  [{size}] no models found, skip")
            continue
        for m, info in per_model.items():
            v_status = "v defined" if info["v_unit_behaviour"] is not None else "no v (too few refused)"
            print(f"  [{size}/{m}] refused/comply harm = {info['n_refused_harm']}/{info['n_comply_harm']}  →  {v_status}")

        donors = [m for m, info in per_model.items() if info["v_unit_behaviour"] is not None]
        recipients = list(per_model.keys())
        print(f"  donors: {donors}")
        print(f"  recipients: {recipients}")

        # 1. cross-cosine matrix
        cos_mat = np.full((len(donors), len(donors)), np.nan)
        for i, da in enumerate(donors):
            for j, db in enumerate(donors):
                va = per_model[da]["v_unit_behaviour"]
                vb = per_model[db]["v_unit_behaviour"]
                cos_mat[i, j] = float(va @ vb)

        # 2. transfer AUC matrix (donor → recipient)
        auc_mat = np.full((len(donors), len(recipients)), np.nan)
        for i, da in enumerate(donors):
            v = per_model[da]["v_unit_behaviour"]
            for j, rb in enumerate(recipients):
                recs = per_model[rb]["harmful_recs"]
                if len(recs) < 5:
                    continue
                y = np.array([1 if r["refused"] else 0 for r in recs])
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                X = np.array([r["raw"] for r in recs])
                proj = X @ v
                try:
                    auc_mat[i, j] = roc_auc_score(y, proj)
                except ValueError:
                    pass

        # 3. cos with Arditi direction per donor (sanity vs v1 finding)
        cos_arditi = {}
        for m in donors:
            v = per_model[m]["v_unit_behaviour"]
            a = per_model[m]["unit_arditi"]
            cos_arditi[m] = float(v @ a)

        # 4. AZR distribution shift (or any "recipient-only" model)
        non_donor_models = [m for m in recipients if m not in donors]

        summary = {
            "size": size,
            "donors": donors,
            "recipients": recipients,
            "non_donors": non_donor_models,
            "cross_cosine_matrix": {
                "rows": donors, "cols": donors,
                "values": cos_mat.tolist(),
            },
            "transfer_auc_matrix": {
                "rows_donor": donors, "cols_recipient": recipients,
                "values": auc_mat.tolist(),
            },
            "cos_arditi_x_behaviour_per_donor": cos_arditi,
            "n_refused_harmful_per_model": {
                m: info["n_refused_harm"] for m, info in per_model.items()
            },
            "n_comply_harmful_per_model": {
                m: info["n_comply_harm"] for m, info in per_model.items()
            },
        }

        # plotting helpers
        plot_cosine_heatmap(cos_mat, donors, size, out_dir)
        plot_transfer_auc_heatmap(auc_mat, donors, recipients, size, out_dir)
        if non_donor_models:
            plot_recipient_only_distribution(per_model, donors, non_donor_models, size, out_dir)

        # save raw arrays for reproducibility
        np.savez(
            os.path.join(out_dir, f"behaviour_arrays_{size}.npz"),
            **{f"{m}__v_unit_behaviour": per_model[m]["v_unit_behaviour"]
               for m in donors},
            **{f"{m}__unit_arditi": per_model[m]["unit_arditi"]
               for m in per_model},
            cross_cosine_matrix=cos_mat,
            transfer_auc_matrix=auc_mat,
            donors=np.array(donors),
            recipients=np.array(recipients),
        )

        with open(os.path.join(out_dir, f"summary_{size}.json"), "w") as fh:
            json.dump(summary, fh, indent=2)

        summary_all[size] = summary

    write_markdown(summary_all, out_dir)
    print(f"\nDone → {out_dir}")


def plot_cosine_heatmap(cos_mat, donors, size, out_dir):
    n = len(donors)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(0.7 * n + 3, 0.7 * n + 2))
    im = ax.imshow(cos_mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(donors, rotation=30, ha="right")
    ax.set_yticklabels(donors)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cos_mat[i,j]:.2f}", ha="center", va="center",
                    color="black" if abs(cos_mat[i,j]) < 0.6 else "white",
                    fontsize=9)
    ax.set_title(f"{size}: cos(behaviour_direction_A, behaviour_direction_B)")
    fig.colorbar(im, ax=ax, label="cosine")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_cos_matrix_{size}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_transfer_auc_heatmap(auc_mat, donors, recipients, size, out_dir):
    nr, nc = auc_mat.shape
    if nr == 0 or nc == 0:
        return
    fig, ax = plt.subplots(figsize=(0.8 * nc + 3, 0.7 * nr + 2))
    im = ax.imshow(auc_mat, vmin=0.3, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(nc)); ax.set_yticks(range(nr))
    ax.set_xticklabels(recipients, rotation=30, ha="right")
    ax.set_yticklabels(donors)
    ax.set_xlabel("recipient (whose prompts)")
    ax.set_ylabel("donor (whose direction)")
    for i in range(nr):
        for j in range(nc):
            v = auc_mat[i, j]
            if not np.isnan(v):
                color = "white" if v < 0.6 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
    ax.set_title(f"{size}: AUC(donor's behaviour direction predicts recipient's refusal)\n"
                 "diagonal = in-sample upper bound", fontsize=10)
    fig.colorbar(im, ax=ax, label="AUC")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"fig_transfer_auc_{size}.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recipient_only_distribution(per_model, donors, recipients_only, size, out_dir):
    """For each non-donor model M, plot its projection distribution on each
    donor's v alongside the donor's own (refused vs comply) on the same axis.
    Lets us see whether AZR's raw activations sit in the donor's refuse or
    comply region.
    """
    if not donors:
        return
    for recipient in recipients_only:
        recs = per_model[recipient]["harmful_recs"]
        if not recs:
            continue
        X_R = np.array([r["raw"] for r in recs])
        y_R = np.array([1 if r["refused"] else 0 for r in recs])

        nd = len(donors)
        fig, axes = plt.subplots(1, nd, figsize=(3.6 * nd, 3.6), sharey=False)
        if nd == 1:
            axes = [axes]
        for ax, da in zip(axes, donors):
            v = per_model[da]["v_unit_behaviour"]
            # donor's own projections
            d_recs = per_model[da]["harmful_recs"]
            d_proj_ref = np.array([r["raw"] @ v for r in d_recs if r["refused"]])
            d_proj_com = np.array([r["raw"] @ v for r in d_recs if not r["refused"]])
            # recipient's projections
            r_proj = X_R @ v
            r_proj_ref = r_proj[y_R == 1]
            r_proj_com = r_proj[y_R == 0]

            all_vals = np.concatenate([d_proj_ref, d_proj_com, r_proj])
            lo, hi = float(all_vals.min()), float(all_vals.max())
            pad = 0.05 * (hi - lo + 1e-6)
            bins = np.linspace(lo - pad, hi + pad, 18)

            ax.hist(d_proj_com, bins=bins, alpha=0.4, color="tab:green",
                    label=f"{da}: comply (n={len(d_proj_com)})")
            ax.hist(d_proj_ref, bins=bins, alpha=0.5, color="tab:red",
                    label=f"{da}: refuse (n={len(d_proj_ref)})")
            ax.hist(r_proj_com, bins=bins, alpha=0.4, color="tab:cyan",
                    histtype="step", linewidth=2,
                    label=f"{recipient}: comply (n={len(r_proj_com)})")
            if len(r_proj_ref):
                ax.axvline(float(r_proj_ref[0]), color="tab:orange",
                           lw=2.5, ls="--",
                           label=f"{recipient}: refuse (n={len(r_proj_ref)})")
            ax.axvline(float(d_proj_com.mean()), color="darkgreen", lw=1, ls=":")
            ax.axvline(float(d_proj_ref.mean()), color="darkred", lw=1, ls=":")
            ax.set_title(f"donor v = {da}", fontsize=9)
            ax.set_xlabel("projection on donor v")
            ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(f"{size}: {recipient} harmful prompts projected onto each donor's behaviour direction",
                     y=1.02, fontsize=11)
        fig.tight_layout()
        safe = recipient.replace("/", "_")
        fig.savefig(os.path.join(out_dir, f"fig_recipient_{safe}_{size}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)


def write_markdown(summary_all, out_dir):
    lines = [
        "# Cross-model behaviour-direction transfer",
        "",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_",
        "",
        ("For each model with ≥3 refused AND ≥3 complied harmful prompts, "
         "compute the within-harmful behaviour direction "
         "`v_M = mean(raw|refuse) − mean(raw|comply)`. Then look at how "
         "these directions relate across models, and whether one model's "
         "direction predicts another's refusal."),
        "",
    ]

    for size, s in summary_all.items():
        if not s:
            continue
        lines += [f"## {size}", ""]
        lines += [
            "**Models with behaviour direction defined** "
            f"(≥{N_MIN_REFUSED} refused, ≥{N_MIN_COMPLIED} complied harmful):",
            "",
        ]
        for d in s["donors"]:
            n_r = s["n_refused_harmful_per_model"][d]
            n_c = s["n_comply_harmful_per_model"][d]
            lines.append(f"- {d}: refused={n_r}, comply={n_c}")
        if s["non_donors"]:
            lines += ["", "**Models with too few refusals (recipient-only):**", ""]
            for d in s["non_donors"]:
                n_r = s["n_refused_harmful_per_model"][d]
                n_c = s["n_comply_harmful_per_model"][d]
                lines.append(f"- {d}: refused={n_r}, comply={n_c}")
        lines += ["", "### Cross-cosine matrix `cos(v_A, v_B)`", ""]
        rows = s["cross_cosine_matrix"]["rows"]
        cols = s["cross_cosine_matrix"]["cols"]
        vals = s["cross_cosine_matrix"]["values"]
        lines.append("| | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for r, row_vals in zip(rows, vals):
            lines.append("| " + r + " | " + " | ".join(
                f"{v:.2f}" if v is not None else "—" for v in row_vals) + " |")
        lines += ["", "### Transfer AUC matrix (donor v predicting recipient's harmful refusal)", ""]
        rows = s["transfer_auc_matrix"]["rows_donor"]
        cols = s["transfer_auc_matrix"]["cols_recipient"]
        vals = s["transfer_auc_matrix"]["values"]
        lines.append("| donor \\ recipient | " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 1))
        for r, row_vals in zip(rows, vals):
            cells = []
            for v in row_vals:
                cells.append(f"{v:.2f}" if v is not None and not (isinstance(v, float) and v != v) else "—")
            lines.append("| " + r + " | " + " | ".join(cells) + " |")
        lines += ["", "### cos(Arditi DiM, behaviour direction) per donor — sanity check vs v1/v2", ""]
        for m, c in s["cos_arditi_x_behaviour_per_donor"].items():
            lines.append(f"- {m}: {c:.3f}")
        lines += ["", ""]

    lines += [
        "## How to read",
        "",
        "**Cross-cosine matrix**:",
        "- High off-diagonal (≥0.5) = models share a behaviour axis. Self-evolving / RLHF preserves the axis even if they rotate prompt-type geometry differently.",
        "- Low off-diagonal (<0.3) = each model has its own private behaviour mediator.",
        "- Watch for: cos(Coder, AZR) >> cos(Coder, RLHF). That would say self-evolving keeps the behaviour axis Coder already had, while RLHF rotates it.",
        "",
        "**Transfer AUC matrix**:",
        "- Diagonal ≈ in-sample upper bound (donor's own AUC on its own prompts). Inflated by fitting.",
        "- Off-diagonal close to diagonal = direction transfers cleanly.",
        "- Off-diagonal near 0.5 = direction is private.",
        "",
        "**Recipient-only distributions** (AZR-Coder-3B + Qwen2.5-7B until 7B gens finish):",
        "- If the recipient's complied prompts cluster on the donor's *comply* mean → activations look like the donor's compliance pattern.",
        "- If the recipient's complied prompts shifted toward the donor's *refuse* mean → activations 'almost wanted to refuse' but generation didn't. Suggests downstream decoupling.",
        "- If the 1 refused AZR prompt is far below the donor's refuse cluster → it's an atypical refusal (e.g. only because the prompt was easy to refuse).",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
