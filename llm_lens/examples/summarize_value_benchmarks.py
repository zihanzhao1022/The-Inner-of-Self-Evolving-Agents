#!/usr/bin/env python3
"""Render a value-benchmark results.json into a paper-style markdown table.

Usage:
    python -m llm_lens.examples.summarize_value_benchmarks results/value_benchmarks_<TS>/results.json

Writes `summary_table.md` next to the input JSON, and prints the same to stdout.
The table includes Δ-vs-Base columns so "drift relative to pretraining" is
visible at a glance, which is the whole point of the comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import sys


PRESENTATION = {
    "truthfulqa_mc1": {
        "label":      "TruthfulQA MC1",
        "metric_key": "accuracy",
        "direction":  "↑",
        "format":     "{:.3f}",
        "delta_fmt":  "{:+.3f}",
    },
    "bbq_ambig": {
        "label":      "BBQ ambig acc",
        "metric_key": "accuracy",
        "direction":  "↑",
        "format":     "{:.3f}",
        "delta_fmt":  "{:+.3f}",
    },
    "advbench_compliance": {
        "label":      "AdvBench logP(comply)",
        "metric_key": "mean_logprob_target",
        "direction":  "↓ (less = more aligned)",
        "format":     "{:.3f}",
        "delta_fmt":  "{:+.3f}",
    },
}

MODEL_ORDER = [
    "Qwen2.5-3B",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-Coder-3B",
    "AZR-Coder-3B",
    # 7B (for future runs)
    "Qwen2.5-7B",
    "Qwen2.5-7B-Instruct",
    "Qwen2.5-Coder-7B",
    "AZR-Base-7B",
    "AZR-Coder-7B",
]
PRETTY_NAME = {
    "Qwen2.5-3B":          "Qwen2.5-3B (base)",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B-Instruct (RLHF)",
    "Qwen2.5-Coder-3B":    "Qwen2.5-Coder-3B (domain)",
    "AZR-Coder-3B":        "AZR-Coder-3B (self-evolved)",
    "Qwen2.5-7B":          "Qwen2.5-7B (base)",
    "Qwen2.5-7B-Instruct": "Qwen2.5-7B-Instruct (RLHF)",
    "Qwen2.5-Coder-7B":    "Qwen2.5-Coder-7B (domain)",
    "AZR-Base-7B":         "AZR-Base-7B (self-evolved direct)",
    "AZR-Coder-7B":        "AZR-Coder-7B (self-evolved via Coder)",
}
BASE_KEY = "Qwen2.5-3B"


def render_table(results: dict) -> str:
    meta = results.get("_meta", {})
    benchmarks = list(meta.get("benchmarks") or [
        k for k in next(iter(v for k, v in results.items()
                              if not k.startswith("_"))).keys()
    ])
    targets = [t for t in MODEL_ORDER if t in results]
    if not targets:
        return "(no models in results)"

    base_metrics = {}
    if BASE_KEY in results:
        for b in benchmarks:
            r = results[BASE_KEY].get(b, {})
            mk = PRESENTATION.get(b, {}).get("metric_key", "accuracy")
            base_metrics[b] = r.get(mk)

    out = []

    # ── Top-level table ─────────────────────────────────────────────────
    n_items = []
    for b in benchmarks:
        any_n = None
        for t in targets:
            r = results.get(t, {}).get(b, {})
            if r:
                any_n = r.get("n_items"); break
        n_items.append(any_n)

    out.append("## Value-benchmark comparison\n")
    n_str = ", ".join(f"{b}: n={n}" for b, n in zip(benchmarks, n_items))
    out.append(f"_Sample sizes — {n_str}._\n")

    # Markdown header
    headers = ["Model"]
    for b in benchmarks:
        p = PRESENTATION.get(b, {})
        headers.append(f"{p.get('label', b)} {p.get('direction','')}")
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")

    for t in targets:
        cells = [PRETTY_NAME.get(t, t)]
        for b in benchmarks:
            p = PRESENTATION.get(b, {"metric_key": "accuracy",
                                     "format": "{:.3f}", "delta_fmt": "{:+.3f}"})
            r = results[t].get(b, {})
            v = r.get(p["metric_key"])
            if v is None:
                cells.append("—"); continue
            txt = p["format"].format(v)
            if t != BASE_KEY and base_metrics.get(b) is not None:
                d = v - base_metrics[b]
                txt += f" ({p['delta_fmt'].format(d)})"
            cells.append(txt)
        out.append("| " + " | ".join(cells) + " |")

    # ── BBQ breakdown ───────────────────────────────────────────────────
    if "bbq_ambig" in benchmarks:
        cats = None
        for t in targets:
            r = results.get(t, {}).get("bbq_ambig", {})
            cats = r.get("by_category")
            if cats: break
        if cats:
            cat_names = list(cats.keys())
            out.append("\n### BBQ by category (accuracy on ambiguous = picks 'unknown' option)\n")
            out.append("| Model | " + " | ".join(cat_names) + " |")
            out.append("|" + "|".join(["---"] * (len(cat_names) + 1)) + "|")

            base_by_cat = {}
            if BASE_KEY in results:
                base_by_cat = results[BASE_KEY].get("bbq_ambig", {}).get("by_category", {})

            for t in targets:
                cs = results.get(t, {}).get("bbq_ambig", {}).get("by_category", {})
                row = [PRETTY_NAME.get(t, t)]
                for cn in cat_names:
                    v = cs.get(cn)
                    if v is None:
                        row.append("—"); continue
                    cell = f"{v:.3f}"
                    if t != BASE_KEY and base_by_cat.get(cn) is not None:
                        d = v - base_by_cat[cn]
                        cell += f" ({d:+.3f})"
                    row.append(cell)
                out.append("| " + " | ".join(row) + " |")

    # ── Interpretation hints ────────────────────────────────────────────
    out.append("\n### How to read this\n")
    out.append("* **Δ in parentheses = vs Base.** Positive Δ on TruthfulQA / BBQ means improvement; negative Δ on AdvBench logP means *less* willing to comply with harmful requests (i.e., better alignment).")
    out.append("* **TruthfulQA MC1**: random baseline ≈ 0.25 (4-way), human ≈ 0.94. Higher = more honest / fewer common-misconception answers.")
    out.append("* **BBQ ambig**: random baseline ≈ 0.33 (3-way). Gold answer is 'cannot be determined'. Higher = less prone to stereotype-aligned guessing.")
    out.append("* **AdvBench logP**: per-token mean log-probability the model assigns to a *compliant continuation* of a harmful request. Lower (more negative) = the compliant continuation is more surprising to the model = more aligned.")

    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results_json", help="Path to results.json from run_value_benchmarks")
    p.add_argument("--out", default=None,
                   help="Output markdown path (default: <input_dir>/summary_table.md)")
    args = p.parse_args()

    with open(args.results_json, encoding="utf-8") as f:
        results = json.load(f)

    md = render_table(results)
    print(md)

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.results_json)), "summary_table.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(f"\n→ written to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
