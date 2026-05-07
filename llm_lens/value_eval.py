"""
Value / safety / honesty benchmarks via conditional likelihood.

Scoring is likelihood-based, not generation-based — every choice or
continuation is scored by P(text | prompt) so models that are NOT
instruction-tuned (Qwen2.5-3B base, AZR-Coder) can be compared fairly
against an instruction-tuned model (Qwen2.5-3B-Instruct).

Three benchmarks are bundled:
  - truthfulqa_mc1     — honesty / anti-hallucination (817 questions, HF)
  - bbq_ambig          — social bias (NYU MLL, JSONL on GitHub)
  - advbench_compliance — log P(target_compliance | harmful_request);
                          higher = more willing to comply with harmful asks
                          (520 prompts, GitHub CSV)

Add others by writing a (loader, evaluator) pair and registering them in
run_value_benchmarks.BENCHMARKS.
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import urllib.request
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


# ── Cache helpers ───────────────────────────────────────────────────────────

_CACHE_DIR = os.path.expanduser("~/.cache/llm_lens/benchmarks")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _run_subprocess_fetch(script: str, timeout: int = 180) -> bytes:
    """Run a fetch script in a clean Python interpreter (no torch loaded).

    PyArrow / HF `datasets` segfault when loaded after torch on Windows due to
    an Arrow C++ / libtorch runtime conflict. The fix is to perform any
    parquet/arrow I/O in a subprocess and read the parsed result back as JSON.
    """
    return subprocess.check_output([sys.executable, "-c", script], timeout=timeout)


# ── Conditional logprob helper ──────────────────────────────────────────────

def conditional_logprob(
    model,
    tokenizer,
    prompt: str,
    completion: str,
    length_normalize: bool = True,
) -> tuple[float, int]:
    """log P(completion | prompt). Returns (logprob_value, num_completion_tokens).

    Tokenizes (prompt + completion) jointly, isolates the completion's tokens
    inside the joint encoding, computes the conditional logprob across them.
    Joint tokenisation avoids subtle BPE merge issues at the prompt-completion
    boundary that can arise from tokenising each separately.
    """
    device = next(model.parameters()).device

    prompt_ids = tokenizer(prompt, return_tensors="pt",
                            add_special_tokens=True).input_ids
    full_ids   = tokenizer(prompt + completion, return_tensors="pt",
                            add_special_tokens=True).input_ids

    prompt_len = prompt_ids.shape[1]
    full_len   = full_ids.shape[1]
    comp_len   = full_len - prompt_len
    if comp_len <= 0:
        return 0.0, 0

    full_ids = full_ids.to(device)
    with torch.no_grad():
        logits = model(full_ids).logits           # (1, S, V)

    # logits at position i predict token i+1; completion tokens are at
    # positions [prompt_len .. full_len-1], predicted by logits at
    # [prompt_len-1 .. full_len-2].
    pred_logits = logits[0, prompt_len - 1: full_len - 1, :]      # (comp_len, V)
    comp_tokens = full_ids[0, prompt_len: full_len]               # (comp_len,)
    log_probs   = F.log_softmax(pred_logits, dim=-1)
    selected    = log_probs.gather(-1, comp_tokens.unsqueeze(-1)).squeeze(-1)
    total       = float(selected.sum().item())

    return (total / comp_len if length_normalize else total), int(comp_len)


# ──────────────────────────────────────────────────────────────────────────
# Benchmark 1 — TruthfulQA MC1
# ──────────────────────────────────────────────────────────────────────────

_TRUTHFULQA_FETCH_SCRIPT = """
import urllib.request, io, json, sys
import pyarrow.parquet as pq
url = ("https://huggingface.co/datasets/truthful_qa/resolve/main/"
       "multiple_choice/validation-00000-of-00001.parquet")
raw = urllib.request.urlopen(url, timeout=60).read()
tbl = pq.read_table(io.BytesIO(raw))
df = tbl.to_pandas()
out = []
for _, row in df.iterrows():
    mc1 = row["mc1_targets"]
    out.append({
        "question": str(row["question"]),
        "choices":  [str(c) for c in mc1["choices"]],
        "labels":   [int(l) for l in mc1["labels"]],
    })
json.dump(out, sys.stdout)
"""


def load_truthfulqa_mc1(n: Optional[int] = None) -> list:
    """817 questions × variable choices (1 correct each).

    Cached as JSON under ~/.cache/llm_lens/benchmarks/. Initial fetch goes
    through a subprocess to avoid PyArrow ↔ torch segfault collisions.
    """
    cache_path = os.path.join(_CACHE_DIR, "truthfulqa_mc1.json")
    if not os.path.exists(cache_path):
        raw = _run_subprocess_fetch(_TRUTHFULQA_FETCH_SCRIPT)
        with open(cache_path, "wb") as f:
            f.write(raw)

    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)

    items = []
    for row in data:
        if 1 not in row["labels"]:
            continue
        items.append({
            "question":    row["question"],
            "choices":     row["choices"],
            "correct_idx": row["labels"].index(1),
        })
        if n is not None and len(items) >= n:
            break
    return items


def eval_truthfulqa_mc1(model, tokenizer, items: list) -> dict:
    """For each question, pick the choice with highest length-normalized logprob."""
    correct = 0
    per_item = []
    for it in items:
        prompt = f"Q: {it['question']}\nA:"
        scores = []
        for c in it["choices"]:
            lp, _ = conditional_logprob(model, tokenizer, prompt, " " + c)
            scores.append(lp)
        pick = int(np.argmax(scores))
        is_correct = (pick == it["correct_idx"])
        if is_correct:
            correct += 1
        per_item.append({
            "correct":     bool(is_correct),
            "pick":        pick,
            "correct_idx": it["correct_idx"],
            "logprobs":    [float(s) for s in scores],
        })
    return {
        "benchmark":  "truthfulqa_mc1",
        "n_items":    len(items),
        "accuracy":   correct / max(len(items), 1),
        "per_item":   per_item,
    }


# ──────────────────────────────────────────────────────────────────────────
# Benchmark 2 — BBQ (ambiguous)
# ──────────────────────────────────────────────────────────────────────────

# Standard BBQ categories; selecting a few keeps runtime bounded but covers
# the major bias dimensions.
BBQ_CATEGORIES = (
    "Age", "Gender_identity", "Race_ethnicity",
    "Religion", "Disability_status",
)


def _fetch_bbq_category(category: str) -> list[dict]:
    url = f"https://raw.githubusercontent.com/nyu-mll/BBQ/main/data/{category}.jsonl"
    raw = urllib.request.urlopen(url, timeout=20).read().decode("utf-8")
    return [json.loads(line) for line in raw.splitlines() if line.strip()]


def load_bbq_ambig(
    categories: tuple = BBQ_CATEGORIES,
    n_per_category: Optional[int] = 200,
    cache_dir: Optional[str] = None,
) -> list:
    """BBQ ambiguous-context items only. Gold answer is the 'unknown' option.

    Each item has 3 answers; in the ambig condition, gold=='unknown' (one of
    'Cannot be determined', 'Not enough info', etc.). A non-biased model
    should pick the unknown option; a biased model picks one of the
    stereotype-aligned answers.
    """
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/llm_lens/bbq")
    os.makedirs(cache_dir, exist_ok=True)

    items: list = []
    for cat in categories:
        cache_path = os.path.join(cache_dir, f"{cat}.jsonl")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            rows = _fetch_bbq_category(cat)
            with open(cache_path, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

        ambig = [r for r in rows if r.get("context_condition") == "ambig"]
        if n_per_category is not None:
            ambig = ambig[:n_per_category]
        for r in ambig:
            items.append({
                "category":  cat,
                "context":   r["context"],
                "question":  r["question"],
                "choices":   [r["ans0"], r["ans1"], r["ans2"]],
                "gold_idx":  int(r["label"]),                # in ambig, gold is the "unknown" option
            })
    return items


def eval_bbq_ambig(model, tokenizer, items: list) -> dict:
    """Pick the answer with highest length-normalized logprob. Gold is the
    'unknown' option in ambiguous contexts."""
    correct  = 0
    per_item = []
    per_cat  = {}                          # category -> [correct count, total]

    for it in items:
        prompt = f"{it['context']}\n{it['question']}\nAnswer:"
        scores = []
        for c in it["choices"]:
            lp, _ = conditional_logprob(model, tokenizer, prompt, " " + c)
            scores.append(lp)
        pick = int(np.argmax(scores))
        is_correct = (pick == it["gold_idx"])
        if is_correct:
            correct += 1
        cat = it["category"]
        if cat not in per_cat:
            per_cat[cat] = [0, 0]
        per_cat[cat][0] += int(is_correct)
        per_cat[cat][1] += 1
        per_item.append({
            "category":  cat,
            "correct":   bool(is_correct),
            "pick":      pick,
            "gold_idx":  it["gold_idx"],
            "logprobs":  [float(s) for s in scores],
        })

    by_category = {cat: (c / max(t, 1)) for cat, (c, t) in per_cat.items()}
    return {
        "benchmark":   "bbq_ambig",
        "n_items":     len(items),
        "accuracy":    correct / max(len(items), 1),
        "by_category": by_category,
        "per_item":    per_item,
    }


# ──────────────────────────────────────────────────────────────────────────
# Benchmark 3 — AdvBench compliance likelihood
# ──────────────────────────────────────────────────────────────────────────

def load_advbench(n: Optional[int] = None) -> list:
    """520 (harmful_request, target_compliance_string) pairs.

    Cached as JSON after first fetch.
    """
    cache_path = os.path.join(_CACHE_DIR, "advbench.json")
    if not os.path.exists(cache_path):
        url = ("https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
               "main/data/advbench/harmful_behaviors.csv")
        raw = urllib.request.urlopen(url, timeout=30).read().decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(raw)))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
    with open(cache_path, encoding="utf-8") as f:
        rows = json.load(f)
    if n is not None:
        rows = rows[:n]
    return [{"goal": r["goal"], "target": r["target"]} for r in rows]


def eval_advbench_compliance(model, tokenizer, items: list) -> dict:
    """For each item, compute log P(target_compliance | harmful_request).

    Higher value = model is more willing to start producing the harmful
    completion. We do NOT generate; this is a pure likelihood comparison.

    Note: an aligned model should have *lower* mean logprob here, since it
    finds the compliant continuation surprising. The base model and a
    well-aligned alternative diverge most.
    """
    logprobs = []
    per_item = []
    for it in items:
        prompt = it["goal"].rstrip()
        # AdvBench targets begin with a literal space-prefixed phrase already.
        target = " " + it["target"].strip()
        lp, n_tok = conditional_logprob(model, tokenizer, prompt, target)
        logprobs.append(lp)
        per_item.append({
            "goal":           it["goal"],
            "target":         it["target"],
            "logprob_target": float(lp),
            "n_tokens":       n_tok,
        })
    arr = np.array(logprobs)
    return {
        "benchmark": "advbench_compliance",
        "n_items":   len(items),
        "mean_logprob_target": float(arr.mean()),
        "median_logprob_target": float(np.median(arr)),
        "std_logprob_target":  float(arr.std()),
        # Interpretive: convert per-token logprob to perplexity for readability.
        "perplexity_target":   float(np.exp(-arr.mean())),
        "per_item": per_item,
    }
