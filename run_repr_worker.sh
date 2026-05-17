#!/usr/bin/env bash
# ============================================================================
# Single-GPU worker for HarmBench representation re-run.
#
# Runs, for ONE model on ONE GPU:
#   1. phase-1 activation trajectory      (results/<short>/<ts>/)
#   2. Arditi refusal-direction (n=200)   (results/refusal_direction_<ts>/)
#
# Both use the harmbench200_alpaca200 paired dataset registered in
# llm_lens/datasets.py.
#
# Usage:
#   bash run_repr_worker.sh <gpu_id> <short_label> <ts> <root_dir>
#
# Example:
#   bash run_repr_worker.sh 1 Qwen2.5-3B 20260518-0100 \
#       results/harmbench_repr_20260518-0100
#
# Idempotent within a TS: on success touches <root>/_done/<short>.done.
# ============================================================================

set -u  # treat unset variables as errors
set -o pipefail

GPU_ID="$1"
SHORT="$2"
TS="$3"
ROOT="$4"

# Lookup model-set + HF path + dtype based on short label.
case "$SHORT" in
    Qwen2.5-3B)              HF="Qwen/Qwen2.5-3B";                            SET="3B"; DTYPE="float32" ;;
    Qwen2.5-3B-Instruct)     HF="Qwen/Qwen2.5-3B-Instruct";                   SET="3B"; DTYPE="float32" ;;
    Qwen2.5-Coder-3B)        HF="Qwen/Qwen2.5-Coder-3B";                      SET="3B"; DTYPE="float32" ;;
    AZR-Coder-3B)            HF="andrewzh/Absolute_Zero_Reasoner-Coder-3b";   SET="3B"; DTYPE="float32" ;;
    Qwen2.5-7B)              HF="Qwen/Qwen2.5-7B";                            SET="7B"; DTYPE="bfloat16" ;;
    Qwen2.5-7B-Instruct)     HF="Qwen/Qwen2.5-7B-Instruct";                   SET="7B"; DTYPE="bfloat16" ;;
    Qwen2.5-Coder-7B)        HF="Qwen/Qwen2.5-Coder-7B";                      SET="7B"; DTYPE="bfloat16" ;;
    AZR-Base-7B)             HF="andrewzh2/Absolute_Zero_Reasoner-Base-7b";   SET="7B"; DTYPE="bfloat16" ;;
    AZR-Coder-7B)            HF="andrewzh/Absolute_Zero_Reasoner-Coder-7b";   SET="7B"; DTYPE="bfloat16" ;;
    *)
        echo "[$(date)] FATAL: unknown short label '$SHORT'" >&2
        exit 64
        ;;
esac

LOG_DIR="$ROOT/_logs"
DONE_DIR="$ROOT/_done"
RUN_DIR="$ROOT/_running"
mkdir -p "$LOG_DIR" "$DONE_DIR" "$RUN_DIR"

LOG="$LOG_DIR/$SHORT.log"
DONE="$DONE_DIR/$SHORT.done"
RUN_TAG="$RUN_DIR/$SHORT.gpu$GPU_ID"

# Mark this job as running. Stays in place until the worker exits.
touch "$RUN_TAG"
trap "rm -f '$RUN_TAG'" EXIT

export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Activate the project's conda env. Use PATH-prepend (matches the pattern in
# run_harmbench_idx0.sh) — lighter than `conda activate` and avoids the
# `.bashrc` not-sourced problem on non-interactive ssh.
export PATH="$HOME/miniconda3/envs/py310/bin:$PATH"

# Move into repo root (worker is launched from the repo root by dispatcher,
# but be defensive in case it's invoked from elsewhere).
cd "$(dirname "$(readlink -f "$0")")"

{
    echo "============================================================"
    echo "[$(date)] START   short=$SHORT  HF=$HF  set=$SET  dtype=$DTYPE  GPU=$GPU_ID  ts=$TS"
    echo "============================================================"

    # --- Stage A: phase-1 trajectory (raw activations + class centroids) ---
    echo "[$(date)] === stage A: phase-1 trajectory ==="
    python -m llm_lens.examples.run_experiment \
        --phase 1 \
        --model "$HF" \
        --dataset harmbench200_alpaca200 \
        --max-per-class 200 \
        --timestamp "$TS" \
        --dtype "$DTYPE" \
        --output results \
        || { echo "[$(date)] STAGE A FAILED for $SHORT" >&2; exit 1; }

    # --- Stage B: Arditi refusal direction ---
    echo "[$(date)] === stage B: refusal direction (n=200, eoi_len=5) ==="
    python -m llm_lens.examples.run_refusal_direction \
        --dataset harmbench200_alpaca200 \
        --max-per-class 200 \
        --model-set "$SET" \
        --targets "$SHORT" \
        --dtype "$DTYPE" \
        --results-root results \
        --output-suffix "${TS}_harmbench_paired_${SET}" \
        || { echo "[$(date)] STAGE B FAILED for $SHORT" >&2; exit 2; }

    echo "[$(date)] DONE    short=$SHORT"
} >> "$LOG" 2>&1

touch "$DONE"
echo "[$(date)] worker $SHORT finished, marker $DONE written"
