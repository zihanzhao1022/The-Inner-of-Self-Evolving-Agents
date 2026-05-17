#!/bin/bash
# run_harmbench_idx0.sh — re-run idx 0-49 across 3 GPUs (3, 4, 6).
# Same chain structure as run_harmbench_batches.sh (verified working overnight).
# Writes to generations_3B_idx0_49/ and generations_7B_idx0_49/.

set -u

PROJECT=$HOME/The-Inner-of-Self-Evolving-Agents
RESULTS=$PROJECT/results/harmbench_eval_20260515-1512
LOG_DIR=$HOME/eval_logs

mkdir -p "$LOG_DIR"

# Bypass conda activate (slow under heavy system load) — use py310's python directly
export PATH=$HOME/miniconda3/envs/py310/bin:$PATH
cd "$PROJECT"

DRIVER_LOG=$LOG_DIR/driver_idx0.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

run_targets() {
  local gpu=$1 mset=$2 dtype=$3 odir=$4 logsuf=$5
  shift 5
  local targets="$*"
  local logf=$LOG_DIR/gpu${gpu}_${logsuf}.log
  log "  [GPU${gpu}] -> ${mset}/${dtype} targets={${targets}}"
  CUDA_VISIBLE_DEVICES=$gpu python -m llm_lens.examples.run_generations \
      --model-set $mset --dtype $dtype \
      --template-mode auto --max-new-tokens 1024 \
      --prompts-file "$RESULTS/prompts_idx0.json" --output-dir "$odir" \
      --targets $targets \
      >> "$logf" 2>&1
  local rc=$?
  log "  [GPU${gpu}] <- ${mset}/${dtype} targets={${targets}} rc=${rc}"
  return $rc
}

PFILE=$RESULTS/prompts_idx0.json
GEN3B=$RESULTS/generations_3B_idx0_49
GEN7B=$RESULTS/generations_7B_idx0_49

log "============================================================"
log " IDX 0-49 RERUN — 3 GPUs (3, 4, 6)"
log "============================================================"

if [ ! -f "$PFILE" ]; then
    log "  building prompts: $PFILE"
    python -m llm_lens.examples.build_harmbench_eval_prompts \
        --start-idx 0 --n-harmful 50 --out "$PFILE" >> "$DRIVER_LOG" 2>&1
fi

# GPU 3 chain: 3B-base + 7B-AZR pair
(
    run_targets 3 3B float32  "$GEN3B" idx0_3B_base  Qwen2.5-3B
    run_targets 3 7B bfloat16 "$GEN7B" idx0_7B_azr   AZR-Base-7B AZR-Coder-7B
) &
PID3=$!

# GPU 4 chain: 3B-Coder + 7B-Coder
(
    run_targets 4 3B float32  "$GEN3B" idx0_3B_coder Qwen2.5-Coder-3B
    run_targets 4 7B bfloat16 "$GEN7B" idx0_7B_coder Qwen2.5-Coder-7B
) &
PID4=$!

# GPU 6 chain: 7B-base + 3B-Inst+AZR + 7B-Inst
(
    run_targets 6 7B bfloat16 "$GEN7B" idx0_7B_base  Qwen2.5-7B
    run_targets 6 3B float32  "$GEN3B" idx0_3B_inst  Qwen2.5-3B-Instruct AZR-Coder-3B
    run_targets 6 7B bfloat16 "$GEN7B" idx0_7B_inst  Qwen2.5-7B-Instruct
) &
PID6=$!

log "  launched PIDs: GPU3=$PID3 GPU4=$PID4 GPU6=$PID6"
echo "GPU3=$PID3 GPU4=$PID4 GPU6=$PID6" > "$LOG_DIR/active_pids_idx0.txt"

wait $PID3 $PID4 $PID6
log "============================================================"
log " IDX 0-49 RERUN COMPLETE"
log "============================================================"
