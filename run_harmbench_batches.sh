#!/bin/bash
# run_harmbench_batches.sh
# Drives 3 HarmBench batches (idx 50/100/150) x {3B fp32, 7B bf16} x 3 GPUs.
# Each GPU runs a sequential chain of (model_set, --targets) combinations.
#
# Output layout (inside results/harmbench_eval_20260515-1512/):
#   prompts_idx50.json, prompts_idx100.json, prompts_idx150.json
#   generations_3B_idx50_99/<model>/generations.jsonl
#   generations_7B_idx50_99/<model>/generations.jsonl
#   ... etc for idx100_149, idx150_199.

set -u  # not -e: a failed model should not kill the whole batch

PROJECT=$HOME/The-Inner-of-Self-Evolving-Agents
RESULTS=$PROJECT/results/harmbench_eval_20260515-1512
LOG_DIR=$HOME/eval_logs
mkdir -p "$LOG_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate py310
cd "$PROJECT"

DRIVER_LOG=$LOG_DIR/driver.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

run_targets() {
  # run_targets <gpu> <model_set> <dtype> <prompts_file> <output_dir> <log_suffix> <targets...>
  local gpu=$1 mset=$2 dtype=$3 pfile=$4 odir=$5 logsuf=$6
  shift 6
  local targets="$*"
  local logf=$LOG_DIR/gpu${gpu}_${logsuf}.log
  log "  [GPU${gpu}] -> ${mset}/${dtype} targets={${targets}} log=${logf}"
  CUDA_VISIBLE_DEVICES=$gpu python -m llm_lens.examples.run_generations \
      --model-set $mset --dtype $dtype \
      --template-mode auto --max-new-tokens 1024 \
      --prompts-file "$pfile" --output-dir "$odir" \
      --targets $targets \
      >> "$logf" 2>&1
  local rc=$?
  log "  [GPU${gpu}] <- ${mset}/${dtype} targets={${targets}} rc=${rc}"
  return $rc
}

for IDX in 50 100 150; do
  END=$((IDX + 49))
  PFILE=$RESULTS/prompts_idx${IDX}.json
  GEN3B=$RESULTS/generations_3B_idx${IDX}_${END}
  GEN7B=$RESULTS/generations_7B_idx${IDX}_${END}

  log "============================================================"
  log " BATCH idx=${IDX}-${END}"
  log "============================================================"

  if [ ! -f "$PFILE" ]; then
    log "  building prompts: $PFILE"
    python -m llm_lens.examples.build_harmbench_eval_prompts \
        --start-idx $IDX --n-harmful 50 --out "$PFILE" >> "$DRIVER_LOG" 2>&1
  else
    log "  prompts already exist: $PFILE"
  fi

  # GPU 3 chain: heavy 3B-base, then 7B AZR pair
  (
    run_targets 3 3B float32  "$PFILE" "$GEN3B" idx${IDX}_3B_base   Qwen2.5-3B
    run_targets 3 7B bfloat16 "$PFILE" "$GEN7B" idx${IDX}_7B_azr    AZR-Base-7B AZR-Coder-7B
  ) &
  PID3=$!

  # GPU 4 chain: 3B-Coder, then 7B-Coder
  (
    run_targets 4 3B float32  "$PFILE" "$GEN3B" idx${IDX}_3B_coder  Qwen2.5-Coder-3B
    run_targets 4 7B bfloat16 "$PFILE" "$GEN7B" idx${IDX}_7B_coder  Qwen2.5-Coder-7B
  ) &
  PID4=$!

  # GPU 6 chain: 7B-base, 3B-Instruct+AZR, 7B-Instruct
  (
    run_targets 6 7B bfloat16 "$PFILE" "$GEN7B" idx${IDX}_7B_base   Qwen2.5-7B
    run_targets 6 3B float32  "$PFILE" "$GEN3B" idx${IDX}_3B_inst   Qwen2.5-3B-Instruct AZR-Coder-3B
    run_targets 6 7B bfloat16 "$PFILE" "$GEN7B" idx${IDX}_7B_inst   Qwen2.5-7B-Instruct
  ) &
  PID6=$!

  log "  launched PIDs: GPU3=$PID3 GPU4=$PID4 GPU6=$PID6"
  echo "GPU3=$PID3 GPU4=$PID4 GPU6=$PID6 IDX=$IDX" >> "$LOG_DIR/active_pids.txt"

  wait $PID3 $PID4 $PID6
  log "  BATCH idx=${IDX}-${END} ALL CHAINS COMPLETE"
done

log "============================================================"
log " ALL THREE BATCHES DONE"
log "============================================================"
