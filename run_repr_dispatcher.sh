#!/usr/bin/env bash
# ============================================================================
# Dispatcher for HarmBench representation re-run.
#
# Strategy:
#   * 9 jobs (one per model), each running phase-1 + refusal-direction.
#   * Tries to use up to 4 GPUs in parallel. If at start-up fewer than 4
#     GPUs are free, we proceed with whatever is free (>=2 required).
#   * Even after that, the dispatcher KEEPS POLLING every POLL_INTERVAL
#     seconds: as soon as a previously-busy GPU frees up (e.g. someone's
#     overnight training ends), the dispatcher claims it for the next
#     queued model. So if you start with 2 GPUs and the cluster empties
#     at midnight, by morning you're running on 4.
#   * "Free" means util < UTIL_THRESH% AND free_memory >= MIN_FREE_MB
#     (default 5%, 17000 MB — fits a bf16 7B comfortably).
#
# Usage:
#   bash run_repr_dispatcher.sh                # uses defaults
#   MAX_PARALLEL=3 bash run_repr_dispatcher.sh # cap at 3 GPUs
#   MIN_FREE_MB=20000 bash run_repr_dispatcher.sh
#
# Output:
#   results/harmbench_repr_<ts>/
#     _queue.txt          remaining model labels
#     _running/<m>.gpu<n> active job markers
#     _done/<m>.done       success markers
#     _logs/<m>.log        per-model stdout/stderr
#     _dispatcher.log      dispatcher's own status log
# ============================================================================

set -u
set -o pipefail

# -------- Tunables (env overrides) --------
MAX_PARALLEL="${MAX_PARALLEL:-4}"          # target ceiling (4 → 3 → 2)
MIN_PARALLEL_TO_START="${MIN_PARALLEL_TO_START:-2}"
POLL_INTERVAL="${POLL_INTERVAL:-90}"        # seconds between GPU re-checks
UTIL_THRESH="${UTIL_THRESH:-5}"             # GPU util% considered idle
MIN_FREE_MB="${MIN_FREE_MB:-17000}"         # free-mem requirement
JOB_GRACE_SEC="${JOB_GRACE_SEC:-30}"        # wait for newly-launched job
                                            # to grab the GPU before
                                            # polling sees it as "free"

# -------- Job queue (full 9-model set; comment out to skip) --------
JOBS_DEFAULT=(
    Qwen2.5-3B
    Qwen2.5-3B-Instruct
    Qwen2.5-Coder-3B
    AZR-Coder-3B
    Qwen2.5-7B
    Qwen2.5-7B-Instruct
    Qwen2.5-Coder-7B
    AZR-Base-7B
    AZR-Coder-7B
)

# Optional override: JOBS env-var, space-separated list of short labels
if [[ -n "${JOBS:-}" ]]; then
    read -ra JOBS_LIST <<< "$JOBS"
else
    JOBS_LIST=("${JOBS_DEFAULT[@]}")
fi

# -------- Output dir --------
TS="${TS:-$(date +%Y%m%d-%H%M)}"
ROOT="results/harmbench_repr_$TS"
mkdir -p "$ROOT/_logs" "$ROOT/_done" "$ROOT/_running"

QUEUE_FILE="$ROOT/_queue.txt"
DISPATCHER_LOG="$ROOT/_dispatcher.log"

# Initialise queue file (one short label per line). Skip models that
# already have a .done marker from a previous run.
> "$QUEUE_FILE"
for m in "${JOBS_LIST[@]}"; do
    if [[ -f "$ROOT/_done/$m.done" ]]; then
        echo "[$(date)] $m already done, skipping" | tee -a "$DISPATCHER_LOG"
    else
        echo "$m" >> "$QUEUE_FILE"
    fi
done

remaining_count() { grep -cve '^$' "$QUEUE_FILE" 2>/dev/null || echo 0; }
running_count()   { ls "$ROOT/_running"/*.gpu* 2>/dev/null | wc -l; }
done_count()      { ls "$ROOT/_done"/*.done    2>/dev/null | wc -l; }

# Pop the next model from queue (atomic via temp file). Echoes the label
# on stdout, empty string if queue empty.
pop_job() {
    local job=""
    if [[ -s "$QUEUE_FILE" ]]; then
        job=$(head -n 1 "$QUEUE_FILE")
        tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    fi
    echo "$job"
}

# Find free GPUs that we don't already have a worker on. Echoes
# space-separated GPU IDs ordered by free-memory descending.
free_gpus() {
    local taken=""
    for rt in "$ROOT/_running"/*.gpu*; do
        [[ -e "$rt" ]] || continue
        taken="$taken $(basename "$rt" | awk -F.gpu '{print $2}')"
    done
    nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.free \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk -v thr="$UTIL_THRESH" -v need="$MIN_FREE_MB" -v taken=" $taken " '
            {
                gsub(/ /,"")
                split($0,a,",")
                gpu=a[1]; util=a[2]; freemb=a[3]
                if (index(taken, " "gpu" ") > 0) next
                if (util+0 <= thr && freemb+0 >= need) {
                    print freemb"\t"gpu
                }
            }' \
        | sort -k1,1nr \
        | awk '{print $2}'
}

launch_worker() {
    local gpu="$1"
    local model="$2"
    echo "[$(date)] LAUNCH gpu=$gpu  model=$model" | tee -a "$DISPATCHER_LOG"
    nohup bash run_repr_worker.sh "$gpu" "$model" "$TS" "$ROOT" \
        > /dev/null 2>&1 &
    sleep "$JOB_GRACE_SEC"   # let CUDA init so subsequent polls see usage
}

echo "[$(date)] === dispatcher start ===" | tee -a "$DISPATCHER_LOG"
echo "  TS=$TS  ROOT=$ROOT" | tee -a "$DISPATCHER_LOG"
echo "  MAX_PARALLEL=$MAX_PARALLEL  MIN_PARALLEL_TO_START=$MIN_PARALLEL_TO_START" \
    | tee -a "$DISPATCHER_LOG"
echo "  POLL=$POLL_INTERVAL  util_thresh=$UTIL_THRESH%  min_free_MB=$MIN_FREE_MB" \
    | tee -a "$DISPATCHER_LOG"
echo "  jobs: $(remaining_count) queued" | tee -a "$DISPATCHER_LOG"

# Boot phase: try to start MAX_PARALLEL right now, but accept >=MIN_PARALLEL_TO_START.
boot_free=( $(free_gpus) )
if [[ ${#boot_free[@]} -lt $MIN_PARALLEL_TO_START ]]; then
    echo "[$(date)] only ${#boot_free[@]} GPU(s) free now — need >= $MIN_PARALLEL_TO_START. " \
         "Waiting for cluster to clear." | tee -a "$DISPATCHER_LOG"
fi

# -------- Main loop --------
while true; do
    rem=$(remaining_count)
    run=$(running_count)
    don=$(done_count)
    total=${#JOBS_LIST[@]}

    if (( rem == 0 && run == 0 )); then
        echo "[$(date)] all $total jobs finished ($don done). Exiting." \
            | tee -a "$DISPATCHER_LOG"
        break
    fi

    free=( $(free_gpus) )
    capacity=$(( MAX_PARALLEL - run ))
    if (( capacity > ${#free[@]} )); then
        capacity=${#free[@]}
    fi
    if (( capacity > rem )); then
        capacity=$rem
    fi

    if (( capacity > 0 )); then
        echo "[$(date)] STATUS run=$run done=$don queued=$rem free_gpus=${free[*]} " \
             "capacity=$capacity" | tee -a "$DISPATCHER_LOG"
        for ((i = 0; i < capacity; i++)); do
            job=$(pop_job)
            [[ -z "$job" ]] && break
            launch_worker "${free[$i]}" "$job"
        done
    else
        echo "[$(date)] STATUS run=$run done=$don queued=$rem free_gpus=${free[*]:-none} " \
             "(no capacity)" | tee -a "$DISPATCHER_LOG"
    fi

    sleep "$POLL_INTERVAL"
done
