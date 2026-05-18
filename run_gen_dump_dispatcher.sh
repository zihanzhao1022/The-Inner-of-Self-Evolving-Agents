#!/usr/bin/env bash
# ============================================================================
# Dispatcher for full generation+residual dump.
#
# Runs dump_generation_residuals.py for each of the 9 models in the project
# matrix, on up to MAX_PARALLEL GPUs concurrently. Each per-model job takes
# ~15-20h (1024 generated tokens × 400 prompts × output_hidden_states).
#
# Strategy mirrors run_repr_dispatcher.sh but each worker invokes the
# generation-dump script (no separate stage A/B; just one long job per model).
#
# Usage:
#   bash run_gen_dump_dispatcher.sh
#   MAX_PARALLEL=3 bash run_gen_dump_dispatcher.sh
#
# Output layout:
#   results/full_gen_residuals_<TS>/
#     _queue.txt
#     _running/<m>.gpu<n>
#     _done/<m>.done
#     _logs/<m>.log
#     _dispatcher.log
#     <SHORT>/ ... (per-model subdirs created by dump script)
# ============================================================================

set -u
set -o pipefail

MAX_PARALLEL="${MAX_PARALLEL:-4}"
MIN_PARALLEL_TO_START="${MIN_PARALLEL_TO_START:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-120}"
UTIL_THRESH="${UTIL_THRESH:-5}"
MIN_FREE_MB="${MIN_FREE_MB:-20000}"   # generation needs more headroom than dump
JOB_GRACE_SEC="${JOB_GRACE_SEC:-60}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"

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

if [[ -n "${JOBS:-}" ]]; then
    read -ra JOBS_LIST <<< "$JOBS"
else
    JOBS_LIST=("${JOBS_DEFAULT[@]}")
fi

TS="${TS:-$(date +%Y%m%d-%H%M)}"
ROOT="results/full_gen_residuals_$TS"
mkdir -p "$ROOT/_logs" "$ROOT/_done" "$ROOT/_running"

QUEUE_FILE="$ROOT/_queue.txt"
DISPATCHER_LOG="$ROOT/_dispatcher.log"

> "$QUEUE_FILE"
for m in "${JOBS_LIST[@]}"; do
    if [[ -f "$ROOT/_done/$m.done" ]]; then
        echo "[$(date)] $m already done, skipping" | tee -a "$DISPATCHER_LOG"
    else
        echo "$m" >> "$QUEUE_FILE"
    fi
done

remaining_count() { local n; n=$(grep -cve '^$' "$QUEUE_FILE" 2>/dev/null); echo "${n:-0}"; }
running_count()   { ls "$ROOT/_running"/*.gpu* 2>/dev/null | wc -l; }
done_count()      { ls "$ROOT/_done"/*.done    2>/dev/null | wc -l; }

pop_job() {
    local job=""
    if [[ -s "$QUEUE_FILE" ]]; then
        job=$(head -n 1 "$QUEUE_FILE")
        tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp" && mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
    fi
    echo "$job"
}

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
                if (util+0 <= thr && freemb+0 >= need) print freemb"\t"gpu
            }' \
        | sort -k1,1nr \
        | awk '{print $2}'
}

# Per-model HF + dtype lookup
declare -A HF_PATH=(
    [Qwen2.5-3B]="Qwen/Qwen2.5-3B"
    [Qwen2.5-3B-Instruct]="Qwen/Qwen2.5-3B-Instruct"
    [Qwen2.5-Coder-3B]="Qwen/Qwen2.5-Coder-3B"
    [AZR-Coder-3B]="andrewzh/Absolute_Zero_Reasoner-Coder-3b"
    [Qwen2.5-7B]="Qwen/Qwen2.5-7B"
    [Qwen2.5-7B-Instruct]="Qwen/Qwen2.5-7B-Instruct"
    [Qwen2.5-Coder-7B]="Qwen/Qwen2.5-Coder-7B"
    [AZR-Base-7B]="andrewzh2/Absolute_Zero_Reasoner-Base-7b"
    [AZR-Coder-7B]="andrewzh/Absolute_Zero_Reasoner-Coder-7b"
)
declare -A DTYPE=(
    [Qwen2.5-3B]="float32"
    [Qwen2.5-3B-Instruct]="float32"
    [Qwen2.5-Coder-3B]="float32"
    [AZR-Coder-3B]="float32"
    [Qwen2.5-7B]="bfloat16"
    [Qwen2.5-7B-Instruct]="bfloat16"
    [Qwen2.5-Coder-7B]="bfloat16"
    [AZR-Base-7B]="bfloat16"
    [AZR-Coder-7B]="bfloat16"
)

launch_worker() {
    local gpu="$1"
    local model="$2"
    local hf="${HF_PATH[$model]}"
    local dtype="${DTYPE[$model]}"
    echo "[$(date)] LAUNCH gpu=$gpu  model=$model  hf=$hf  dtype=$dtype" | tee -a "$DISPATCHER_LOG"
    touch "$ROOT/_running/$model.gpu$gpu"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export PATH="$HOME/miniconda3/envs/py310/bin:$PATH"
        cd "$(dirname "$(readlink -f "$0")")"
        python -m llm_lens.examples.dump_generation_residuals \
            --model "$hf" --short "$model" --template-mode auto \
            --dataset harmbench200_alpaca200 --n-per-class 200 \
            --max-new-tokens "$MAX_NEW_TOKENS" \
            --dtype "$dtype" --storage-dtype match \
            --output-root "$ROOT" \
            >> "$ROOT/_logs/$model.log" 2>&1
        local rc=$?
        if [[ $rc -eq 0 ]]; then
            touch "$ROOT/_done/$model.done"
            echo "[$(date)] DONE   $model on gpu=$gpu" | tee -a "$DISPATCHER_LOG"
        else
            echo "[$(date)] FAILED $model on gpu=$gpu (rc=$rc)" | tee -a "$DISPATCHER_LOG"
        fi
        rm -f "$ROOT/_running/$model.gpu$gpu"
    ) &
    sleep "$JOB_GRACE_SEC"
}

echo "[$(date)] === gen-dump dispatcher start ===" | tee -a "$DISPATCHER_LOG"
echo "  TS=$TS  ROOT=$ROOT  MAX_NEW=$MAX_NEW_TOKENS" | tee -a "$DISPATCHER_LOG"
echo "  MAX_PARALLEL=$MAX_PARALLEL  MIN_FREE_MB=$MIN_FREE_MB" | tee -a "$DISPATCHER_LOG"
echo "  jobs queued: $(remaining_count)" | tee -a "$DISPATCHER_LOG"

while true; do
    rem=$(remaining_count); run=$(running_count); don=$(done_count)
    total=${#JOBS_LIST[@]}
    if (( rem == 0 && run == 0 )); then
        echo "[$(date)] all $total jobs finished ($don done). Exiting." \
            | tee -a "$DISPATCHER_LOG"
        break
    fi
    free=( $(free_gpus) )
    capacity=$(( MAX_PARALLEL - run ))
    if (( capacity > ${#free[@]} )); then capacity=${#free[@]}; fi
    if (( capacity > rem )); then capacity=$rem; fi
    if (( capacity > 0 )); then
        echo "[$(date)] STATUS run=$run done=$don queued=$rem free=${free[*]} cap=$capacity" \
            | tee -a "$DISPATCHER_LOG"
        for ((i = 0; i < capacity; i++)); do
            job=$(pop_job); [[ -z "$job" ]] && break
            launch_worker "${free[$i]}" "$job"
        done
    else
        echo "[$(date)] STATUS run=$run done=$don queued=$rem free=${free[*]:-none} (no capacity)" \
            | tee -a "$DISPATCHER_LOG"
    fi
    sleep "$POLL_INTERVAL"
done
