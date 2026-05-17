#!/bin/bash
# eval_watchdog.sh
# Captures eval state every 60s for Claude's overnight monitoring.
# Writes:
#   ~/eval_logs/latest_status.txt        — overwritten each cycle, the "current state"
#   ~/eval_logs/snapshots/status_*.txt   — per-cycle history (kept 24 h then auto-pruned)
#   ~/eval_logs/watchdog.log             — watchdog's own log
#   ~/eval_logs/alerts.log               — anomalies needing attention
#   ~/eval_logs/progress.txt             — human-readable progress summary

set -u

LOG_DIR=$HOME/eval_logs
PROJECT=$HOME/The-Inner-of-Self-Evolving-Agents
RESULTS=$PROJECT/results/harmbench_eval_20260515-1512

mkdir -p "$LOG_DIR/snapshots"

WD_LOG=$LOG_DIR/watchdog.log
ALERTS=$LOG_DIR/alerts.log
STATUS=$LOG_DIR/latest_status.txt
PROGRESS=$LOG_DIR/progress.txt

log_wd() { echo "[$(date '+%F %T')] $*" >> "$WD_LOG"; }
alert()  { echo "[$(date '+%F %T')] $*" | tee -a "$ALERTS" >> "$WD_LOG"; }

log_wd "watchdog started, pid=$$"

# Track which GPUs are "ours" vs others, and when free GPUs (not 3/4/6) free up
declare -A free_seen_at

while true; do
    NOW=$(date '+%F %T')

    # Active processes
    GEN_PIDS=$(pgrep -u "$USER" -f "run_generations" 2>/dev/null | tr '\n' ' ')
    GEN_COUNT=$(echo "$GEN_PIDS" | wc -w)
    DRIVER_BASH_PIDS=$(pgrep -u "$USER" -f "bash.*run_harmbench_batches" 2>/dev/null | tr '\n' ' ')
    HOLD_PIDS=$(pgrep -u "$USER" -f "gpu_hold.py" 2>/dev/null | tr '\n' ' ')

    # GPU state
    GPU_RAW=$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,temperature.gpu --format=csv,noheader 2>&1)
    COMPUTE_APPS=$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>&1)

    # JSONL line counts (real progress indicator)
    JSONL_REPORT=""
    if [ -d "$RESULTS" ]; then
        while IFS= read -r jf; do
            [ -z "$jf" ] && continue
            LINES=$(wc -l < "$jf" 2>/dev/null || echo 0)
            REL=${jf#$RESULTS/}
            JSONL_REPORT+="  $LINES lines  $REL"$'\n'
        done < <(find "$RESULTS" -name 'generations.jsonl' 2>/dev/null | sort)
    fi

    # Log file sizes
    LOG_REPORT=""
    for f in "$LOG_DIR"/gpu*.log; do
        [ -f "$f" ] || continue
        SIZE=$(stat -c %s "$f" 2>/dev/null)
        MTIME=$(stat -c '%y' "$f" 2>/dev/null | cut -d'.' -f1)
        BN=$(basename "$f")
        LOG_REPORT+="  ${SIZE} B  mtime=${MTIME}  ${BN}"$'\n'
    done

    # Write latest_status.txt
    {
        echo "=== STATUS as of $NOW ==="
        echo
        echo "[processes]"
        echo "  gen_count=$GEN_COUNT"
        echo "  gen_pids=$GEN_PIDS"
        echo "  driver_bash_pids=$DRIVER_BASH_PIDS"
        echo "  hold_pids=$HOLD_PIDS"
        echo
        echo "[gpu state — all 8]"
        echo "$GPU_RAW"
        echo
        echo "[my compute-apps allocations]"
        echo "$COMPUTE_APPS"
        echo
        echo "[generation jsonl line counts]"
        echo -n "$JSONL_REPORT"
        echo
        echo "[log file sizes]"
        echo -n "$LOG_REPORT"
        echo
        echo "[driver.log tail 10]"
        tail -10 "$LOG_DIR/driver.log" 2>/dev/null
    } > "$STATUS"

    # History snapshot
    DSTR=$(date '+%F_%H-%M-%S')
    cp "$STATUS" "$LOG_DIR/snapshots/status_${DSTR}.txt"

    # Prune snapshots older than 24h
    find "$LOG_DIR/snapshots" -name 'status_*.txt' -mmin +1440 -delete 2>/dev/null

    # === Anomaly detection ===

    # 1) Driver should be running unless all batches done
    DRIVER_DONE=0
    if grep -q "ALL THREE BATCHES DONE" "$LOG_DIR/driver.log" 2>/dev/null; then
        DRIVER_DONE=1
    fi
    if [ -z "$DRIVER_BASH_PIDS" ] && [ "$DRIVER_DONE" -eq 0 ]; then
        # Don't double-alert
        if ! tail -50 "$ALERTS" 2>/dev/null | grep -q "driver_dead"; then
            alert "driver_dead — no bash matching run_harmbench_batches but batches not complete"
        fi
    fi

    # 2) GEN process count anomalies
    # Expected: between 1 and 3 alive (some GPU chains may finish before others)
    if [ "$GEN_COUNT" -eq 0 ] && [ "$DRIVER_DONE" -eq 0 ] && [ -n "$DRIVER_BASH_PIDS" ]; then
        # Driver alive but no gen process — likely transition between models, ok briefly
        :  # not an alert
    fi

    # 3) Log file stale (no growth in 15 min while gen process active)
    for f in "$LOG_DIR"/gpu*.log; do
        [ -f "$f" ] || continue
        # Get age of last modification
        AGE_SEC=$(( $(date +%s) - $(stat -c %Y "$f") ))
        BN=$(basename "$f")
        # only check active logs (look for matching active chain by GPU number in name)
        GPU_ID=$(echo "$BN" | sed -n 's/gpu\([0-9]*\)_.*/\1/p')
        if [ -n "$GPU_ID" ] && [ "$AGE_SEC" -gt 900 ]; then
            # Check if there's any process using this GPU
            USING=$(echo "$COMPUTE_APPS" | grep -c "$(echo "$COMPUTE_APPS" | head -1)")  # placeholder
            # Only alert once per stale log per hour
            STALE_KEY="stale_${BN}_$(date +%H)"
            if [ ! -f "$LOG_DIR/.alert_${STALE_KEY}" ]; then
                touch "$LOG_DIR/.alert_${STALE_KEY}"
                # Don't alert if this log corresponds to a completed chain
                if grep -q "done — generated" "$f" 2>/dev/null; then
                    : # finished naturally
                else
                    alert "stale_log ${BN}: no growth for ${AGE_SEC}s but no completion marker"
                fi
            fi
        fi
    done

    # 4) Detect newly-free GPUs (others' processes released)
    for IDX in 0 1 2 5 7; do
        MEM_USED=$(echo "$GPU_RAW" | awk -F', *' -v i=$IDX '$1==i {gsub(" MiB","",$2); print $2}')
        if [ -n "$MEM_USED" ] && [ "$MEM_USED" -lt 200 ]; then
            # GPU has < 200 MB used by anyone — newly free
            KEY="gpu_${IDX}_free"
            if [ -z "${free_seen_at[$KEY]:-}" ]; then
                free_seen_at[$KEY]=$NOW
                log_wd "GPU $IDX appears free (mem_used=${MEM_USED} MB) — could expand if needed"
            fi
        else
            unset 'free_seen_at[gpu_'$IDX'_free]' 2>/dev/null
        fi
    done

    # === Progress summary (human-readable) ===
    {
        echo "Last update: $NOW"
        echo
        echo "Driver: $([ -n "$DRIVER_BASH_PIDS" ] && echo "RUNNING" || ([ "$DRIVER_DONE" -eq 1 ] && echo "ALL DONE" || echo "STOPPED"))"
        echo "Gen procs alive: $GEN_COUNT"
        echo
        echo "Per-batch progress (jsonl line counts, target 50 each):"
        for IDX in 50 100 150; do
            END=$((IDX + 49))
            echo "  idx ${IDX}-${END}:"
            for mset in 3B 7B; do
                DIR="$RESULTS/generations_${mset}_idx${IDX}_${END}"
                if [ -d "$DIR" ]; then
                    for model_dir in "$DIR"/*/; do
                        [ -d "$model_dir" ] || continue
                        MODEL=$(basename "$model_dir")
                        JF="$model_dir/generations.jsonl"
                        if [ -f "$JF" ]; then
                            LINES=$(wc -l < "$JF" 2>/dev/null)
                            BAR=""
                            PCT=$((LINES * 100 / 50))
                            FILL=$((LINES * 20 / 50))
                            for ((i=0; i<20; i++)); do
                                if [ $i -lt $FILL ]; then BAR+="#"; else BAR+="-"; fi
                            done
                            printf "    %-3s %-22s [%s] %3d%%  (%d/50)\n" "$mset" "$MODEL" "$BAR" "$PCT" "$LINES"
                        fi
                    done
                fi
            done
        done
    } > "$PROGRESS"

    sleep 60
done
