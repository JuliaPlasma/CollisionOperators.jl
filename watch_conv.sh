#!/usr/bin/env bash
# Convergence watcher for the aniso_logsq 1000->4000 run.
# Exits (notifying the agent) on: DONE (step>=4000), TROUBLE (nan / iter near max /
# stall / unit died), or hourly HEARTBEAT. Agent re-inspects on each exit.
CSV=conservation_history_aniso100_logsq.csv
UNIT=junyi-task-julia-Gonzalez-diagnostics-aniso_logsq_4000.service
TARGET=4000
ITER_MAX=1500          # 75% of preset max_iter=2000 -> Gonzalez not converging
STALL_LIMIT=8          # 8 * 120s = 16 min no step progress while active
start=$(date +%s)
prev_step=-1
stall=0
while true; do
    last=$(tail -1 "$CSV")
    step=$(printf '%s' "$last" | cut -d, -f1)
    maxiter=$(tail -15 "$CSV" | awk -F, '{if($7+0>m)m=$7+0}END{print m+0}')
    nan=$(tail -15 "$CSV" | grep -ci 'nan\|inf')
    active=$(systemctl --user show "$UNIT" -p ActiveState --value)

    if [ "${step:-0}" -ge "$TARGET" ] 2>/dev/null; then
        echo "DONE step=$step maxiter=$maxiter"; exit 0
    fi
    if [ "$active" != "active" ]; then
        echo "UNIT_DIED active=$active step=$step maxiter=$maxiter"; exit 2
    fi
    if [ "${nan:-0}" -gt 0 ]; then
        echo "TROUBLE_NAN step=$step (nan/inf in CSV)"; exit 3
    fi
    if [ "${maxiter:-0}" -ge "$ITER_MAX" ] 2>/dev/null; then
        echo "TROUBLE_ITER maxiter=$maxiter step=$step (Gonzalez stalling)"; exit 4
    fi
    if [ "$step" = "$prev_step" ]; then
        stall=$((stall+1))
    else
        stall=0; prev_step=$step
    fi
    if [ "$stall" -ge "$STALL_LIMIT" ]; then
        echo "TROUBLE_STALL step=$step (no progress ~16min, inner solve stuck)"; exit 5
    fi
    now=$(date +%s)
    if [ $((now-start)) -ge 3600 ]; then
        echo "HEARTBEAT step=$step maxiter=$maxiter residual_ok"; exit 0
    fi
    sleep 120
done
