#!/usr/bin/env bash
# Run run_perturbations.py stages one after the other, in the background, with the Mac kept awake.
#
#   ./run_batch.sh start     launch all stages in a new batch folder (returns immediately; safe to close the terminal)
#   ./run_batch.sh status    which stage is running + its latest progress / ETA line
#   ./run_batch.sh watch     follow progress live (Ctrl+C stops watching, not the run)
#   ./run_batch.sh stop      stop the batch (cached predictions are kept; a new start reuses them)
#   ./run_batch.sh estimate  expected duration of every stage, runs nothing
#
# Each start gets its own folder, results/perturbations/batches/<YYYY-MM-DD_HHMM>_<NAME>/, holding everything of that
# batch and nothing else: summary_by_variation.csv and metrics_by_recording.csv (rows of all stages, `run` column =
# stage), predictions/<detector>/<condition>/<recording>.tsv (copy of every prediction used), runs/<stage>.json
# (command, git commit, computed vs reused, computed_files = predictions new in the cache, failures, start/end),
# logs/<stage>.log and logs/batch.log. Copy the folder and you have the whole batch. Older results
# (included_patients/, earlier batches) are never touched.
# Stages run in order and stop at the first failure. The prediction cache is shared, so a later stage, or a new
# batch, skips the detector runs already done. A macOS notification is shown when the batch ends.
set -euo pipefail
cd "$(dirname "$0")"

NAME="eventnet_dinoeeg-norej_floating"   # describes this batch; part of its folder name
DETECTORS="eventnet,dinoeeg_norej"
# stage name|extra arguments for run_perturbations.py (no --variations = the core conditions, CORE_VARIATIONS)
STAGES=(
  "1_disconnection|--variations disconnected_mild,disconnected_severe,floating_mild,floating_severe"
  "2_core|"
)

BATCHES=results/perturbations/batches
STATE=$BATCHES/.current   # line 1: pid, line 2: batch folder

running() { [[ -f $STATE ]] && kill -0 "$(sed -n 1p "$STATE")" 2>/dev/null; }
batch_dir() { [[ -f $STATE ]] && sed -n 2p "$STATE"; }
notify() { osascript -e "display notification \"$1\" with title \"002 perturbations\"" 2>/dev/null || true; }

run_stages() {
  local dir=$1 out=${1#results/perturbations/}
  mkdir -p "$dir/logs"
  echo "$(date '+%F %T') batch started: $NAME (detectors: $DETECTORS)" >> "$dir/logs/batch.log"
  for stage in "${STAGES[@]}"; do
    name=${stage%%|*}; extra=${stage#*|}
    echo "$(date '+%F %T') $name started" >> "$dir/logs/batch.log"
    # shellcheck disable=SC2086
    if poetry run python run_perturbations.py --detectors "$DETECTORS" --out "$out" --run-label "$name" $extra \
        > "$dir/logs/$name.log" 2>&1; then
      echo "$(date '+%F %T') $name done" >> "$dir/logs/batch.log"
    else
      echo "$(date '+%F %T') $name FAILED, batch stopped (see logs/$name.log)" >> "$dir/logs/batch.log"
      notify "$name failed, batch stopped"
      exit 1
    fi
  done
  echo "$(date '+%F %T') batch finished" >> "$dir/logs/batch.log"
  notify "All stages finished: $(basename "$dir")"
}

current_log() { ls -t "$(batch_dir)"/logs/[0-9]*.log 2>/dev/null | head -1; }

case "${1:-}" in
  start)
    if running; then echo "Already running: $(batch_dir). Use: $0 status"; exit 1; fi
    dir=$BATCHES/$(date +%Y-%m-%d_%H%M)_$NAME
    mkdir -p "$dir/logs"
    # caffeinate -i keeps the Mac awake while the batch runs; nohup survives closing the terminal
    nohup caffeinate -i "$0" _run "$dir" > /dev/null 2>&1 &
    printf '%s\n%s\n' "$!" "$dir" > "$STATE"
    echo "Started: $dir"
    echo "Follow with: $0 watch    Check with: $0 status"
    ;;
  _run)
    run_stages "$2"
    ;;
  status)
    [[ -f $STATE ]] || { echo "No batch started yet"; exit 0; }
    if running; then echo "Running: $(batch_dir)"; else echo "Not running. Last batch: $(batch_dir)"; fi
    tail -n 4 "$(batch_dir)/logs/batch.log" 2>/dev/null || true
    log=$(current_log)
    if [[ -n $log ]]; then
      echo "--- $(basename "$log")"
      grep -E "^\[[0-9]+/[0-9]+\] Running" "$log" | tail -1 || true
      grep "\[progress\]" "$log" | tail -1 || true
      echo "     failed detector runs: $(grep -c FAILED "$log" || true)"
    fi
    ;;
  watch)
    log=$(current_log)
    [[ -n $log ]] || { echo "No stage log yet"; exit 1; }
    echo "Following $log (the next stage writes a new log: run watch again then)"
    tail -f "$log" | grep --line-buffered -E "progress|Running|FAILED|Recall"
    ;;
  stop)
    if running; then
      pid=$(sed -n 1p "$STATE")
      pkill -f "run_batch.sh _run" 2>/dev/null || true
      pkill -f "run_perturbations.py --detectors $DETECTORS" 2>/dev/null || true
      kill "$pid" 2>/dev/null || true
      echo "$(date '+%F %T') batch stopped by user" >> "$(batch_dir)/logs/batch.log"
      echo "Stopped: $(batch_dir). Cached predictions are kept; a new start reuses them."
    else
      echo "Not running"
    fi
    ;;
  estimate)
    echo "Each stage is estimated on today's cache. Stage 2 includes stage 1's conditions, which will be cached"
    echo "by the time it runs, so the batch takes about as long as the last stage's figure, not the sum."
    for stage in "${STAGES[@]}"; do
      name=${stage%%|*}; extra=${stage#*|}
      echo; echo "== $name"
      # shellcheck disable=SC2086
      poetry run python run_perturbations.py --detectors "$DETECTORS" $extra --estimate 2>/dev/null \
        | sed -n '/To run/,$p'
    done
    ;;
  *)
    sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
