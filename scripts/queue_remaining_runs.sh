#!/usr/bin/env bash
# Wait for any in-flight train job, then run the remaining inventory finetunes
# sequentially (Whisper run3–4, then XLSR run1–4). Already-finished jobs are
# skipped by launch_experiment_runs.sh when eval_predictions.txt exists.
#
# Usage (from repo root on MoDyCo):
#   nohup bash scripts/queue_remaining_runs.sh > logs/nohup_queue.log 2>&1 &
#
# Optional env:
#   WAIT_POLL_SEC=60          # how often to check if training is still running
#   ALSO_WHISPER_RUNS="3 4"   # default
#   ALSO_XLSR_RUNS="1 2 3 4"  # default

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
mkdir -p logs

POLL="${WAIT_POLL_SEC:-60}"
WHISPER_RUNS="${ALSO_WHISPER_RUNS:-3 4}"
XLSR_RUNS="${ALSO_XLSR_RUNS:-1 2 3 4}"
QUEUE_LOG="logs/queue_remaining_runs.log"

log() { echo "[$(date -Is)] $*" | tee -a "$QUEUE_LOG"; }

train_running() {
  # Match train scripts / launcher, but not this queue script itself.
  pgrep -f 'scripts/(3_train_whisper_ru|3_train_wav2vec_long)\.py' >/dev/null 2>&1 \
    || pgrep -f 'scripts/launch_experiment_runs\.sh' >/dev/null 2>&1
}

log "Queue started (cwd=${ROOT})"
log "Will wait for current train jobs, then whisper [${WHISPER_RUNS}] then xlsr [${XLSR_RUNS}]"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv || true

while train_running; do
  log "Training still running — sleep ${POLL}s"
  sleep "$POLL"
done

log "No active train job — starting remaining Whisper runs"
# shellcheck disable=SC2086
if ! bash scripts/launch_experiment_runs.sh whisper ${WHISPER_RUNS} 2>&1 | tee -a "$QUEUE_LOG"; then
  log "WARNING: Whisper segment exited non-zero — continuing with XLSR"
fi

log "Whisper queue segment done — starting XLSR runs"
# shellcheck disable=SC2086
if ! bash scripts/launch_experiment_runs.sh xlsr ${XLSR_RUNS} 2>&1 | tee -a "$QUEUE_LOG"; then
  log "WARNING: XLSR segment exited non-zero"
fi

log "Queue finished"
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv || true
