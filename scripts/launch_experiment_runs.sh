#!/usr/bin/env bash
# Sequential finetunes for Aleksandra's Run1–4 inventory (XLSR + Whisper-small RU).
# Run from repo root on MoDyCo:  ~/morgane/nenets
#
# Usage:
#   bash scripts/launch_experiment_runs.sh              # all runs, whisper then xlsr
#   bash scripts/launch_experiment_runs.sh whisper 1 2  # whisper only, runs 1–2
#   bash scripts/launch_experiment_runs.sh xlsr 1       # xlsr run 1 only
#   MODEL=whisper RUNS="1 2 3 4" bash scripts/launch_experiment_runs.sh
#
# Logs: logs/train_${MODEL}_run${N}.log
# Skip a job if its output dir already has eval_predictions.txt

set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${ROOT}/venv/bin/python"
mkdir -p logs

MODE="${1:-all}"
shift || true
if [[ $# -gt 0 ]]; then
  RUNS=("$@")
else
  # shellcheck disable=SC2206
  RUNS=(${RUNS:-1 2 3 4})
fi

run_one() {
  local model="$1" run="$2"
  local script out log
  case "$model" in
    whisper)
      script=scripts/3_train_whisper_ru.py
      out="2_models/whisper-small-nenets-ru-run${run}"
      ;;
    xlsr)
      script=scripts/3_train_wav2vec_long.py
      out="2_models/wav2vec2-large-xlsr-nenets-run${run}"
      ;;
    *)
      echo "Unknown model: $model" >&2
      return 1
      ;;
  esac
  log="logs/train_${model}_run${run}.log"

  if [[ -f "${out}/eval_predictions.txt" ]]; then
    echo "[skip] ${model} run${run}: ${out}/eval_predictions.txt exists"
    return 0
  fi

  echo "============================================================"
  echo "  START ${model} RUN=${run}  $(date -Is)"
  echo "  dataset=1_data_prepared/experiment_runs/run${run}"
  echo "  output=${out}"
  echo "  log=${log}"
  echo "============================================================"

  RUN="${run}" OUTPUT_DIR="${out}" \
    "${PY}" "${script}" 2>&1 | tee "${log}"

  echo "  DONE ${model} RUN=${run}  $(date -Is)"
}

models=()
case "$MODE" in
  all) models=(whisper xlsr) ;;
  whisper|xlsr) models=("$MODE") ;;
  *)
    echo "Usage: $0 [all|whisper|xlsr] [run ...]" >&2
    exit 1
    ;;
esac

echo "Repo: ${ROOT}"
echo "Python: ${PY}"
"${PY}" -c 'import torch; print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")'
nvidia-smi --query-gpu=memory.free,memory.used,memory.total --format=csv || true

for model in "${models[@]}"; do
  for run in "${RUNS[@]}"; do
    run_one "$model" "$run"
  done
done

echo "All requested jobs finished at $(date -Is)"
