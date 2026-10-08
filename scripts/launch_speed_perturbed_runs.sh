#!/usr/bin/env bash
# Sequential finetunes on speed-perturbed corpora (original + 0.9× + 1.1×).
# Same protocol as launch_experiment_runs.sh; separate output dirs (*-sp-runN).
#
# Usage (MoDyCo, repo root):
#   bash scripts/launch_speed_perturbed_runs.sh              # xlsr then whisper, 1–4
#   bash scripts/launch_speed_perturbed_runs.sh xlsr         # XLSR priority
#   bash scripts/launch_speed_perturbed_runs.sh whisper 1 2
#
# Prerequisites:
#   python scripts/13_build_speed_perturbed_corpus.py
#
# Logs: logs/train_${MODEL}_sp_run${N}.log

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
  local script out log dataset
  dataset="1_data_prepared/experiment_runs_sp/run${run}"
  case "$model" in
    whisper)
      script=scripts/3_train_whisper_ru.py
      out="2_models/whisper-small-nenets-ru-sp-run${run}"
      ;;
    xlsr)
      script=scripts/3_train_wav2vec_long.py
      out="2_models/wav2vec2-large-xlsr-nenets-sp-run${run}"
      ;;
    *)
      echo "Unknown model: $model" >&2
      return 1
      ;;
  esac
  log="logs/train_${model}_sp_run${run}.log"

  if [[ ! -f "${dataset}/train/metadata.csv" ]]; then
    echo "[error] missing ${dataset} — run scripts/13_build_speed_perturbed_corpus.py" >&2
    return 1
  fi

  if [[ -f "${out}/eval_predictions.txt" ]]; then
    echo "[skip] ${model} sp-run${run}: ${out}/eval_predictions.txt exists"
    return 0
  fi

  echo "============================================================"
  echo "  START ${model} SP-RUN=${run}  $(date -Is)"
  echo "  dataset=${dataset}"
  echo "  output=${out}"
  echo "  log=${log}"
  echo "============================================================"

  RUN="${run}" DATASET_PATH="${dataset}" OUTPUT_DIR="${out}" \
    "${PY}" "${script}" 2>&1 | tee "${log}"

  echo "  DONE ${model} SP-RUN=${run}  $(date -Is)"
}

models=()
case "$MODE" in
  all) models=(xlsr whisper) ;;  # XLSR first (priority)
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

echo "All requested SP jobs finished at $(date -Is)"
