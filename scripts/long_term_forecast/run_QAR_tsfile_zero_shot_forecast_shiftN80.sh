#!/usr/bin/env bash
set -euo pipefail

# Zero-shot forecasting on QAR tsfile compact caches.
# Uses the same QAR_forecast loader as the full-shot task, but calls
# run.py --task_name zero_shot_forecast --is_training 0.

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
COMPACT_ROOT="${COMPACT_ROOT:-${DATA_ROOT}/compact/context60/hist5/predict_2_3}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
MODELS="${MODELS:-Chronos2 TiRex2 TimesFM Toto Moirai}"
RUN_TAG="${RUN_TAG:-qar_context60_zero_shot_$(date +%Y%m%d_%H%M%S)}"

SEQ_LEN="${SEQ_LEN:-460}"
LABEL_LEN="${LABEL_LEN:-20}"
PRED_LEN="${PRED_LEN:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-2021}"
SPLIT_BOUNDARY="${SPLIT_BOUNDARY:-legacy_float}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}"

D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
N_HEADS="${N_HEADS:-8}"
E_LAYERS="${E_LAYERS:-2}"
D_LAYERS="${D_LAYERS:-1}"
DROPOUT="${DROPOUT:-0.1}"

LOG_DIR="${LOG_DIR:-./logs/zero_shot_forecast/${RUN_TAG}}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/summary.tsv}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
if ! command -v -- "${PYTHON}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi
EXTRA_ARGS=()
if [[ "${NORMAL_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--qar_forecast_normal_only)
fi

mkdir -p "${LOG_DIR}"
if [[ ! -f "${SUMMARY_FILE}" ]]; then
  printf 'dataset\tmodel\tstatus\tlog\tresult_dir\n' > "${SUMMARY_FILE}"
fi

echo "Run tag: ${RUN_TAG}"
echo "Task: zero_shot_forecast"
echo "Compact root: ${COMPACT_ROOT}"
echo "Datasets: ${DATASETS}"
echo "Models: ${MODELS}"
echo "Window: seq_len=${SEQ_LEN}, label_len=${LABEL_LEN}, pred_len=${PRED_LEN}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Logs: ${LOG_DIR}"
echo "QAR_SPLIT_STRATEGY: ${QAR_SPLIT_STRATEGY}"
if [[ -n "${CHRONOS2_MODEL_PATH:-}" ]]; then
  echo "CHRONOS2_MODEL_PATH: ${CHRONOS2_MODEL_PATH}"
fi

failures=0
for dataset in ${DATASETS}; do
  root_path="${COMPACT_ROOT}/${dataset}/"
  cache_path="${root_path}/qar_compact_shiftN80.npz"
  if [[ ! -f "${cache_path}" ]]; then
    echo "Missing compact cache: ${cache_path}" >&2
    exit 1
  fi

  feature_count="$("${PYTHON}" - "${cache_path}" <<'PY'
import sys
import numpy as np
cache = np.load(sys.argv[1], allow_pickle=False)
print(int(cache["x"].shape[2]))
PY
)"

  for model in ${MODELS}; do
    des="${RUN_TAG}_${dataset}_${model}"
    model_id="${dataset}_QAR_zero_shot_shiftN80"
    log_file="${LOG_DIR}/${dataset}_${model}.log"
    echo "[$(date '+%F %T')] START dataset=${dataset} model=${model} features=${feature_count}"

    set +e
    "${PYTHON}" -u run.py \
      --task_name zero_shot_forecast \
      --is_training 0 \
      --root_path "${root_path}" \
      --qar_split_strategy "${QAR_SPLIT_STRATEGY}" \
      --qar_split_boundary "${SPLIT_BOUNDARY}" \
      --data_path qar_compact_shiftN80.npz \
      --model_id "${model_id}" \
      --model "${model}" \
      --data QAR_forecast \
      --features M \
      --target var_0 \
      --freq h \
      --seq_len "${SEQ_LEN}" \
      --label_len "${LABEL_LEN}" \
      --pred_len "${PRED_LEN}" \
      --enc_in "${feature_count}" \
      --dec_in "${feature_count}" \
      --c_out "${feature_count}" \
      --e_layers "${E_LAYERS}" \
      --d_layers "${D_LAYERS}" \
      --batch_size "${BATCH_SIZE}" \
      --d_model "${D_MODEL}" \
      --n_heads "${N_HEADS}" \
      --d_ff "${D_FF}" \
      --dropout "${DROPOUT}" \
      --des "${des}" \
      --itr 1 \
      --seed "${SEED}" \
      "${EXTRA_ARGS[@]}" \
      --num_workers "${NUM_WORKERS}" \
      > "${log_file}" 2>&1
    status=$?
    set -e

    # results may be a symlink to the large shared filesystem on the server.
    result_dir="$(find -L ./results -maxdepth 1 -type d -name "*${des}_0" -print -quit 2>/dev/null || true)"
    printf '%s\t%s\t%s\t%s\t%s\n' "${dataset}" "${model}" "${status}" "${log_file}" "${result_dir}" >> "${SUMMARY_FILE}"

    if [[ "${status}" -ne 0 ]]; then
      failures=$((failures + 1))
      echo "[$(date '+%F %T')] FAIL dataset=${dataset} model=${model}; see ${log_file}" >&2
    else
      echo "[$(date '+%F %T')] DONE dataset=${dataset} model=${model}"
    fi
  done
done

echo "Summary: ${SUMMARY_FILE}"
if (( failures > 0 )); then exit 1; fi
