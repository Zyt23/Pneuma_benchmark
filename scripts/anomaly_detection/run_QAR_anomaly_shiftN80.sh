#!/usr/bin/env bash
set -euo pipefail

# QAR one-class anomaly detection.
#
# Train/val use only normal samples (label 0). Test uses the chronological last
# 20% of normal samples plus the chronological last 20% of fault samples. The
# threshold is selected from normal validation reconstruction errors, so no
# fault/test labels are used for training or thresholding.

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-data}"
COMPACT_ROOT="${COMPACT_ROOT:-${DATA_ROOT}/compact/classification}"
RUN_TAG="${RUN_TAG:-qar_anomaly_oneclass_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
IS_TRAINING="${IS_TRAINING:-1}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
MODELS="${MODELS:-KANAD AnomalyTransformer TranAD USAD OmniAnomaly}"

SEQ_LEN="${SEQ_LEN:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-5}"
SEED="${SEED:-2021}"
ITR="${ITR:-1}"
SPLIT_BOUNDARY="${SPLIT_BOUNDARY:-legacy_float}"
PATIENCE="${PATIENCE:-2}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-2}"
N_HEADS="${N_HEADS:-4}"
# Pure one-class defaults: the threshold is computed automatically from the
# normal validation scores only. No fault label is used for fitting, direction
# selection, threshold selection, or checkpoint selection.
THRESHOLD_PERCENTILE="${THRESHOLD_PERCENTILE:-95.0}"
THRESHOLD_SOURCE="${THRESHOLD_SOURCE:-val}"
SCORE_DIRECTION="${SCORE_DIRECTION:-high}"
NUM_WORKERS="${NUM_WORKERS:-0}"
CUDA_DEVICE="${CUDA_DEVICE:-${GPU:-0}}"
LOCAL_GPU="${LOCAL_GPU:-0}"
CHECKPOINTS="${CHECKPOINTS:-./checkpoints/anomaly/${RUN_TAG}}"

SUMMARY_DIR="${SUMMARY_DIR:-logs/anomaly_detection/${RUN_TAG}}"
mkdir -p "${SUMMARY_DIR}" "${CHECKPOINTS}"
SUMMARY_FILE="${SUMMARY_DIR}/summary.tsv"
if [ ! -f "${SUMMARY_FILE}" ]; then
  printf "dataset\tmodel\tstatus\tenc_in\troot_path\tlog_file\n" > "${SUMMARY_FILE}"
fi

cd "${PROJECT_ROOT}" || exit 1
if ! command -v -- "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

failures=0
for dataset in ${DATASETS}; do
  root_path="${COMPACT_ROOT}/${dataset}"
  cache_path="${root_path}/qar_compact_shiftN80.npz"
  if [ ! -f "${cache_path}" ]; then
    failures=$((failures + 1))
    echo "[skip] ${dataset}: missing ${cache_path}"
    for model in ${MODELS}; do
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${model}" "2" "" "${root_path}" "" >> "${SUMMARY_FILE}"
    done
    continue
  fi

  enc_in=$("${PYTHON_BIN}" - <<PY
import numpy as np
c = np.load(r"${cache_path}", allow_pickle=False)
print(int(c["x"].shape[2]))
PY
)

  for model in ${MODELS}; do
    log_file="${SUMMARY_DIR}/${dataset}_${model}.log"
    echo "[run] dataset=${dataset} model=${model} enc_in=${enc_in} log=${log_file}"
    set +e
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -u run.py \
      --task_name anomaly_detection \
      --is_training "${IS_TRAINING}" \
      --model_id "${RUN_TAG}_${dataset}" \
      --model "${model}" \
      --data QAR_anomaly \
      --root_path "${root_path}" \
      --data_path qar_compact_shiftN80.npz \
      --features M \
      --seq_len "${SEQ_LEN}" \
      --pred_len 0 \
      --enc_in "${enc_in}" \
      --dec_in "${enc_in}" \
      --c_out "${enc_in}" \
      --d_model "${D_MODEL}" \
      --d_ff "${D_FF}" \
      --e_layers "${E_LAYERS}" \
      --d_layers 1 \
      --n_heads "${N_HEADS}" \
      --top_k 3 \
      --batch_size "${BATCH_SIZE}" \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      --learning_rate "${LEARNING_RATE}" \
      --seed "${SEED}" \
      --qar_split_boundary "${SPLIT_BOUNDARY}" \
      --itr "${ITR}" \
      --num_workers "${NUM_WORKERS}" \
      --gpu "${LOCAL_GPU}" \
      --checkpoints "${CHECKPOINTS}" \
      --des oneclass_val_threshold \
      --anomaly_threshold_source "${THRESHOLD_SOURCE}" \
      --anomaly_threshold_percentile "${THRESHOLD_PERCENTILE}" \
      --anomaly_score_direction "${SCORE_DIRECTION}" \
      --anomaly_level window \
      > "${log_file}" 2>&1
    status=$?
    set -e
    if (( status != 0 )); then failures=$((failures + 1)); fi
    printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${dataset}" "${model}" "${status}" "${enc_in}" "${root_path}" "${log_file}" >> "${SUMMARY_FILE}"
  done
done

echo "[done] summary: ${SUMMARY_FILE}"
if (( failures > 0 )); then exit 1; fi
