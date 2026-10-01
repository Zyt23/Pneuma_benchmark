#!/usr/bin/env bash
set -euo pipefail

# End-to-end smoke test for a fresh local setup.
#
# Expected local layout after downloading the public data package:
#   data/tsfile_datasets_deidentified.zip
#   data/iotdb-2.0.2-lib/
#
# This script builds small compact caches from the TsFile input and runs one small
# experiment for forecasting, classification, and anomaly detection.

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
ZIP_PATH="${ZIP_PATH:-${DATA_ROOT}/tsfile_datasets_deidentified.zip}"
DATASETS="${DATASETS:-dataset5}"
RUN_PREFIX="${RUN_PREFIX:-quickstart_smoke_$(date +%Y%m%d_%H%M%S)}"

echo "[1/4] Build compact caches from ${ZIP_PATH}"
PYTHON="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" \
ZIP_PATH="${ZIP_PATH}" \
DATASETS="${DATASETS}" \
HISTORY_COUNTS="5" \
MAX_PER_CLASS="${MAX_PER_CLASS:-12}" \
bash scripts/reproduce/prepare_current_caches.sh

echo "[2/4] Forecast smoke"
PYTHON="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" \
COMPACT_ROOT="${DATA_ROOT}/compact/forecast_segments/predict_2_3" \
DATASETS="${DATASETS}" \
MODELS="DLinear" \
RUN_TAG="${RUN_PREFIX}_forecast" \
SEQ_LEN=60 LABEL_LEN=20 PRED_LEN=20 \
TRAIN_EPOCHS=1 PATIENCE=1 BATCH_SIZE=16 NUM_WORKERS=0 \
USE_MULTI_GPU=0 CUDA_DEVICES="${CUDA_DEVICES:-0}" \
CHECKPOINTS="./checkpoints/${RUN_PREFIX}/forecast" \
LOG_DIR="./logs/${RUN_PREFIX}/forecast" \
bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh

echo "[3/4] Classification smoke"
PYTHON="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" \
COMPACT_ROOT="${DATA_ROOT}/compact/classification" \
DATASETS="${DATASETS}" \
MODELS="DLinear" \
RUN_TAG="${RUN_PREFIX}_classification" \
TRAIN_EPOCHS=1 PATIENCE=1 BATCH_SIZE=16 \
USE_MULTI_GPU=0 CUDA_DEVICES="${CUDA_DEVICES:-0}" \
CHECKPOINTS="./checkpoints/${RUN_PREFIX}/classification" \
LOG_DIR="./logs/${RUN_PREFIX}/classification" \
bash scripts/reproduce/run_current_classification.sh

echo "[4/4] Anomaly-detection smoke"
PYTHON="${PYTHON}" \
PYTHON_BIN="${PYTHON}" \
DATA_ROOT="${DATA_ROOT}" \
COMPACT_ROOT="${DATA_ROOT}/compact/classification" \
DATASETS="${DATASETS}" \
MODELS="AnomalyTransformer" \
RUN_TAG="${RUN_PREFIX}_anomaly" \
SEQ_LEN=80 TRAIN_EPOCHS=1 PATIENCE=1 BATCH_SIZE=16 NUM_WORKERS=0 \
CUDA_DEVICE="${CUDA_DEVICES:-0}" LOCAL_GPU=0 \
CHECKPOINTS="./checkpoints/${RUN_PREFIX}/anomaly" \
SUMMARY_DIR="./logs/${RUN_PREFIX}/anomaly" \
bash scripts/reproduce/run_current_anomaly.sh

echo "[done] summaries are under logs/${RUN_PREFIX}/"
