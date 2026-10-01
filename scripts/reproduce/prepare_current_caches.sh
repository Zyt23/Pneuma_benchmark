#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
ZIP_PATH="${ZIP_PATH:-${DATA_ROOT}/tsfile_datasets_deidentified.zip}"
IOTDB_LIB="${IOTDB_LIB:-${DATA_ROOT}/iotdb-2.0.2-lib}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
HISTORY_COUNTS="${HISTORY_COUNTS:-2 3 5 8 12}"
MAX_PER_CLASS="${MAX_PER_CLASS:-0}"

CLASSIFICATION_ROOT="${CLASSIFICATION_ROOT:-${DATA_ROOT}/compact/classification}"
FORECAST_SEGMENT_ROOT="${FORECAST_SEGMENT_ROOT:-${DATA_ROOT}/compact/forecast_segments}"
CONTEXT60_ROOT="${CONTEXT60_ROOT:-${DATA_ROOT}/compact/context60}"
HISTORY_ARGS=()
if [[ "${NORMAL_ONLY:-0}" == "1" ]]; then
  HISTORY_ARGS+=(--normal_only)
fi

echo "[classification compact] -> ${CLASSIFICATION_ROOT}"
"${PYTHON}" tools/prepare/prepare_tsfile_compact_custom_conditions.py \
  --zip_path "${ZIP_PATH}" \
  --task classification \
  --max_per_class "${MAX_PER_CLASS}" \
  --classification_output_root "${CLASSIFICATION_ROOT}" \
  --datasets ${DATASETS} \
  --iotdb_lib "${IOTDB_LIB}"

echo "[forecast phase segments] -> ${FORECAST_SEGMENT_ROOT}"
"${PYTHON}" tools/prepare/prepare_tsfile_compact_custom_conditions.py \
  --zip_path "${ZIP_PATH}" \
  --task forecast \
  --max_per_class "${MAX_PER_CLASS}" \
  --forecast_output_root "${FORECAST_SEGMENT_ROOT}" \
  --forecast_anchors ${ANCHORS} \
  --datasets ${DATASETS} \
  --iotdb_lib "${IOTDB_LIB}"

echo "[context60 history caches] -> ${CONTEXT60_ROOT}"
"${PYTHON}" scripts/long_term_forecast/build_qar_history_forecast_compacts.py \
  --source_root "${FORECAST_SEGMENT_ROOT}" \
  --output_root "${CONTEXT60_ROOT}" \
  --datasets "${DATASETS}" \
  --anchors "${ANCHORS}" \
  --history_counts ${HISTORY_COUNTS} \
  --segment_len 80 \
  --current_context_len 60 \
  --target_len 20 \
  "${HISTORY_ARGS[@]}"

echo "[done]"
echo "CLASSIFICATION_ROOT=${CLASSIFICATION_ROOT}"
echo "FORECAST_SEGMENT_ROOT=${FORECAST_SEGMENT_ROOT}"
echo "CONTEXT60_ROOT=${CONTEXT60_ROOT}"
