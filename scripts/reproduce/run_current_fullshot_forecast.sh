#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
FORECAST_SEGMENT_ROOT="${FORECAST_SEGMENT_ROOT:-${DATA_ROOT}/compact/forecast_segments}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
MODELS="${MODELS:-OLinear xPatch DUET TimeMixer TimeXer iTransformer DLinear PatchTST TimesNet Autoformer}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-qar_fullshot_forecast}"

for anchor in ${ANCHORS}; do
  COMPACT_ROOT="${FORECAST_SEGMENT_ROOT}/${anchor}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS}" \
  RUN_TAG="${RUN_TAG_PREFIX}_${anchor}_$(date +%Y%m%d_%H%M%S)" \
  SEQ_LEN=60 LABEL_LEN=20 PRED_LEN=20 \
  bash scripts/long_term_forecast/run_QAR_tsfile_forecast_shiftN80.sh
done
