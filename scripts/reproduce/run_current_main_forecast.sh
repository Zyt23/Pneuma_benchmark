#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
CONTEXT60_ROOT="${CONTEXT60_ROOT:-${DATA_ROOT}/compact/context60}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
MODELS="${MODELS:-Chronos2 TiRex2 TimesFM Toto Moirai}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-qar_context60_main}"

for anchor in ${ANCHORS}; do
  echo "[main] anchor=${anchor} models=${MODELS}"
  COMPACT_ROOT="${CONTEXT60_ROOT}/hist5/${anchor}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS}" \
  RUN_TAG="${RUN_TAG_PREFIX}_hist5_${anchor}_$(date +%Y%m%d_%H%M%S)" \
  SEQ_LEN=460 LABEL_LEN=20 PRED_LEN=20 \
  bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
done
