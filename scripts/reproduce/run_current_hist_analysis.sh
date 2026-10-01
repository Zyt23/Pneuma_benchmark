#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
CONTEXT60_ROOT="${CONTEXT60_ROOT:-${DATA_ROOT}/compact/context60}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
HISTORY_COUNTS="${HISTORY_COUNTS:-2 3 5 8 12}"
MODELS="${MODELS:-Chronos2 TiRex2 Toto}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-qar_context60_hist_analysis}"

for hist in ${HISTORY_COUNTS}; do
  seq_len=$((hist * 80 + 60))
  for anchor in ${ANCHORS}; do
    echo "[hist] hist=${hist} anchor=${anchor} seq_len=${seq_len} models=${MODELS}"
    COMPACT_ROOT="${CONTEXT60_ROOT}/hist${hist}/${anchor}" \
    DATASETS="${DATASETS}" \
    MODELS="${MODELS}" \
    RUN_TAG="${RUN_TAG_PREFIX}_hist${hist}_${anchor}_$(date +%Y%m%d_%H%M%S)" \
    SEQ_LEN="${seq_len}" LABEL_LEN=20 PRED_LEN=20 \
    bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
  done
done
