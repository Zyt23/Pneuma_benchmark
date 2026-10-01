#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
CONTEXT60_ROOT="${CONTEXT60_ROOT:-${DATA_ROOT}/compact/context60}"
UNIVARIATE_ROOT="${UNIVARIATE_ROOT:-${DATA_ROOT}/compact/context60_univariate}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
ANCHORS="${ANCHORS:-predict_2_3 predict_4_5 predict_5_6 predict_8_9}"
MODELS="${MODELS:-Chronos2 TiRex2 Toto}"
TARGETS="${TARGETS:-0}"
RUN_TAG_PREFIX="${RUN_TAG_PREFIX:-qar_context60_unimulti}"

for anchor in ${ANCHORS}; do
  echo "[multivariate direct] anchor=${anchor}"
  COMPACT_ROOT="${CONTEXT60_ROOT}/hist5/${anchor}" \
  DATASETS="${DATASETS}" \
  MODELS="${MODELS}" \
  RUN_TAG="${RUN_TAG_PREFIX}_multivar_hist5_${anchor}_$(date +%Y%m%d_%H%M%S)" \
  SEQ_LEN=460 LABEL_LEN=20 PRED_LEN=20 \
  bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh

  for target in ${TARGETS}; do
    uni_root="${UNIVARIATE_ROOT}/hist5/${anchor}/target_${target}"
    echo "[univariate compact] anchor=${anchor} target=${target} -> ${uni_root}"
    "${PYTHON}" tools/prepare/project_qar_compact_univariate.py \
      --base_root "${CONTEXT60_ROOT}/hist5/${anchor}" \
      --output_root "${uni_root}" \
      --target "${target}" \
      --datasets "${DATASETS}"

    echo "[univariate run] anchor=${anchor} target=${target}"
    COMPACT_ROOT="${uni_root}" \
    DATASETS="${DATASETS}" \
    MODELS="${MODELS}" \
    RUN_TAG="${RUN_TAG_PREFIX}_univar_hist5_${anchor}_target${target}_$(date +%Y%m%d_%H%M%S)" \
    SEQ_LEN=460 LABEL_LEN=20 PRED_LEN=20 \
    bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
  done
done

echo "[note] For the paper table, run all feature indices and aggregate by stitching pred.npy tensors in feature-index order."
