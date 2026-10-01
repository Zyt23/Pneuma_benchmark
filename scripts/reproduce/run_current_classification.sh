#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
COMPACT_ROOT="${COMPACT_ROOT:-${DATA_ROOT}/compact/classification}" \
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}" \
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}" \
RUN_TAG="${RUN_TAG:-qar_current_classification_$(date +%Y%m%d_%H%M%S)}" \
bash scripts/classification/run_QAR_tsfile_shiftN80.sh
