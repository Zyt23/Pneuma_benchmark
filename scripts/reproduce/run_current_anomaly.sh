#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data}"
COMPACT_ROOT="${COMPACT_ROOT:-${DATA_ROOT}/compact/classification}" \
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}" \
MODELS="${MODELS:-KANAD AnomalyTransformer TranAD USAD OmniAnomaly}" \
RUN_TAG="${RUN_TAG:-qar_current_anomaly_$(date +%Y%m%d_%H%M%S)}" \
bash scripts/anomaly_detection/run_QAR_anomaly_shiftN80.sh
