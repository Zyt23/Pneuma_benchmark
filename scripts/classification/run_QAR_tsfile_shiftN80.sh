#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA_ROOT="${DATA_ROOT:-data}"
DATASETS="${DATASETS:-dataset5 dataset6 dataset7 dataset8 dataset8-1 dataset10 dataset11 dataset13 dataset14}"
MODELS="${MODELS:-Transformer TimesNet PatchTST DLinear iTransformer}"
RUN_TAG="${RUN_TAG:-qar_classification_shiftN80_$(date +%Y%m%d_%H%M%S)}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-50}"
SEED="${SEED:-2021}"
ITR="${ITR:-1}"
PATIENCE="${PATIENCE:-3}"
BATCH_SIZE="${BATCH_SIZE:-128}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"
DEVICES="${DEVICES:-0,1}"
USE_MULTI_GPU="${USE_MULTI_GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SPLIT_BOUNDARY="${SPLIT_BOUNDARY:-legacy_float}"
SAVE_EPOCH_CHECKPOINTS="${SAVE_EPOCH_CHECKPOINTS:-0}"
CLASS_WEIGHT="${CLASS_WEIGHT:-balanced}"
EARLY_STOP_METRIC="${EARLY_STOP_METRIC:-macro_f1}"
QAR_SPLIT_STRATEGY="${QAR_SPLIT_STRATEGY:-per_class_chrono}"
PATCH_LEN="${PATCH_LEN:-16}"
STRIDE="${STRIDE:-8}"
D_MODEL="${D_MODEL:-64}"
D_FF="${D_FF:-128}"
E_LAYERS="${E_LAYERS:-3}"
N_HEADS="${N_HEADS:-8}"
SEQ_LEN="${SEQ_LEN:-2000}"
COMPACT_ROOT="${COMPACT_ROOT:-${DATA_ROOT}/compact/classification}"
CHECKPOINTS="${CHECKPOINTS:-./checkpoints/classification/${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-./logs/classification/${RUN_TAG}}"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/summary.tsv}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"

if ! command -v -- "${PYTHON}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}" "${CHECKPOINTS}"
if [[ ! -f "${SUMMARY_FILE}" ]]; then
  printf 'dataset\tmodel\tstatus\tlog\tresult_dir\n' > "${SUMMARY_FILE}"
fi

echo "Run tag: ${RUN_TAG}"
echo "Datasets: ${DATASETS}"
echo "Models: ${MODELS}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "USE_MULTI_GPU: ${USE_MULTI_GPU}"
echo "CLASS_WEIGHT: ${CLASS_WEIGHT}"
echo "EARLY_STOP_METRIC: ${EARLY_STOP_METRIC}"
echo "QAR_SPLIT_STRATEGY: ${QAR_SPLIT_STRATEGY}"
echo "Seeds: base=${SEED}, repetitions=${ITR}"
echo "PATCH_LEN: ${PATCH_LEN}"
echo "STRIDE: ${STRIDE}"
echo "MODEL_SHAPE: d_model=${D_MODEL} d_ff=${D_FF} e_layers=${E_LAYERS} n_heads=${N_HEADS}"
echo "Checkpoints: ${CHECKPOINTS}"
echo "Logs: ${LOG_DIR}"

GPU_ARGS=()
if [[ "${USE_MULTI_GPU}" != "0" ]]; then
  GPU_ARGS+=(--use_multi_gpu --devices "${DEVICES}")
fi

failures=0
for dataset in ${DATASETS}; do
  root_path="${COMPACT_ROOT}/${dataset}/"
  cache_path="${root_path}/qar_compact_shiftN80.npz"
  if [[ ! -f "${cache_path}" ]]; then
    echo "Missing compact cache: ${cache_path}" >&2
    exit 1
  fi

  for model in ${MODELS}; do
    des="${RUN_TAG}_${dataset}_${model}"
    model_id="${dataset}_QAR_shiftN80"
    log_file="${LOG_DIR}/${dataset}_${model}.log"
    echo "[$(date '+%F %T')] START dataset=${dataset} model=${model}"

    set +e
    "${PYTHON}" -u run.py \
      --task_name classification \
      --is_training 1 \
      --root_path "${root_path}" \
      --qar_split_strategy "${QAR_SPLIT_STRATEGY}" \
      --qar_split_boundary "${SPLIT_BOUNDARY}" \
      --model_id "${model_id}" \
      --model "${model}" \
      --data QAR_shift \
      --phase_a_shift -80 \
      --seq_len "${SEQ_LEN}" \
      --e_layers "${E_LAYERS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --d_model "${D_MODEL}" \
      --d_ff "${D_FF}" \
      --n_heads "${N_HEADS}" \
      --top_k 5 \
      --num_kernels 6 \
      --patch_len "${PATCH_LEN}" \
      --stride "${STRIDE}" \
      --dropout 0.2 \
      --lradj cosine \
      --des "${des}" \
      --seed "${SEED}" \
      --itr "${ITR}" \
      --learning_rate 0.001 \
      --train_epochs "${TRAIN_EPOCHS}" \
      --patience "${PATIENCE}" \
      "${GPU_ARGS[@]}" \
      --checkpoints "${CHECKPOINTS}" \
      --save_epoch_checkpoints "${SAVE_EPOCH_CHECKPOINTS}" \
      --class_weight "${CLASS_WEIGHT}" \
      --early_stop_metric "${EARLY_STOP_METRIC}" \
      > "${log_file}" 2>&1
    status=$?
    set -e

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
