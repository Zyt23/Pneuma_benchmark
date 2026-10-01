# Pneuma Context60 Evaluation Examples

This directory contains a small, reproducible example of the current Pneuma
protocol used for the paper tables.  It is intentionally not the full
benchmark runner: it keeps representative data, the evaluation contract, and
one or two smoke experiments for each task family.

The `dataset_demo` caches are generated toy data, not airline records. Example
tables illustrate software behavior and must not be reported as measured
benchmark results. The separate public TsFile subset is described in
[`docs/DATASETS.md`](../../docs/DATASETS.md).

## Current Protocol

- Phase anchors: `predict_2_3`, `predict_4_5`, `predict_5_6`, `predict_8_9`.
- One phase segment has length `80`.
- Main foundation-model table: `history_count = 5`.
- Forecast input: `history_count * 80` previous-flight points plus the first
  `60` points of the current flight.
- Forecast target: the next `20` points of the current flight.
- For `history_count = 5`, use `seq_len = 5 * 80 + 60 = 460`,
  `label_len = 20`, and `pred_len = 20`.
- The table metric is `loss = (MAE + RMSE) / 2`.
- Main-table rows are averaged over four phase anchors inside each dataset,
  then averaged over datasets.
- Main-table score is `100 / (1 + loss)` at dataset level, then averaged over
  datasets.
- The univariate experiment recomputes metrics from stitched tensors:
  every feature is forecast as a one-channel run, feature predictions are
  stacked back into `(N, pred_len, C)`, and MAE/RMSE/loss are computed once on
  that tensor.  The multivariate number is computed directly from the full
  `(N, pred_len, C)` prediction tensor.

The full protocol metadata is in `protocol_context60.json`.

## Smoke Run

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File examples/qar_context60/run_smoke.ps1
```

or on Linux/macOS:

```bash
bash examples/qar_context60/run_smoke.sh
```

The smoke run creates tiny representative compact caches and writes example
tables under:

```text
examples/qar_context60/outputs/smoke/
```

Expected outputs include:

- `main_table_example.csv`
- `main_anchor_detail_example.csv`
- `hist_analysis_example.csv`
- `uni_multi_example.csv`
- `classification_example.csv`
- `smoke_summary.json`

To test the real repository entry point (`run.py`) with the same tiny caches:

```bash
python examples/qar_context60/run_tslib_smoke.py --tasks forecast classification anomaly
```

This runs one-epoch DLinear examples for forecasting, classification, and
one-class anomaly detection.  It writes logs under
`examples/qar_context60/outputs/tslib_smoke/`.

## Using Real TsFile Data

The real-data path keeps the same compact-cache contract as the example data:

```bash
python tools/prepare/prepare_tsfile_compact_custom_conditions.py \
  --zip_path data/tsfile_datasets_subset10.zip \
  --task forecast \
  --forecast_output_root data/compact/forecast_segments \
  --datasets dataset5 dataset6 \
  --forecast_anchors predict_2_3 predict_4_5 predict_5_6 predict_8_9

python scripts/long_term_forecast/build_qar_history_forecast_compacts.py \
  --source_root data/compact/forecast_segments \
  --output_root data/compact/context60 \
  --datasets dataset5 dataset6 \
  --anchors predict_2_3 predict_4_5 predict_5_6 predict_8_9 \
  --history_counts 5 \
  --segment_len 80 \
  --current_context_len 60 \
  --target_len 20
```

Then evaluate or run models with the same loader settings:

```bash
SEQ_LEN=460 LABEL_LEN=20 PRED_LEN=20 \
COMPACT_ROOT=data/compact/context60/hist5/predict_2_3 \
DATASETS="dataset5" MODELS="Chronos2" \
bash scripts/long_term_forecast/run_QAR_tsfile_zero_shot_forecast_shiftN80.sh
```

Classification compact caches use the same `qar_compact_shiftN80.npz` filename
and can be built with `--task classification`.

The current full experiment launchers are collected under `scripts/reproduce/`.
