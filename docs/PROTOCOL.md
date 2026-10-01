# Protocol and compatibility notes

This release makes the available code runnable and records its limitations. It
does **not** certify that every historical paper cell can be regenerated from
the public subset. Use a cache checksum, split indices, model implementation,
checkpoint and run configuration together when comparing experiments.

## Inputs and metrics

| Task | Input | Output | Reported metric |
| --- | --- | --- | --- |
| Traditional full-shot forecasting | Current flight: 60 points | Next 20 points | Error = (MAE + RMSE) / 2 |
| Foundation main, history 5 | Five previous 80-point segments + current 60 points = 460 | Next 20 points | Same Error |
| Foundation history sweep | 220 / 300 / 460 / 700 / 1020 points for history 2 / 3 / 5 / 8 / 12 | Next 20 points | Same Error |
| Classification | One flight sequence, usually 2235 x 16 | Binary flight label | Precision at fault recall 10% |
| One-class anomaly detection | Flight windows from the classification cache; normal training samples only | Flight anomaly score | Precision at fault recall 10% |

Classification exceptions: dataset11 has T=80; dataset13 has T=2920, C=20;
dataset14 has C=8. Anomaly runs have their own window/aggregation settings:
the QAR anomaly loader deterministically resamples each entire flight cache
to `seq_len` evenly spaced points (default launcher: 200), then averages the
reconstruction score within that flight. Record this resampling length, not
just the original cache shape. Smoke runs use 80 points instead.

Forecasting uses anchors `predict_2_3`, `predict_4_5`, `predict_5_6`,
`predict_8_9`. Average Error over the four anchors for each dataset. The main
Score is the macro average over nine datasets of `100 / (1 + dataset_Error)`;
it is not `100 / (1 + mean_dataset_Error)`. For classification/anomaly ranking
Score, take 100 times the nine-dataset mean precision.

`utils/benchmark_metrics.py` defines P@R10 as precision at the highest score
threshold whose fault recall reaches at least 0.10, including all score ties.
It reports achieved recall, TP and FP. Exact recall 0.10 may be unattainable
in a finite test set. A separately named precision-envelope statistic is also
saved. Accuracy, F1, Balanced
Accuracy and validation-calibrated operating thresholds are diagnostic metrics,
not interchangeable with P@R10. `ranking_metrics.json` uses test labels only to
evaluate a ranking curve, not to select a deployment threshold. Anomaly ranking
uses the raw score before point adjustment.

## Splits and seeds

Neural QAR loaders use chronological 70/10/20 splitting separately by class.
Only the training loader shuffles. Fixed `split_flags`, when supported and
present, take priority. The legacy neural boundary uses `int(n * (0.7 + 0.1))`;
floating-point rounding can differ from `8*n//10`. The release preserves this
as the default `legacy_float`; `QAR_SPLIT_BOUNDARY=integer` is an explicit new
protocol. ROCKET/aeon runners also expose `rocket_float` (`int(0.8*n)`) and
`floor_sum` for archived script compatibility. Record the convention and actual
indices; equal nominal percentages do not prove identical test samples.

`run.py --seed` now seeds Python, NumPy and PyTorch **after** parsing arguments;
iteration i uses seed+i and saves `run_config.json`. Previously the entry point
seeded 2021 before parsing, which could make a requested seed ineffective.
Historical standalone runners also have other defaults (ROCKET: 20260714;
legacy Moirai adaptation: 20260816). No claim is made that all submitted cells
used 2021 or three independent seeds. Preserve old logs and verify them.

Give every new run a distinct RUN_TAG, especially when changing SEED. Changing
the seed alone does not change the default result-directory name. Multi-seed
SD must come from actual independent runs (sample SD, ddof=1). Deterministic
zero-shot point estimators can legitimately have zero seed variance; inference
seed changes are not independent training runs.

## Legacy preprocessing limitations

The archived forecast segments were standardized using the entire 80-point
window before splitting context and target. Target values therefore influence
the normalization statistics. This is a **legacy, non-causal protocol**, not a
strict future-free forecasting evaluation. The release preserves it for
historical comparisons; it does not silently change paper numbers.

The old forecast code checked `QAR_FORECAST`, while launchers use
`QAR_forecast`, so its apparent normal-only filter was inactive. Current default
forecast loaders retain both labels. Use `NORMAL_ONLY=1` explicitly for a new
normal-only profile, including rebuilding foundation history caches; filtering
only the current flight does not remove fault flights from its history. This
choice still does not repair full-window normalization.

History construction orders available segments by time within a dataset. It
does not enforce same-aircraft histories. A 10% subset changes both
the available histories and the train/test population. It cannot reproduce
the full-data paper merely by using the same seed.

Flights with missing required channels/transitions or insufficient window
length are rejected. Nonfinite-value safeguards remain in the builder/loader;
their existence is not proof that an arbitrary input contains no missing data.

## Model identity

See [MODEL_IMPLEMENTATIONS.md](MODEL_IMPLEMENTATIONS.md). In particular, the
current `MambaSL.py` is a real selective-scan implementation, replacing a
convolution fallback under the same filename. Old fallback weights and scores
must not be attributed to this replacement. Full model retraining is required
to produce results for the replacement.
