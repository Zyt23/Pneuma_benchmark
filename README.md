# Pneuma_benchmark

Pneuma_benchmark provides runnable benchmark code for aircraft pneumatic-system
time-series experiments, including forecasting, classification, and
anomaly-detection tasks.

The repository stores code, launch scripts, and small representative examples.
Large raw data, generated compact caches, checkpoints, logs, and result
workbooks are not tracked in git.

**Release scope:** the full-data deidentification pipeline preserves the available
input population. The exact submitted paper snapshot is not recovered. Read [protocol limitations](docs/PROTOCOL.md)
and [model provenance](docs/MODEL_IMPLEMENTATIONS.md) before interpreting
results. In particular, archived forecast normalization includes target
statistics, and several named models are local adapters. `MambaSL.py` now
contains the real selective-scan implementation; old convolution-fallback
weights do not apply.

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

These are core dependencies. Foundation models and other optional backbones
need separate packages/environments; see [ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

Obtain the full deidentified TsFile archive separately and place it under the
repository root. No complete dataset is hosted in this Git repository:

```text
data/
  tsfile_datasets_deidentified.zip
  iotdb-2.0.2-lib/
```

Cache preparation needs a JDK with `java` and `javac` on `PATH`. It does not
require installing or starting an IoTDB server. Install compatible TsFile
reader JARs under `data/iotdb-2.0.2-lib/`, or set `IOTDB_LIB` to their location.
They are software dependencies, not part of the deidentified data ZIP.

Run a full smoke test:

```bash
bash scripts/reproduce/run_quickstart_smoke.sh
```

The smoke test builds small compact caches from the TsFile input, then runs one
small forecasting task, one classification task, and one anomaly-detection task.
It limits preprocessing to 12 samples per class by default; it does not produce
paper-quality results. To check all datasets, set `DATASETS` explicitly.

## Data Labels

The deidentification tool removes original flight identifiers from filenames
and internal TsFile device metadata. Artificial dates encode order for
chronological splitting. Numeric signals and duration remain; see
[DEIDENTIFICATION.md](docs/DEIDENTIFICATION.md) for the privacy scope.

Normal and fault samples are identified by directory label:

```text
tsfile_datasets/<dataset>_tsfile/0/*.tsfile  # normal
tsfile_datasets/<dataset>_tsfile/1/*.tsfile  # fault
```

During cache preparation, these directory labels are stored in each compact
`.npz` file as the `labels` array. Classification uses labels `0` and `1`
directly. One-class anomaly detection trains on normal samples and evaluates
normal/fault separation.

## Current Forecast Protocol

The current foundation-model forecasting protocol uses:

- anchors: `predict_2_3`, `predict_4_5`, `predict_5_6`, `predict_8_9`;
- history count: `5`;
- input length: `5 * 80 + 60 = 460`;
- prediction length: `20`;
- loss: `(MAE + RMSE) / 2`;
- dataset score: average loss over the four anchors;
- per-dataset score: `100 / (1 + dataset_score)`;
- main Score: the mean of those scores across the nine datasets.

Traditional full-shot models use the current flight's **60->20**, not a
460-point input. Classification and anomaly main metrics are **P@R10**;
`ranking_metrics.json` records that definition separately from accuracy/F1.

The machine-readable protocol record is
`examples/qar_context60/protocol_context60.json`.

## Reproducing Experiments

Build all compact caches for the current protocol:

```bash
bash scripts/reproduce/prepare_current_caches.sh
```

This uses all available input samples (`MAX_PER_CLASS=0`), unlike smoke mode.
To process a private original archive, set `ZIP_PATH=/private/tsfile_datasets.zip`.
Old Parquet/CSV archives and prebuilt cache packages are not required.

Run the public experiment launchers:

```bash
bash scripts/reproduce/run_current_fullshot_forecast.sh
bash scripts/reproduce/run_current_main_forecast.sh
bash scripts/reproduce/run_current_hist_analysis.sh
bash scripts/reproduce/run_current_unimulti.sh
bash scripts/reproduce/run_current_classification.sh
bash scripts/reproduce/run_current_anomaly.sh
```

See `docs/QUICKSTART.md` for setup details, `scripts/reproduce/README.md` for
launcher descriptions, and `docs/DATASETS.md` for dataset scenario names.
Supplementary classifiers, anomaly baselines, event-window builders and Moirai
LoRA runners are indexed in [SUPPLEMENTARY.md](docs/SUPPLEMENTARY.md).
The full-data deidentification tool and its signal-preserving checks are
documented in [DEIDENTIFICATION.md](docs/DEIDENTIFICATION.md).

Use a distinct `RUN_TAG` (or `RUN_TAG_PREFIX` for multi-anchor wrappers) for
each seed/configuration. `SEED=2022` alone does not rename every output folder.

## Repository Layout

- `run.py`: main training and evaluation entry point.
- `requirements.txt`: Python dependencies.
- `data_provider/`: compact-cache loaders and standard TSLib loaders.
- `exp/`: task implementations.
- `models/`, `layers/`, `utils/`: model definitions and shared utilities.
- `scripts/reproduce/`: current protocol launchers.
- `scripts/classification/`, `scripts/long_term_forecast/`,
  `scripts/anomaly_detection/`: lower-level task launchers.
- `tools/prepare/`: TsFile-to-compact-cache builders.
- `tools/tsfile/`: Java TsFile window-dumping helpers.
- `tools/collect/`: result table collectors.
- `examples/qar_context60/`: tiny representative caches and smoke examples.
- `docs/`: setup and dataset notes.

## Notes

Generated caches are written under `data/compact/`, temporary extraction files
under `data/work/`, checkpoints under `checkpoints/`, logs under `logs/`, and
results under `results/`.

Some internal loader names, cache filenames, and legacy script names retain
their original identifiers for compatibility with existing experiment scripts.

Code licensing does not cover private airline records or external pretrained
weights. Raw internal archives, session transcripts, credentials and
manually constructed paper-like result tables are excluded from this repository.
The clearly labeled `dataset_demo` toy examples are retained for software checks.
