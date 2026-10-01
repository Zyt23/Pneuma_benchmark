# Quick Start

This guide starts from a fresh clone and a full deidentified Pneuma TsFile archive.

## 1. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` lists Python packages only. Building compact caches from raw
TsFile data also needs a JDK with `java` and `javac`.
For optional model packages and the tested core environment, see
[ENVIRONMENTS.md](ENVIRONMENTS.md).

You do not need to install or start an IoTDB server. The benchmark only needs
compatible TsFile reader JARs installed under `data/iotdb-2.0.2-lib/`, or a
directory selected with `IOTDB_LIB`. The JARs are separate software dependencies.

## 2. Place Data

Obtain the full deidentified archive separately. Place the ZIP under the
repository root without extracting it:

```text
data/
  tsfile_datasets_deidentified.zip
  iotdb-2.0.2-lib/
```

The `0/` directory contains normal samples and the `1/` directory contains
fault samples for each dataset. Original flight identifiers are removed from
both filenames and internal device metadata by the deidentification tool.
For a private original ZIP, set `ZIP_PATH` to its location instead.

## 3. Run The Smoke Test

```bash
bash scripts/reproduce/run_quickstart_smoke.sh
```

This command reads `data/tsfile_datasets_deidentified.zip`, uses the Java TsFile
helpers and `data/iotdb-2.0.2-lib/` to extract fixed windows, writes compact
`.npz` caches under `data/compact/`, and runs one small forecasting,
classification, and anomaly-detection experiment.
By default it preprocesses at most 12 flights per class from dataset5 and runs
one epoch. Use a unique `RUN_PREFIX` to preserve previous smoke outputs.

Outputs are written under:

```text
data/compact/
data/work/
checkpoints/
logs/
results/
```

## 4. Run Larger Experiments

After caches are prepared, use the launchers in `scripts/reproduce/`:

```bash
bash scripts/reproduce/prepare_current_caches.sh
bash scripts/reproduce/run_current_fullshot_forecast.sh
bash scripts/reproduce/run_current_main_forecast.sh
bash scripts/reproduce/run_current_hist_analysis.sh
bash scripts/reproduce/run_current_unimulti.sh
bash scripts/reproduce/run_current_classification.sh
bash scripts/reproduce/run_current_anomaly.sh
```

Common overrides include `DATASETS`, `MODELS`, `DATA_ROOT`, `PYTHON`, and
`CUDA_DEVICES`.
Also set `SEED` and a unique `RUN_TAG`/`RUN_TAG_PREFIX` when comparing seeds.
For a core-model CPU check, use the tested CPU-only PyTorch environment.

The available full archive is not a recovered exact submitted-paper snapshot.
Read [PROTOCOL.md](PROTOCOL.md) before comparing its numbers with the paper.
Do not mix a changed split, normal-only filtering, or rebuilt normalization
with legacy scores under a single protocol label.

## Data Flow

```text
raw TsFile zip -> deterministic compact cache -> run.py training/evaluation
```

The raw TsFile zip remains the source data. Compact caches make repeated runs
deterministic and faster, while preserving `sources`, `time_keys`, and labels
for chronological splitting.
