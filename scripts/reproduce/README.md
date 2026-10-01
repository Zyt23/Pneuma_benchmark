# Reproduction Launchers

These scripts are the public entry points for reproducing the current Pneuma
experiments.  Older exploratory sweeps and repair scripts are intentionally not
kept in this repository.
Supplementary runners recovered for release are listed in
`docs/SUPPLEMENTARY.md`; these entry points do not imply that every historical
paper result has been independently reproduced.

Run them from the repository root.  Override paths and model lists through
environment variables.

Default data layout:

```text
data/
  tsfile_datasets_deidentified.zip
  iotdb-2.0.2-lib/
```

Generated caches use `data/compact/`.

Cache preparation needs a JDK with `java` and `javac`.  It does not require
installing or starting an IoTDB server; the required TsFile reader jars should
be installed under `data/iotdb-2.0.2-lib/` or selected with `IOTDB_LIB`.

1. `prepare_current_caches.sh`
   Builds classification compact caches, four-anchor forecast segment caches,
   and context60 history caches from a TsFile zip.

2. `run_current_main_forecast.sh`
   Runs the current context60 main-table foundation-model forecast protocol:
   hist5, four anchors, nine datasets, and the five foundation models.

3. `run_current_hist_analysis.sh`
   Runs the lookback/history analysis for hists `2 3 5 8 12` with
   `Chronos2 TiRex2 Toto`.

4. `run_current_unimulti.sh`
   Launches the multivariate direct runs and prepares/runs per-feature
   univariate compact caches for the stitched univariate-vs-multivariate
   analysis.  Final aggregation should recompute metrics from `pred.npy` and
   `true.npy`.

5. `run_current_classification.sh`
   Runs classification on the current classification compact caches.
   The default is a small neural model selection. To select the seven neural
   Table 4 names explicitly, use `MODELS="MambaSL VSFormer LITE TimesNet
   PatchTST DLinear iTransformer"`; consult model-provenance warnings first.
   MiniROCKET/MultiROCKET use the separate runner in `docs/SUPPLEMENTARY.md`.

6. `run_current_anomaly.sh`
   Runs one-class anomaly detection on the current classification compact
   caches.

7. `run_quickstart_smoke.sh`
   Builds caches from the default `data/` package and runs one tiny forecasting,
   classification, and anomaly-detection experiment.

8. `run_current_fullshot_forecast.sh`
   Traditional full-shot models, current-flight 60->20, four anchors.

The smoke default is at most 12 samples per class before window rejection;
the preparation default outside smoke is all available input samples. The input
defaults to the full deidentified ZIP; `ZIP_PATH` also accepts a private original.
Core seed, split-boundary
and normal-only controls are documented in `docs/PROTOCOL.md`. Keep separate
RUN_TAGs for each seed to avoid overwriting outputs.
