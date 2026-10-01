# Environments and validation

Use separate environments for the core benchmark and foundation models.
`requirements.txt` installs the core training/data utilities; it is not an
assertion that every optional model can coexist in that environment.

## Tested core environment

The release smoke run used Linux, Python 3.10.12, PyTorch 2.4.1+cpu,
NumPy 2.2.6, SciPy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2,
matplotlib 3.10.9, einops 0.8.2, PyWavelets 1.8.0 and datasets 2.21.0.
Raw TsFile conversion additionally needs a JDK (`java`, `javac`) and the reader
JARs installed as a separate software dependency. No running IoTDB service is needed.

On 2026-11-12, the historical TsFile subset passed ZIP CRC validation. A small
pilot selection was converted for all nine main datasets. One-epoch runs of
DLinear forecasting (60->20), DLinear classification and AnomalyTransformer
detection completed:
27/27 successful task statuses. These are installation/shape/metric checks,
**not paper replications or performance estimates**. Anomaly smoke windows
are 80 points; they do not establish full-sequence anomaly performance.

On 2026-11-14, the full deidentified archive passed schema and compressed
time/value chunk comparisons for every file, as well as a complete ZIP CRC
scan. For identical pilot populations, 45 cache pairs across the
nine main datasets (classification and four forecast anchors) had identical
numeric arrays, masks, labels and source order. Only source identifiers and
artificial order dates changed. Thirteen protocol/metadata tests passed.
This checks the transformation, not model performance or paper-score recovery.

Run protocol unit tests and a source audit with:

```bash
python -m unittest discover -s tests -v
python tools/audit/release_audit.py
```

The scanner checks Python and Bash syntax and recognizable accidental secret
patterns. It is not a full security, licensing or model-equivalence audit.

## Optional dependencies

| Model / runner | Additional environment requirement |
| --- | --- |
| MambaSL | Compatible CUDA/PyTorch build of `mamba_ssm`; no fallback |
| MiniROCKET / MultiROCKET / HIVE-COTE2 | `aeon`; record its version and scikit-learn/NumPy versions |
| InceptionTime | aeon plus a compatible TensorFlow/Keras installation |
| CSDI | Upstream source checkout supplied with `--csdi-root`; see runner help |
| DCdetector | Upstream source checkout supplied with `--source-root` |
| MOMENT | `momentfm` and separately obtained pretrained weights |
| Chronos2 | Chronos package exposing `Chronos2Pipeline` |
| TiRex2 | `tirex-2` package exposing `tirex2`; matching TiRex-2 checkpoint |
| TimesFM | TimesFM 2.5 implementation exposing `TimesFM_2p5_200M_torch` and `ForecastConfig`; old `timesfm==1.3.0` is not sufficient |
| Toto | `toto-models` exposing `toto2`; matching model revision |
| Moirai / Moirai LoRA | Compatible Uni2TS environment and the matching 2.0 / 1.0 / 1.1 model files |

Some other inherited TSLib models need additional packages such as
`sktime`, `reformer-pytorch` or `local-attention`. Imports are lazy: core
models do not require installing every optional model dependency.

Historical foundation runs used separate Python 3.11/3.12 environments with
PyTorch 2.9.1. Those environments are a provenance reference, not a verified
fresh-install lockfile. No full foundation-weight inference or supplementary
model training was rerun during this release check. Download weights under
their upstream terms and pass local paths where supported; private server
paths and credentials are not part of this repository.
