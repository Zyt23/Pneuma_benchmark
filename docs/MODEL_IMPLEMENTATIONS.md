# Model implementation provenance

Paper names identify the intended comparison family; they do not by themselves
establish that an adapter is the author's official implementation.

| Repository module | Implementation status |
| --- | --- |
| `models/MambaSL.py` | Former `MambaSingleLayer.py`, renamed to the canonical MambaSL entry point. Uses `layers.MambaBlock.Mamba_TimeVariant`, which calls `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`. No convolution fallback. |
| `models/VSFormer.py` | Local value/temporal-difference transformer adapter; omits the upstream motif/statistical-prior pipeline. Not an exact official VSFormer reproduction. |
| `models/LITE.py` | Local PyTorch LITE-inspired classifier; not the official aeon TensorFlow implementation. |
| `models/OLinear.py` | Local adapter with an identity/no-Q default, not the complete upstream Q-matrix preprocessing pipeline. |
| `models/ModernTCNAD.py` | Reconstruction-task adapter using the included ModernTCN layer; adapted task/loss settings must be reported. |
| MiniROCKET / MultiROCKET | aeon transforms + StandardScaler + balanced RidgeClassifierCV; separate sklearn-style runner. |
| HIVE-COTE2 / InceptionTime | aeon classifiers; supplementary runners with explicit seed and split options. |
| Chronos2 / TiRex2 / TimesFM / Toto / Moirai | Wrappers around external packages and separately downloaded pretrained weights. Pin both package revision and model revision. |

The deleted Mamba convolution fallback used gated depthwise convolutions and
mean/last pooling. It was not a Mamba state-space model. Its archived outputs
are not validation of the current `MambaSL.py`, and its checkpoints are not
compatible with the replacement. The current SSM path was syntax-checked but
was not trained or CUDA-validated during release preparation because the
validation environment lacks `mamba_ssm`. Install that dependency explicitly;
missing dependencies must fail visibly rather than select another architecture.

The retained VSFormer/LITE/OLinear adapters should be described as adapters in
comparisons. Replacing them with official code is a separate experiment, not
a filename-only fix. Other retained model modules have not all been audited
line by line against their upstream publications.

ModernTCN layer provenance: upstream local snapshot commit
`56a9a2c018385cd5acef015378cae7f084d1b11c`; its MIT notice is retained in
[`licenses/ModernTCN.txt`](../licenses/ModernTCN.txt). The repository's existing
license and upstream notices remain applicable to their respective code.
The code license does not grant rights to airline records or model weights.
