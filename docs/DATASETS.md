# Pneuma Dataset Mapping

The main launchers select the current protocol datasets listed below. For each
dataset, label directory `0` contains normal samples and label directory `1`
contains fault samples.

| Dataset | Paper label | Aircraft family | Fault scenario |
| --- | --- | --- | --- |
| dataset5 | 320-DIFF | A320 | pressure-sensing line fault |
| dataset6 | 320-HPV | A320 | HPV valve fault |
| dataset7 | 320-PRV | A320 | PRV valve fault |
| dataset8 | 320-WEAV | A320 | duct leakage |
| dataset8-1 | 320-LOW | A320 | PRV valve leakage |
| dataset10 | 321-DIFF | A321 | pressure-sensing line fault |
| dataset11 | 321-WEAV | A321 | duct leakage |
| dataset13 | 787-CAC | 787 | air-compressor fault |
| dataset14 | 777-PRSOV | 777 | PRSOV fault |

The deidentification tool replaces original filenames and internal device IDs.
Dataset names, `0`/`1` labels and sample ordinals suffice for the main launchers.

## Archive coverage and compatibility

The full deidentified archive retains all input flights and labels, including
`dataset9` and `dataset12`, which the main tables exclude. Window extraction
rejects flights that cannot provide a complete window.

The available `dataset13` records differ from the subset used in the submitted
manuscript. Exact reproduction requires the original selection manifest, which
was unavailable during release preparation.

## Deidentification and access

Artificial dates in the new deidentified filenames encode only sample order;
real calendar dates, routes, registrations and maintenance identities are
removed from names and device paths. Flight duration remains observable in
the relative within-flight ticks. The code license does not grant
redistribution rights to private raw records. Data access/reuse terms must
come from the data provider separately.

The historical subset renamed ZIP entries but retained source identifiers in
TsFile device metadata. The full-archive procedure removes these identifiers
as described in [DEIDENTIFICATION.md](DEIDENTIFICATION.md).
