# Full-data deidentification

`tools/prepare/deidentify_tsfile_archive.py` processes the complete input
archive. It does not require old Parquet/CSV source archives or existing caches.
It needs Python core dependencies, a JDK, and the same TsFile reader/writer
JARs used by the normal preprocessing pipeline.

```bash
python tools/prepare/deidentify_tsfile_archive.py \
  --input-zip /private/input/tsfile_datasets.zip \
  --output-dir /release/deidentified \
  --private-dir /private/deidentification_audit \
  --iotdb-lib /software/iotdb-2.0.2-lib \
  --threads 4
```

Both output directories must be new and separate. The tool processes all
datasets and labels in the archive. `--limit-per-group 2` is an optional pilot
and must not be used for a release claiming complete coverage.

## What changes

- Original filenames become neutral sample IDs and artificial ordinal dates.
- Internal TsFile device identifiers are rebuilt under neutral sample IDs.
- Metadata indices and bloom filters are freshly written; original file
  properties and source-specific device metadata are not copied.
- ZIP entry dates become 1980-01-01; original comments and extra fields are
  removed. New ZIP64 size/offset records needed for archives above 4 GiB are
  retained as container bookkeeping. Public manifests contain no source
  filenames or original hashes.

Each artificial day represents one successive flight in the existing dataset
order, starting at 2000-01-01. These are ordinal codes, **not real flight dates**.
Assigning a distinct artificial day also preserves ordering with the existing
day-level time-key parser. Per-class order and the cross-class tie order used
by history assembly remain unchanged. Between-flight calendar gaps are lost.

## What is preserved and checked

Dataset IDs, normal/fault directory labels, measurement names, numeric values,
missing-value layouts, flight-phase values and source-relative nanosecond ticks
are retained. The tool fails on measurements outside the benchmark allowlist,
nonnumeric fields, multiple devices, unsupported timestamp scales or calendar
epoch timestamps. Source-relative counter offsets may exceed one flight; the
profile permits counters up to one year and spans up to 72 hours per file.

The implementation uses the Apache TsFile writer to copy compressed chunks and
rebuild metadata. It hashes schemas, chunk type masks, counts, time bounds and
all compressed time/value bytes before and after each file. Thus timestamps
and numeric chunks must match exactly; this does not introduce quantization,
noise, interpolation or normalization. A readback inspection verifies new
device identities. The completed ZIP also receives a complete CRC scan.

Implementation reference: Apache TsFile's
[`TsFileIOWriter.writeChunk`](https://apache.googlesource.com/tsfile/+/ea0945935a9bee5154072a8349bd4df7c9a04143/java/tsfile/src/main/java/org/apache/tsfile/write/writer/TsFileIOWriter.java).
The tool was compiled against the available 2.1.0 snapshot JAR; other TsFile
versions need their own compatibility check.

## Output separation

The distributable directory contains the deidentified ZIP, its SHA256 checksum,
README and a verification summary. The ZIP includes a manifest with only public
sample IDs, dataset IDs, binary labels and ordinals.

The **private** directory contains the original-to-public mapping, source
plan, per-file integrity checks and build status. Never upload this directory
with the dataset. The original archive also remains private.

## Privacy and numerical scope

This is direct-identifier removal with benchmark signals preserved. Aircraft
registrations, real dates, flight numbers, routes and maintenance text are
removed from names/paths. Duration and sensor patterns remain, including the
benchmark's altitude/airspeed channels where applicable. Matching external
records could permit linkage; this tool does not guarantee irreversible
anonymity or differential privacy.

The same population and preprocessing settings should produce identical
numeric caches, with renamed source fields and artificial time keys. This
does not repair legacy normalization, change labels, recover a missing original
paper population, or establish that all archived paper results are correct.
See [PROTOCOL.md](PROTOCOL.md) for those separate limitations.

To build experiment caches from the deidentified ZIP:

```bash
ZIP_PATH=/release/deidentified/tsfile_datasets_deidentified.zip \
IOTDB_LIB=/software/iotdb-2.0.2-lib \
DATA_ROOT=/work/generated_data \
bash scripts/reproduce/prepare_current_caches.sh
```
