#!/usr/bin/env python
"""Create a proportional TsFile zip subset for public benchmark release.

The output keeps the original TsFile zip layout:

    tsfile_datasets/<dataset>_tsfile/<label>/<file>.tsfile

Rows are selected independently for each dataset/label group.  The default
selection is chronological, evenly spaced, and approximately 10% per group.
Manifest files record the split that the benchmark loader will use later:
per-class chronological TRAIN/VAL/TEST = 70/10/20.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import shutil
import zipfile
from pathlib import Path


DATASET_RE = re.compile(r"^tsfile_datasets/(dataset(?:\d+)(?:-\d+)?)_tsfile/([01])/(.+\.tsfile)$")
DEFAULT_DATASETS = [
    "dataset5",
    "dataset6",
    "dataset7",
    "dataset8",
    "dataset8-1",
    "dataset10",
    "dataset11",
    "dataset13",
    "dataset14",
]


def time_key_from_source(source: str) -> int:
    base = os.path.basename(str(source).replace("\\", "/"))
    candidates: list[tuple[int, int, int, int, int, int]] = []

    def add(year, month, day, hour=0, minute=0, second=0) -> None:
        try:
            year = int(year)
            month = int(month)
            day = int(day)
            hour = int(hour)
            minute = int(minute)
            second = int(second)
        except Exception:
            return
        if 2000 <= year <= 2099 and 1 <= month <= 12 and 1 <= day <= 31:
            candidates.append((year, month, day, hour, minute, second))

    for match in re.finditer(r"(20\d{2})[-_](\d{2})[-_](\d{2})[ T_](\d{2})[-_:](\d{2})[-_:](\d{2})", base):
        add(*match.groups())
    for match in re.finditer(r"(20\d{2})(\d{2})(\d{2})[_-](\d{2})(\d{2})(\d{2})", base):
        add(*match.groups())
    for match in re.finditer(r"(20\d{2})[-_](\d{2})[-_](\d{2})", base):
        add(*match.groups())
    for match in re.finditer(r"(20\d{2})(\d{2})(\d{2})", base):
        add(*match.groups())

    if not candidates:
        return 99999999999999
    year, month, day, hour, minute, second = min(candidates)
    return int(f"{year:04d}{month:02d}{day:02d}{hour:02d}{minute:02d}{second:02d}")


def split_bounds_keep_test(n: int) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    if n == 1:
        return 0, 0
    if n == 2:
        return 1, 1
    train_end = int(n * 0.7)
    val_end = int(n * 0.8)
    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))
    return train_end, val_end


def split_name(index: int, n: int) -> str:
    train_end, val_end = split_bounds_keep_test(n)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def select_evenly(items: list[zipfile.ZipInfo], ratio: float, min_per_group: int) -> list[tuple[int, zipfile.ZipInfo]]:
    n = len(items)
    if n == 0:
        return []
    k = int(round(n * ratio))
    k = max(min_per_group, k)
    k = min(n, k)
    if k == n:
        return list(enumerate(items))
    if k == 1:
        return [(0, items[0])]
    positions = []
    seen = set()
    for step in range(k):
        idx = int(round(step * (n - 1) / (k - 1)))
        while idx in seen and idx + 1 < n:
            idx += 1
        while idx in seen and idx - 1 >= 0:
            idx -= 1
        seen.add(idx)
        positions.append(idx)
    positions.sort()
    return [(idx, items[idx]) for idx in positions]


def scan_zip(zip_path: Path, datasets: set[str]) -> dict[tuple[str, str], list[zipfile.ZipInfo]]:
    groups: dict[tuple[str, str], list[zipfile.ZipInfo]] = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            match = DATASET_RE.match(name)
            if not match:
                continue
            dataset, label, _ = match.groups()
            if dataset not in datasets:
                continue
            groups.setdefault((dataset, label), []).append(info)
    for key in groups:
        groups[key].sort(key=lambda item: (time_key_from_source(item.filename), item.filename))
    return groups


def compression_args(name: str, level: int) -> dict:
    if name == "stored":
        return {"compression": zipfile.ZIP_STORED}
    if name == "deflated":
        if hasattr(zipfile, "ZIP_DEFLATED"):
            return {"compression": zipfile.ZIP_DEFLATED, "compresslevel": level}
    raise ValueError(f"Unsupported compression: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a proportional QAR TsFile subset zip.")
    parser.add_argument("--zip_path", type=Path, required=True)
    parser.add_argument("--output_zip", type=Path, required=True)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--min_per_group", type=int, default=3)
    parser.add_argument("--compression", choices=["stored", "deflated"], default="deflated")
    parser.add_argument("--compresslevel", type=int, default=1)
    args = parser.parse_args()

    if not (0 < args.ratio <= 1):
        raise ValueError("--ratio must be in (0, 1]")
    if not args.zip_path.exists():
        raise FileNotFoundError(args.zip_path)

    datasets = set(args.datasets)
    groups = scan_zip(args.zip_path, datasets)
    missing = []
    for dataset in args.datasets:
        for label in ("0", "1"):
            if (dataset, label) not in groups:
                missing.append(f"{dataset}/{label}")
    if missing:
        raise ValueError("Missing dataset/label groups: " + ", ".join(missing))

    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    summary_rows = []
    selected_total = 0
    original_total = 0

    zip_kwargs = compression_args(args.compression, args.compresslevel)
    with zipfile.ZipFile(args.zip_path) as src, zipfile.ZipFile(args.output_zip, "w", **zip_kwargs) as dst:
        for dataset in args.datasets:
            for label in ("0", "1"):
                items = groups[(dataset, label)]
                selected = select_evenly(items, args.ratio, args.min_per_group)
                original_total += len(items)
                selected_total += len(selected)
                train_end, val_end = split_bounds_keep_test(len(selected))
                split_counts = {"train": 0, "val": 0, "test": 0}
                for subset_idx, (original_idx, info) in enumerate(selected):
                    with src.open(info) as handle, dst.open(info.filename, "w") as output:
                        shutil.copyfileobj(handle, output, length=1024 * 1024)
                    split = split_name(subset_idx, len(selected))
                    split_counts[split] += 1
                    manifest_rows.append(
                        {
                            "dataset": dataset,
                            "label": label,
                            "split": split,
                            "source_member": info.filename,
                            "time_key": time_key_from_source(info.filename),
                            "original_index_in_group": original_idx,
                            "subset_index_in_group": subset_idx,
                            "original_group_count": len(items),
                            "subset_group_count": len(selected),
                            "file_size": info.file_size,
                            "compress_size": info.compress_size,
                        }
                    )
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "label": label,
                        "original_count": len(items),
                        "subset_count": len(selected),
                        "ratio": len(selected) / len(items) if items else math.nan,
                        "train": train_end,
                        "val": val_end - train_end,
                        "test": len(selected) - val_end,
                        "split_train_count": split_counts["train"],
                        "split_val_count": split_counts["val"],
                        "split_test_count": split_counts["test"],
                    }
                )

        manifest_text = rows_to_csv(manifest_rows)
        summary_text = rows_to_csv(summary_rows)
        readme_text = build_readme(args, original_total, selected_total)
        dst.writestr("QAR_TSFILE_SUBSET10_README.md", readme_text)
        dst.writestr("QAR_TSFILE_SUBSET10_MANIFEST.csv", manifest_text)
        dst.writestr("QAR_TSFILE_SUBSET10_SUMMARY.csv", summary_text)

    manifest_path = args.output_zip.with_name(args.output_zip.stem + "_manifest.csv")
    summary_path = args.output_zip.with_name(args.output_zip.stem + "_summary.csv")
    readme_path = args.output_zip.with_name(args.output_zip.stem + "_README.md")
    manifest_path.write_text(rows_to_csv(manifest_rows), encoding="utf-8")
    summary_path.write_text(rows_to_csv(summary_rows), encoding="utf-8")
    readme_path.write_text(build_readme(args, original_total, selected_total), encoding="utf-8")

    print(f"Wrote {args.output_zip}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {summary_path}")
    print(f"Selected {selected_total}/{original_total} files ({selected_total / original_total:.4%})")


def rows_to_csv(rows: list[dict]) -> str:
    if not rows:
        return ""
    import io

    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def build_readme(args: argparse.Namespace, original_total: int, selected_total: int) -> str:
    datasets = " ".join(args.datasets)
    return f"""# QAR TsFile 10% Subset

This zip is a proportional public subset of the QAR TsFile data used by the
current benchmark protocol.

- Selection unit: dataset/label group.
- Selection method: chronological evenly spaced sampling.
- Target ratio: {args.ratio:.2%}.
- Selected files: {selected_total}/{original_total} ({selected_total / original_total:.4%}).
- Benchmark split: the repository loaders split each label chronologically into
  train/val/test = 70/10/20 after compact-cache construction.

Use it with the benchmark repository:

```bash
ZIP_PATH=/path/to/{args.output_zip.name} \\
IOTDB_LIB=data/iotdb-2.0.2-lib \\
DATASETS="{datasets}" \\
bash scripts/reproduce/prepare_current_caches.sh
```

Then run smoke or benchmark launchers from `scripts/reproduce/`.
"""


if __name__ == "__main__":
    main()
