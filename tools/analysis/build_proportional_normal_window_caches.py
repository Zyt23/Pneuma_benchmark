#!/usr/bin/env python3
"""Build A320-PRV window caches with proportional normal subsampling.

The original W30/W50/W80/W100 sensitivity caches retain every normal flight
while retaining only the last W positive flights before each inferred event.
That changes the class prevalence as W changes.  This controlled variant keeps
30/50/80/100 percent of the normal class, respectively, while leaving the
positive examples exactly as they are in the original window caches.

Normal examples are selected deterministically (seed 2021) and separately
inside the original chronological train/validation/test segments.  This keeps
all three temporal regions represented and makes the smaller normal sets nested
subsets of the larger ones.  W100 is copied byte-for-byte as the anchor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


WINDOWS = (30, 50, 80, 100)
FRACTIONS = {30: 0.30, 50: 0.50, 80: 0.80, 100: 1.00}


def split_bounds(n: int) -> tuple[int, int]:
    """Match the historical Table-4 per-class 70/10/20 split."""
    n = int(n)
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def priority(source: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}\t{source}".encode("utf-8")).digest()


def choose_nested(indices: np.ndarray, sources: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count < 0 or count > len(indices):
        raise ValueError((count, len(indices)))
    ranked = sorted(indices.tolist(), key=lambda idx: (priority(str(sources[idx]), seed), int(idx)))
    return np.asarray(sorted(ranked[:count]), dtype=np.int64)


def selected_normals(
    labels: np.ndarray,
    time_keys: np.ndarray,
    sources: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    normal = np.flatnonzero(labels == 0)
    ordered = normal[np.argsort(time_keys[normal], kind="mergesort")]
    old_train_end, old_val_end = split_bounds(len(ordered))
    old_parts = (
        ordered[:old_train_end],
        ordered[old_train_end:old_val_end],
        ordered[old_val_end:],
    )

    target_total = int(math.floor(len(ordered) * fraction))
    new_train_end, new_val_end = split_bounds(target_total)
    target_counts = (
        new_train_end,
        new_val_end - new_train_end,
        target_total - new_val_end,
    )
    chosen_parts = [
        choose_nested(part, sources, wanted, seed)
        for part, wanted in zip(old_parts, target_counts)
    ]
    selected = np.sort(np.concatenate(chosen_parts))
    return selected, {
        "normal_total": int(target_total),
        "normal_train": int(target_counts[0]),
        "normal_val": int(target_counts[1]),
        "normal_test": int(target_counts[2]),
    }


def class_split_counts(labels: np.ndarray, time_keys: np.ndarray) -> dict[str, int]:
    result: dict[str, int] = {}
    for label, prefix in ((0, "normal"), (1, "positive")):
        group = np.flatnonzero(labels == label)
        group = group[np.argsort(time_keys[group], kind="mergesort")]
        train_end, val_end = split_bounds(len(group))
        result[f"{prefix}_total"] = int(len(group))
        result[f"{prefix}_train"] = int(train_end)
        result[f"{prefix}_val"] = int(val_end - train_end)
        result[f"{prefix}_test"] = int(len(group) - val_end)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", default="dataset7")
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    selected_by_window: dict[int, set[str]] = {}
    for window in WINDOWS:
        source = args.source_root / f"W{window}" / args.dataset / "qar_compact_shiftN80.npz"
        destination = args.output_root / f"W{window}" / args.dataset / "qar_compact_shiftN80.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)

        if window == 100:
            shutil.copy2(source, destination)
        else:
            with np.load(source, allow_pickle=False) as payload:
                cache = {key: payload[key] for key in payload.files}
            labels = cache["labels"].astype(np.int64, copy=False)
            sources = cache["sources"].astype(str)
            time_keys = cache.get("time_keys", np.arange(len(labels), dtype=np.int64))
            normal_selected, _ = selected_normals(
                labels, time_keys, sources, FRACTIONS[window], args.seed
            )
            positive = np.flatnonzero(labels == 1)
            keep = np.sort(np.concatenate([normal_selected, positive]))
            filtered = {
                key: (value[keep] if value.ndim and value.shape[0] == len(labels) else value)
                for key, value in cache.items()
            }
            filtered["normal_retention_fraction"] = np.asarray([FRACTIONS[window]], dtype=np.float64)
            filtered["normal_subsample_seed"] = np.asarray([args.seed], dtype=np.int64)
            filtered["normal_subsample_method"] = np.asarray([
                "nested SHA256 priority within original per-class chronological train/val/test segments"
            ])
            np.savez_compressed(destination, **filtered)

        with np.load(destination, allow_pickle=False) as payload:
            labels = payload["labels"].astype(np.int64, copy=False)
            sources = payload["sources"].astype(str)
            time_keys = payload.get("time_keys", np.arange(len(labels), dtype=np.int64))
        counts = class_split_counts(labels, time_keys)
        selected_by_window[window] = set(sources[labels == 0].tolist())
        row: dict[str, object] = {
            "dataset": args.dataset,
            "window": window,
            "normal_retention_fraction": FRACTIONS[window],
            "subsample_seed": args.seed,
            "samples": int(len(labels)),
            **counts,
            "overall_positive_fraction": float(np.mean(labels == 1)),
            "test_positive_fraction": counts["positive_test"]
            / (counts["positive_test"] + counts["normal_test"]),
            "source_cache": str(source),
            "output_cache": str(destination),
            "source_sha256": file_sha256(source),
            "output_sha256": file_sha256(destination),
            "w100_byte_identical": window != 100 or file_sha256(source) == file_sha256(destination),
        }
        rows.append(row)
        (destination.parent / "proportional_normal_metadata.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    for smaller, larger in zip(WINDOWS, WINDOWS[1:]):
        if not selected_by_window[smaller].issubset(selected_by_window[larger]):
            raise RuntimeError(f"normal sets are not nested: W{smaller} !subset W{larger}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = args.output_root / "proportional_normal_cache_manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_root / "README.md").write_text(
        "# Proportional-normal A320-PRV window caches\n\n"
        "Positive examples are unchanged from the original W-specific caches. "
        "Normal examples retain W/100 of the W100 normal class, deterministically "
        "within each original chronological split. W100 is byte-identical to the "
        "main-table cache.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
