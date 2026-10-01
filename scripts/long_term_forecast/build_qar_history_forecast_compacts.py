#!/usr/bin/env python
"""Build QAR history-context forecast compact caches.

Input caches are phase-transition segment caches where each sample is one
flight/transition segment with length 80, e.g. transition-40 ... transition+39.

Default mode: for a history count K, every output sample is:

    X[previous K flights, each 80 points] + Y[current flight, 80 points]

The standard QAR forecast loader can then run with:

    seq_len = K * 80
    pred_len = 80
    label_len = 80

Current-context mode: pass ``--current_context_len 60 --target_len 20`` to use:

    X[previous K flights, each 80 points] + X[current flight, first 60 points]
        -> Y[current flight, next 20 points]

Then run with:

    seq_len = K * 80 + 60
    pred_len = 20
    label_len = 20

Earlier entries from the same dataset and anchor are used as history; they
are not necessarily from the same aircraft. This assembler preserves source
normalization, including target statistics in historical 80-point caches.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


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

DEFAULT_ANCHORS = ["predict_2_3", "predict_4_5", "predict_5_6", "predict_8_9"]


def parse_list(text: str, default: Iterable[str]) -> list[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        out = {name: data[name] for name in data.files}
    if "mask" not in out:
        out["mask"] = np.ones(out["x"].shape[:2], dtype=np.float32)
    if "labels" not in out:
        out["labels"] = np.zeros(out["x"].shape[0], dtype=np.int64)
    if "sources" not in out:
        out["sources"] = np.asarray([str(i) for i in range(out["x"].shape[0])])
    if "time_keys" not in out:
        out["time_keys"] = np.arange(out["x"].shape[0], dtype=np.int64)
    return out


def build_one(
    cache_path: Path,
    out_path: Path,
    history_count: int,
    segment_len: int,
    current_context_len: int = 0,
    target_len: int = 0,
    normal_only: bool = False,
) -> dict:
    cache = load_cache(cache_path)
    x = np.asarray(cache["x"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=np.float32)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    sources = np.asarray(cache["sources"]).astype(str)
    time_keys = np.asarray(cache["time_keys"], dtype=np.int64)
    feature_cols = np.asarray(cache.get("feature_cols", np.array([f"var_{i}" for i in range(x.shape[2])]))).astype(str)

    if x.ndim != 3:
        raise ValueError(f"{cache_path}: expected x shape (N,T,C), got {x.shape}")
    if history_count < 1 or segment_len < 1:
        raise ValueError('history_count and segment_len must be positive')
    if normal_only:
        keep = labels == 0
        x, mask, labels, sources, time_keys = (a[keep] for a in (x, mask, labels, sources, time_keys))
    if current_context_len or target_len:
        if current_context_len <= 0 or target_len <= 0:
            raise ValueError("current-context mode requires both current_context_len and target_len > 0")
        required_segment_len = current_context_len + target_len
        if required_segment_len != segment_len:
            raise ValueError('current_context_len + target_len must equal segment_len')
    else:
        required_segment_len = segment_len

    if x.shape[1] < required_segment_len:
        raise ValueError(f"{cache_path}: segment length {x.shape[1]} < requested {required_segment_len}")

    order = np.lexsort((sources, time_keys))
    x = x[order, :required_segment_len, :]
    mask = mask[order, :required_segment_len]
    labels = labels[order]
    sources = sources[order]
    time_keys = time_keys[order]

    rows_x = []
    rows_mask = []
    rows_labels = []
    rows_sources = []
    rows_time_keys = []
    rows_history_sources = []
    for pos in range(history_count, x.shape[0]):
        hist_slice = slice(pos - history_count, pos)
        history_x = x[hist_slice, :segment_len, :].reshape(history_count * segment_len, x.shape[2])
        history_mask = mask[hist_slice, :segment_len].reshape(history_count * segment_len)
        if current_context_len and target_len:
            current_context_x = x[pos, :current_context_len, :]
            target_x = x[pos, current_context_len:current_context_len + target_len, :]
            current_context_mask = mask[pos, :current_context_len]
            target_mask = mask[pos, current_context_len:current_context_len + target_len]
            context_x = np.concatenate([history_x, current_context_x], axis=0)
            context_mask = np.concatenate([history_mask, current_context_mask], axis=0)
        else:
            context_x = history_x
            target_x = x[pos, :segment_len, :]
            context_mask = history_mask
            target_mask = mask[pos, :segment_len]
        rows_x.append(np.concatenate([context_x, target_x], axis=0))
        rows_mask.append(np.concatenate([context_mask, target_mask], axis=0))
        rows_labels.append(labels[pos])
        rows_sources.append(sources[pos])
        rows_time_keys.append(time_keys[pos])
        rows_history_sources.append(" | ".join(sources[hist_slice].tolist()))

    if not rows_x:
        raise ValueError(f"{cache_path}: no samples after requiring {history_count} history segments")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_x = np.stack(rows_x, axis=0).astype(np.float32)
    out_mask = np.stack(rows_mask, axis=0).astype(np.float32)
    out_labels = np.asarray(rows_labels, dtype=np.int64)
    out_sources = np.asarray(rows_sources)
    out_time_keys = np.asarray(rows_time_keys, dtype=np.int64)
    out_history_sources = np.asarray(rows_history_sources)

    descriptor, temporary_name = tempfile.mkstemp(prefix=out_path.stem + '.', suffix='.npz', dir=out_path.parent)
    os.close(descriptor)
    # Complete a temporary ZIP before replacing any existing cache.
    np.savez_compressed(
        temporary_name,
        x=out_x,
        mask=out_mask,
        labels=out_labels,
        class_names=np.asarray(["0", "1"]),
        feature_cols=feature_cols,
        phase_a_shift=np.asarray([-80], dtype=np.int64),
        sources=out_sources,
        time_keys=out_time_keys,
        history_sources=out_history_sources,
        history_count=np.asarray([history_count], dtype=np.int64),
        segment_len=np.asarray([segment_len], dtype=np.int64),
        seq_len=np.asarray([history_count * segment_len + current_context_len], dtype=np.int64),
        pred_len=np.asarray([target_len or segment_len], dtype=np.int64),
        current_context_len=np.asarray([current_context_len], dtype=np.int64),
        target_len=np.asarray([target_len or segment_len], dtype=np.int64),
        normal_only=np.asarray([normal_only], dtype=np.bool_),
    )
    os.replace(temporary_name, out_path)

    return {
        "samples": int(out_labels.shape[0]),
        "class0": int((out_labels == 0).sum()),
        "class1": int((out_labels == 1).sum()),
        "seq_len": int(history_count * segment_len + current_context_len),
        "pred_len": int(target_len or segment_len),
        "current_context_len": int(current_context_len),
        "total_len": int(out_x.shape[1]),
        "feature_count": int(out_x.shape[2]),
        "first_time_key": int(out_time_keys.min()),
        "last_time_key": int(out_time_keys.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--datasets", default=" ".join(DEFAULT_DATASETS))
    parser.add_argument("--anchors", default=" ".join(DEFAULT_ANCHORS))
    parser.add_argument("--history_counts", nargs="+", type=int, default=[2, 3, 5, 8, 12])
    parser.add_argument("--segment_len", type=int, default=80)
    parser.add_argument("--current_context_len", type=int, default=0)
    parser.add_argument("--target_len", type=int, default=0)
    parser.add_argument('--normal_only', action='store_true', help='filter both historical and target flights before assembly')
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DEFAULT_DATASETS)
    anchors = parse_list(args.anchors, DEFAULT_ANCHORS)
    manifest_rows = []
    for history_count in args.history_counts:
        for anchor in anchors:
            for dataset in datasets:
                cache_path = args.source_root / anchor / dataset / "qar_compact_shiftN80.npz"
                out_path = args.output_root / f"hist{history_count}" / anchor / dataset / "qar_compact_shiftN80.npz"
                print(f"build hist{history_count} {anchor} {dataset}", flush=True)
                stats = build_one(
                    cache_path,
                    out_path,
                    history_count,
                    args.segment_len,
                    current_context_len=args.current_context_len,
                    target_len=args.target_len,
                    normal_only=args.normal_only,
                )
                row = {
                    "history_count": history_count,
                    "anchor": anchor,
                    "dataset": dataset,
                    "cache_file": str(out_path),
                    **stats,
                }
                manifest_rows.append(row)

    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "history_forecast_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    (args.output_root / "history_forecast_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(args.output_root)


if __name__ == "__main__":
    main()
