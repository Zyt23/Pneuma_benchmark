#!/usr/bin/env python3
"""Create tiny representative QAR compact caches for context60 examples.

The generated files mimic the compact-cache contract used by the real TsFile
pipeline:

    <root>/phase_segments/<anchor>/<dataset>/qar_compact_shiftN80.npz
    <root>/context60/histK/<anchor>/<dataset>/qar_compact_shiftN80.npz
    <root>/classification/<dataset>/qar_compact_shiftN80.npz

They are synthetic and intentionally small, but the metadata and shapes match
the current 80*hist+60 -> 20 protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


ANCHORS = ["predict_2_3", "predict_4_5", "predict_5_6", "predict_8_9"]
DATASET = "dataset_demo"
FEATURE_COLS = np.asarray(
    ["bleed_air_pressure", "manifold_pressure", "pack_flow", "valve_position"]
)
SEGMENT_LEN = 80
CURRENT_CONTEXT_LEN = 60
TARGET_LEN = 20


def time_keys(n_samples: int) -> np.ndarray:
    return np.asarray([20240101000000 + i * 1000000 for i in range(n_samples)], dtype=np.int64)


def sources(anchor: str, labels: np.ndarray) -> np.ndarray:
    rows = []
    for i, label in enumerate(labels):
        rows.append(f"{DATASET}/{int(label)}/2024-01-{i + 1:02d}_{anchor}.tsfile")
    return np.asarray(rows)


def make_phase_segments(anchor: str, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    anchor_idx = ANCHORS.index(anchor)
    labels = np.asarray([i % 2 for i in range(n_samples)], dtype=np.int64)
    t = np.linspace(0.0, 1.0, SEGMENT_LEN, dtype=np.float32)
    transition = 1.0 / (1.0 + np.exp(-12.0 * (t - 0.45))).astype(np.float32)
    x = np.zeros((n_samples, SEGMENT_LEN, len(FEATURE_COLS)), dtype=np.float32)

    for i, label in enumerate(labels):
        drift = 0.006 * i
        phase_shift = 0.06 * anchor_idx
        fault_ramp = np.clip((t - 0.55) / 0.45, 0.0, 1.0) * float(label)
        bleed_pressure = (
            0.62
            + 0.07 * np.sin(2.0 * np.pi * (t + phase_shift))
            + 0.13 * transition
            + drift
            + 0.08 * fault_ramp
        )
        manifold_pressure = (
            0.48
            + 0.05 * np.cos(2.0 * np.pi * (t * 0.7 + phase_shift))
            + 0.18 * t
            + 0.03 * anchor_idx
            - 0.04 * fault_ramp
        )
        pack_flow = (
            0.38
            + 0.04 * np.sin(4.0 * np.pi * t + anchor_idx)
            + 0.09 * transition
            + 0.5 * drift
            + 0.03 * fault_ramp
        )
        valve_position = (
            0.28
            + 0.16 * transition
            + 0.03 * np.sin(2.0 * np.pi * (t + 0.2))
            + 0.015 * anchor_idx
            + 0.015 * label
        )
        x[i, :, 0] = bleed_pressure
        x[i, :, 1] = manifold_pressure
        x[i, :, 2] = pack_flow
        x[i, :, 3] = valve_position

    return x, labels, sources(anchor, labels), time_keys(n_samples)


def write_compact(path: Path, x: np.ndarray, labels: np.ndarray, src: np.ndarray, keys: np.ndarray, **meta: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "x": x.astype(np.float32, copy=False),
        "mask": np.ones(x.shape[:2], dtype=np.float32),
        "labels": labels.astype(np.int64, copy=False),
        "class_names": np.asarray(["0", "1"]),
        "feature_cols": FEATURE_COLS.astype(str),
        "phase_a_shift": np.asarray([-80], dtype=np.int64),
        "sources": src.astype(str),
        "time_keys": keys.astype(np.int64, copy=False),
    }
    for key, value in meta.items():
        payload[key] = np.asarray([value], dtype=np.int64) if isinstance(value, int) else np.asarray(value)
    np.savez_compressed(path, **payload)


def build_context_cache(phase_cache: Path, out_path: Path, history_count: int) -> dict[str, int]:
    cache = np.load(phase_cache, allow_pickle=False)
    x = np.asarray(cache["x"], dtype=np.float32)
    mask = np.asarray(cache["mask"], dtype=np.float32)
    labels = np.asarray(cache["labels"], dtype=np.int64)
    src = cache["sources"].astype(str)
    keys = np.asarray(cache["time_keys"], dtype=np.int64)
    order = np.lexsort((src, keys))
    x = x[order]
    mask = mask[order]
    labels = labels[order]
    src = src[order]
    keys = keys[order]

    rows_x: list[np.ndarray] = []
    rows_mask: list[np.ndarray] = []
    rows_labels: list[int] = []
    rows_sources: list[str] = []
    rows_time_keys: list[int] = []
    rows_history_sources: list[str] = []
    for pos in range(history_count, x.shape[0]):
        hist_slice = slice(pos - history_count, pos)
        history_x = x[hist_slice, :SEGMENT_LEN, :].reshape(history_count * SEGMENT_LEN, x.shape[2])
        history_mask = mask[hist_slice, :SEGMENT_LEN].reshape(history_count * SEGMENT_LEN)
        current_x = x[pos, :CURRENT_CONTEXT_LEN, :]
        target_x = x[pos, CURRENT_CONTEXT_LEN : CURRENT_CONTEXT_LEN + TARGET_LEN, :]
        current_mask = mask[pos, :CURRENT_CONTEXT_LEN]
        target_mask = mask[pos, CURRENT_CONTEXT_LEN : CURRENT_CONTEXT_LEN + TARGET_LEN]
        rows_x.append(np.concatenate([history_x, current_x, target_x], axis=0))
        rows_mask.append(np.concatenate([history_mask, current_mask, target_mask], axis=0))
        rows_labels.append(int(labels[pos]))
        rows_sources.append(str(src[pos]))
        rows_time_keys.append(int(keys[pos]))
        rows_history_sources.append(" | ".join(src[hist_slice].tolist()))

    if not rows_x:
        raise ValueError(f"{phase_cache}: not enough samples for history_count={history_count}")

    out_x = np.stack(rows_x, axis=0).astype(np.float32)
    out_mask = np.stack(rows_mask, axis=0).astype(np.float32)
    out_labels = np.asarray(rows_labels, dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        x=out_x,
        mask=out_mask,
        labels=out_labels,
        class_names=np.asarray(["0", "1"]),
        feature_cols=FEATURE_COLS.astype(str),
        phase_a_shift=np.asarray([-80], dtype=np.int64),
        sources=np.asarray(rows_sources),
        time_keys=np.asarray(rows_time_keys, dtype=np.int64),
        history_sources=np.asarray(rows_history_sources),
        history_count=np.asarray([history_count], dtype=np.int64),
        segment_len=np.asarray([SEGMENT_LEN], dtype=np.int64),
        seq_len=np.asarray([history_count * SEGMENT_LEN + CURRENT_CONTEXT_LEN], dtype=np.int64),
        pred_len=np.asarray([TARGET_LEN], dtype=np.int64),
        current_context_len=np.asarray([CURRENT_CONTEXT_LEN], dtype=np.int64),
        target_len=np.asarray([TARGET_LEN], dtype=np.int64),
    )
    return {
        "samples": int(out_x.shape[0]),
        "seq_len": int(history_count * SEGMENT_LEN + CURRENT_CONTEXT_LEN),
        "pred_len": TARGET_LEN,
        "feature_count": int(out_x.shape[2]),
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("examples/qar_context60/data"))
    parser.add_argument("--samples", type=int, default=14)
    parser.add_argument("--history-counts", type=int, nargs="+", default=[2, 5])
    parser.add_argument("--clean", action="store_true", help="Remove the output directory before regenerating data.")
    args = parser.parse_args()

    if args.clean and args.output.exists():
        shutil.rmtree(args.output)

    manifest_rows: list[dict[str, object]] = []
    for anchor in ANCHORS:
        x, labels, src, keys = make_phase_segments(anchor, args.samples)
        phase_path = args.output / "phase_segments" / anchor / DATASET / "qar_compact_shiftN80.npz"
        write_compact(
            phase_path,
            x,
            labels,
            src,
            keys,
            segment_len=SEGMENT_LEN,
            current_context_len=CURRENT_CONTEXT_LEN,
            target_len=TARGET_LEN,
        )
        manifest_rows.append(
            {
                "kind": "phase_segment",
                "dataset": DATASET,
                "anchor": anchor,
                "history_count": "",
                "samples": int(x.shape[0]),
                "seq_len": "",
                "pred_len": "",
                "path": phase_path.as_posix(),
            }
        )

        for hist in args.history_counts:
            out_path = args.output / "context60" / f"hist{hist}" / anchor / DATASET / "qar_compact_shiftN80.npz"
            stats = build_context_cache(phase_path, out_path, hist)
            manifest_rows.append(
                {
                    "kind": "context60_forecast",
                    "dataset": DATASET,
                    "anchor": anchor,
                    "history_count": hist,
                    "samples": stats["samples"],
                    "seq_len": stats["seq_len"],
                    "pred_len": stats["pred_len"],
                    "path": out_path.as_posix(),
                }
            )

    class_x, class_labels, class_src, class_keys = make_phase_segments("predict_2_3", args.samples)
    class_path = args.output / "classification" / DATASET / "qar_compact_shiftN80.npz"
    write_compact(class_path, class_x, class_labels, class_src, class_keys, segment_len=SEGMENT_LEN)
    manifest_rows.append(
        {
            "kind": "classification",
            "dataset": DATASET,
            "anchor": "predict_2_3",
            "history_count": "",
            "samples": int(class_x.shape[0]),
            "seq_len": SEGMENT_LEN,
            "pred_len": "",
            "path": class_path.as_posix(),
        }
    )

    write_manifest(args.output / "manifest.csv", manifest_rows)
    (args.output / "manifest.json").write_text(json.dumps(manifest_rows, indent=2), encoding="utf-8")
    print(f"Wrote representative data to {args.output}")
    print(f"Rows in manifest: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
