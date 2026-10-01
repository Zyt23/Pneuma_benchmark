#!/usr/bin/env python3
"""Build nested event-local W30/W50/W80/W100 caches.

Dataset14 has an explicit ``eventNN`` identifier in every positive filename.
Dataset7 does *not*: its tail and timestamp can support a reproducibility-only
gap heuristic, but a time gap is not proof of a maintenance-event boundary.
Strict dataset7 builds therefore require a source-to-event manifest.  The old
30-day inference can only be enabled explicitly and is labelled non-strict in
the generated metadata.

Event rank is computed on the raw conversion manifest before compact-cache
filtering; a compact-cache positive is retained iff its raw rank to the event
is <= W.  All normal samples and their original order are unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


DATASETS = ("dataset7", "dataset14")
WINDOWS = (30, 50, 80, 100)
TAIL_RE = re.compile(r"(?<![A-Z0-9])(B-[A-Z0-9]+)")
EVENT_RE = re.compile(r"(?:^|/)event(\d+)_", re.IGNORECASE)
DATETIME_RE = re.compile(
    r"(20\d{2})[-_](\d{2})[-_](\d{2})(?:[_-](\d{2})[-_](\d{2})[-_](\d{2}))?"
)


def basename(value: str) -> str:
    return os.path.basename(value.replace("\\", "/"))


def parse_datetime(value: str) -> datetime:
    match = DATETIME_RE.search(basename(value))
    if not match:
        raise ValueError(f"timestamp not found in {value!r}")
    parts = [int(part) if part is not None else 0 for part in match.groups()]
    return datetime(*parts)


def read_positive_manifest(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [
            row["source"]
            for row in rows
            if row.get("label") == "1" and row.get("status") == "OK"
        ]


def read_event_manifest(path: Path) -> dict[str, str]:
    """Read an explicit ``source,event_id`` mapping for dataset7 positives."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        required = {"source", "event_id"}
        if not rows.fieldnames or not required.issubset(rows.fieldnames):
            raise ValueError(
                f"event manifest must contain {sorted(required)}; "
                f"got {rows.fieldnames}"
            )
        mapping: dict[str, str] = {}
        for row in rows:
            key = source_key(row["source"])
            event_id = row["event_id"].strip()
            if not key or not event_id:
                raise ValueError(f"empty source/event_id in {path}: {row}")
            if key in mapping and mapping[key] != event_id:
                raise ValueError(f"conflicting event ids for {key}")
            mapping[key] = event_id
    return mapping


def event_groups(
    dataset: str,
    sources: list[str],
    dataset7_event_by_source: dict[str, str] | None = None,
    allow_dataset7_gap_inference: bool = False,
) -> tuple[dict[str, list[str]], str, bool]:
    grouped: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
    if dataset == "dataset14":
        for source in sources:
            match = EVENT_RE.search(source.replace("\\", "/"))
            if not match:
                raise ValueError(f"dataset14 source lacks event id: {source}")
            grouped[f"event{int(match.group(1)):02d}"].append((parse_datetime(source), source))
        boundary_method = "explicit eventNN in released positive filename"
        semantically_strict = True
    elif dataset == "dataset7":
        if dataset7_event_by_source is not None:
            missing = [
                source for source in sources
                if source_key(source) not in dataset7_event_by_source
            ]
            if missing:
                raise ValueError(
                    "dataset7 explicit event manifest is incomplete; "
                    f"missing={len(missing)}, examples={missing[:5]}"
                )
            for source in sources:
                event = dataset7_event_by_source[source_key(source)]
                grouped[event].append((parse_datetime(source), source))
            boundary_method = "explicit source-to-maintenance-event manifest"
            semantically_strict = True
        elif allow_dataset7_gap_inference:
            by_tail: dict[str, list[tuple[datetime, str]]] = defaultdict(list)
            for source in sources:
                match = TAIL_RE.search(basename(source))
                if not match:
                    raise ValueError(f"dataset7 source lacks tail: {source}")
                by_tail[match.group(1)].append((parse_datetime(source), source))
            for tail, values in by_tail.items():
                values.sort()
                block = 1
                previous: datetime | None = None
                for when, source in values:
                    if previous is not None and (when - previous).total_seconds() > 30 * 86400:
                        block += 1
                    grouped[f"{tail}_gapblock{block:02d}"].append((when, source))
                    previous = when
            boundary_method = "30-day inter-flight gap inference (reproduction only)"
            semantically_strict = False
        else:
            raise ValueError(
                "dataset7 has no explicit event id. Supply "
                "--dataset7-event-manifest for a strict build, or explicitly "
                "use --allow-dataset7-gap-inference for non-strict reproduction."
            )
    else:
        raise ValueError(dataset)
    groups = {
        event: [source for _, source in sorted(values)]
        for event, values in sorted(grouped.items())
    }
    return groups, boundary_method, semantically_strict


def source_key(value: str) -> str:
    return basename(value).lower()


def digest_sources(sources: np.ndarray, labels: np.ndarray) -> str:
    sha = hashlib.sha256()
    for source, label in zip(sources.astype(str), labels.astype(int)):
        sha.update(str(label).encode("ascii"))
        sha.update(b"\t")
        sha.update(source.encode("utf-8"))
        sha.update(b"\n")
    return sha.hexdigest()


def build_dataset(
    dataset: str,
    cache_path: Path,
    stats_path: Path,
    output_root: Path,
    dataset7_event_by_source: dict[str, str] | None = None,
    allow_dataset7_gap_inference: bool = False,
) -> list[dict[str, object]]:
    raw_sources = read_positive_manifest(stats_path)
    groups, boundary_method, semantically_strict = event_groups(
        dataset,
        raw_sources,
        dataset7_event_by_source=dataset7_event_by_source,
        allow_dataset7_gap_inference=allow_dataset7_gap_inference,
    )

    rank_by_key: dict[str, tuple[str, int, int]] = {}
    duplicate_keys: list[str] = []
    for event, values in groups.items():
        size = len(values)
        for position, source in enumerate(values):
            key = source_key(source)
            if key in rank_by_key:
                duplicate_keys.append(key)
            rank_by_key[key] = (event, size - position, size)
    if duplicate_keys:
        raise ValueError(f"duplicate raw positive source keys: {duplicate_keys[:5]}")

    with np.load(cache_path, allow_pickle=False) as loaded:
        cache = {key: loaded[key] for key in loaded.files}
    labels = cache["labels"].astype(np.int64, copy=False)
    sources = cache["sources"].astype(str)
    positive = np.flatnonzero(labels == 1)

    event_ids = np.full(labels.shape[0], "", dtype="U64")
    ranks = np.full(labels.shape[0], -1, dtype=np.int64)
    raw_event_sizes = np.zeros(labels.shape[0], dtype=np.int64)
    missing: list[str] = []
    for index in positive:
        item = rank_by_key.get(source_key(sources[index]))
        if item is None:
            missing.append(sources[index])
            continue
        event_ids[index], ranks[index], raw_event_sizes[index] = item
    if missing:
        raise ValueError(
            f"{dataset}: {len(missing)} compact positives absent from raw manifest; "
            f"examples={missing[:5]}"
        )

    rows: list[dict[str, object]] = []
    original_digest = digest_sources(sources, labels)
    for window in WINDOWS:
        keep = (labels == 0) | ((labels == 1) & (ranks <= window))
        indices = np.flatnonzero(keep)
        out = {key: value[indices] if value.ndim and value.shape[0] == labels.shape[0] else value
               for key, value in cache.items()}
        out["event_ids"] = event_ids[indices]
        out["flights_to_event_rank"] = ranks[indices]
        out["raw_event_sizes"] = raw_event_sizes[indices]
        out["annotation_window"] = np.array([window], dtype=np.int64)
        out["annotation_method"] = np.array([
            "event-local last-W raw flights; normals unchanged; raw rank precedes compact filtering"
        ])
        out["event_boundary_method"] = np.array([boundary_method])
        out["semantically_strict_event_boundary"] = np.array(
            [semantically_strict], dtype=np.bool_
        )

        destination = output_root / f"W{window}" / dataset
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination / "qar_compact_shiftN80.npz", **out)

        kept_labels = out["labels"].astype(int)
        row = {
            "dataset": dataset,
            "window": window,
            "samples": int(kept_labels.size),
            "normal": int((kept_labels == 0).sum()),
            "positive": int((kept_labels == 1).sum()),
            "events": len(groups),
            "short_events": int(sum(len(values) < window for values in groups.values())),
            "min_raw_event_size": int(min(map(len, groups.values()))),
            "max_raw_event_size": int(max(map(len, groups.values()))),
            "event_boundary_method": boundary_method,
            "semantically_strict_event_boundary": semantically_strict,
            "source_label_sha256": digest_sources(out["sources"], out["labels"]),
            "w100_matches_original_source_order": bool(
                window != 100 or digest_sources(out["sources"], out["labels"]) == original_digest
            ),
            "cache": str(destination / "qar_compact_shiftN80.npz"),
        }
        rows.append(row)
        (destination / "window_metadata.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-root", type=Path, required=True)
    parser.add_argument("--raw-stats-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset7-event-manifest",
        type=Path,
        help="CSV with source,event_id columns from the maintenance-event table",
    )
    parser.add_argument(
        "--allow-dataset7-gap-inference",
        action="store_true",
        help="Reproduce the old 30-day grouping; generated dataset7 metadata is non-strict",
    )
    args = parser.parse_args()

    dataset7_event_by_source = (
        read_event_manifest(args.dataset7_event_manifest)
        if args.dataset7_event_manifest else None
    )

    all_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        all_rows.extend(build_dataset(
            dataset,
            args.compact_root / dataset / "qar_compact_shiftN80.npz",
            args.raw_stats_root / dataset / "tsfile_conversion_stats.tsv",
            args.output_root,
            dataset7_event_by_source=dataset7_event_by_source,
            allow_dataset7_gap_inference=args.allow_dataset7_gap_inference,
        ))
    args.output_root.mkdir(parents=True, exist_ok=True)
    fields = list(all_rows[0])
    with (args.output_root / "window_cache_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == "__main__":
    main()
