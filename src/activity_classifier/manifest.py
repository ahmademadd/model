"""Portable CSV manifests: one row per independently segmented repetition."""

from __future__ import annotations

import csv
import os
from pathlib import Path


FIELDS = (
    "sample_id", "path", "activity", "parent_recording_id", "subject_id",
    "session_id", "n_rows", "duration_seconds", "quality_flag", "issues", "split",
)


def read_manifest(path: str | Path) -> list[dict]:
    """Resolve each sample path relative to the manifest, not the working directory."""
    source = Path(path).resolve()
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "path", "activity", "quality_flag"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source}: missing manifest columns {sorted(required - set(reader.fieldnames or []))}")
        records = list(reader)
    seen = set()
    for row in records:
        if not row["sample_id"] or row["sample_id"] in seen:
            raise ValueError(f"{source}: blank or duplicate sample_id {row['sample_id']!r}")
        seen.add(row["sample_id"])
        if not row["path"]:
            raise ValueError(f"{source}: sample {row['sample_id']} has no path")
        row["path"] = str((source.parent / row["path"]).resolve())
        if row["quality_flag"] not in {"accepted", "rejected"}:
            raise ValueError(f"{source}: unknown quality_flag for {row['sample_id']}")
    return records


def write_manifest(records: list[dict], path: str | Path) -> None:
    """Write paths relative to the destination; input paths should be absolute."""
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FIELDS)
    for record in records:
        fields.extend(key for key in record if key not in fields)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["path"] = Path(os.path.relpath(Path(row["path"]).resolve(), destination.parent)).as_posix()
            writer.writerow(row)
