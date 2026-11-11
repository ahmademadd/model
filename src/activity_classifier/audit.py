"""Catalog repetition CSVs and retain rejected samples with actionable reasons."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

import numpy as np

from .config import DEFAULT_FEATURES, load_config
from .manifest import FIELDS, write_manifest
from .sequence import load_sequence


def build_manifest(input_dir: str | Path, pattern: str = "part_*.csv",
                   feature_columns: list[str] | None = None) -> list[dict]:
    root = Path(input_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"Input directory does not exist: {root}")
    sources = sorted(path for path in root.rglob(pattern) if path.is_file() and path.name != "manifest.csv")
    if not sources:
        raise ValueError(f"No repetition files matching {pattern!r} under {root}")
    features = DEFAULT_FEATURES if feature_columns is None else feature_columns
    records = []
    for source in sources:
        relative = source.relative_to(root)
        record = dict.fromkeys(FIELDS, "")
        record.update(sample_id=relative.with_suffix("").as_posix(), path=str(source),
                      parent_recording_id=relative.parent.as_posix() if relative.parent != Path(".") else root.name,
                      quality_flag="accepted", n_rows=0)
        issues = []
        try:
            with source.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            record["n_rows"] = len(rows)
            activities = {str(row.get("activity") or "").strip() for row in rows}
            if not rows:
                issues.append("no sensor rows")
            elif "" in activities:
                issues.append("activity is missing on one or more rows")
            elif len(activities) != 1:
                issues.append(f"activity must be constant throughout a repetition: {sorted(activities)}")
            else:
                record["activity"] = activities.pop()
            load_sequence(source, features)
            times = np.asarray([float(row["secondsElapsed"]) for row in rows])
            record["duration_seconds"] = float(times[-1] - times[0])
        except (ValueError, TypeError, KeyError, OSError, csv.Error) as error:
            issues.append(str(error))
        if issues:
            record["quality_flag"] = "rejected"
            record["issues"] = "; ".join(issues)
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="part_*.csv")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        records = build_manifest(args.input, args.pattern, config["feature_columns"])
        if args.output.resolve() in {Path(row["path"]) for row in records}:
            raise ValueError("Output must not overwrite a source repetition")
        write_manifest(records, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    accepted = [row for row in records if row["quality_flag"] == "accepted"]
    print(f"Audited {len(records)} repetitions: {len(accepted)} accepted, {len(records) - len(accepted)} rejected")
    for activity, count in sorted(Counter(row["activity"] for row in accepted).items()):
        groups = {row["parent_recording_id"] for row in accepted if row["activity"] == activity}
        print(f"  {activity}: {count} repetitions in {len(groups)} parent recordings")
    print(f"Manifest: {args.output.resolve()}")
    print("Subject/session IDs are blank; fill verified provenance before splitting by those fields.")


if __name__ == "__main__":
    main()
