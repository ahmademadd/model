"""Read and validate recorded sensor sequences without preprocessing them."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .preprocessing import _quaternion_indices, _quaternion_norms


def load_sequence(path: str | Path, feature_columns: list[str]) -> np.ndarray:
    """Read recorded sensor values as float64 and validate them.
    Checks columns, finite values, timestamp order, and nonzero quaternion norms.
    """

    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be nonempty and unique")
    forbidden = {"activity", "activityEncoded", "weight", "set", "reps", "secondsElapsed"}
    if forbidden.intersection(columns):
        raise ValueError(f"Metadata cannot be sensor features: {sorted(forbidden.intersection(columns))}")
    source = Path(path)
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns + ["secondsElapsed"]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: no sensor rows")
    try:
        times = np.asarray([float(row["secondsElapsed"]) for row in rows], dtype=np.float64)
        values = np.asarray([[float(row[column]) for column in columns] for row in rows], dtype=np.float64)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{source}: invalid sensor values ({error})") from error
    if not np.isfinite(values).all():
        raise ValueError(f"{source}: sensor values must all be finite")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{source}: secondsElapsed must be finite and strictly increasing")
    try:
        indices = _quaternion_indices(columns)
        _quaternion_norms(values, indices)
    except ValueError as error:
        raise ValueError(f"{source}: {error}") from error
    if np.any(np.abs(values) > np.finfo(np.float32).max):
        raise ValueError(f"{source}: sensor values exceed float32 range")
    return values
