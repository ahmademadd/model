"""Identical sensor validation and transformations for training and inference."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


QUATERNION_COLUMNS = tuple(f"wristMotion_quaternion{axis}" for axis in "WXYZ")


def _quaternion_indices(feature_columns: list[str]) -> list[int]:
    present = [name in feature_columns for name in QUATERNION_COLUMNS]
    if any(present) and not all(present):
        raise ValueError("Select all four quaternion columns together, or omit all four")
    return [feature_columns.index(name) for name in QUATERNION_COLUMNS] if all(present) else []


def _validate_array(values, width: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != width:
        raise ValueError(f"Expected a nonempty (timesteps, {width}) sensor array")
    if not np.isfinite(array).all():
        raise ValueError("Sensor values must all be finite")
    return array


def load_sequence(path: str | Path, feature_columns: list[str]) -> np.ndarray:
    """Read a CSV and convert WXYZ quaternions to continuous start-relative rotations.

    Uses inverse(q_start) * q_t, matching preprocessing/relative_quaternions.py.
    Applying this again to already start-relative quaternions preserves the rotations.
    Timestamps validate temporal ordering but never become model input features.
    """
    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be nonempty and unique")
    forbidden = {"activity", "activityEncoded", "weight", "set", "reps", "secondsElapsed"}
    if forbidden.intersection(columns):
        raise ValueError(f"Metadata cannot be sensor features: {sorted(forbidden.intersection(columns))}")
    indices = _quaternion_indices(columns)
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
        values = _validate_array([[float(row[column]) for column in columns] for row in rows], len(columns))
    except (ValueError, TypeError) as error:
        raise ValueError(f"{source}: invalid sensor values ({error})") from error
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{source}: secondsElapsed must be finite and strictly increasing")
    if indices:
        quaternions = values[:, indices]
        norms = np.linalg.norm(quaternions, axis=1)
        if np.any(norms <= 0) or not np.isfinite(norms).all():
            raise ValueError(f"{source}: quaternion must have a finite nonzero norm")
        quaternions = quaternions / norms[:, None]
        w, x, y, z = quaternions[0] * np.array([1, -1, -1, -1])
        v, i, j, k = quaternions.T
        relative = np.column_stack((
            w*v - x*i - y*j - z*k, w*i + x*v + y*k - z*j,
            w*j - x*k + y*v + z*i, w*k + x*j - y*i + z*v,
        ))
        relative /= np.linalg.norm(relative, axis=1, keepdims=True)
        relative[0] = [1, 0, 0, 0]
        for row in range(1, len(relative)):
            if np.dot(relative[row - 1], relative[row]) < 0:
                relative[row] *= -1
        values[:, indices] = relative
    result = values.astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{source}: sensor values exceed float32 range")
    return result


class Standardizer:
    """Fit scalar-channel means/scales on training timesteps; leave unit quaternions intact."""

    def __init__(self, feature_columns: list[str]):
        self.feature_columns = list(feature_columns)
        if not self.feature_columns or len(set(self.feature_columns)) != len(self.feature_columns):
            raise ValueError("feature_columns must be nonempty and unique")
        self.quaternion_indices = _quaternion_indices(self.feature_columns)
        self.mean = None
        self.scale = None

    def fit(self, sequences) -> "Standardizer":
        count = 0
        mean = np.zeros(len(self.feature_columns), dtype=np.float64)
        moment = np.zeros_like(mean)
        for sequence in sequences:
            values = _validate_array(sequence, len(self.feature_columns))
            n_rows = len(values)
            batch_mean = values.mean(axis=0)
            delta = batch_mean - mean
            total = count + n_rows
            moment += ((values - batch_mean) ** 2).sum(axis=0) + delta**2 * count * n_rows / total
            mean += delta * n_rows / total
            count = total
        if not count:
            raise ValueError("Cannot fit preprocessing without training samples")
        scale = np.sqrt(np.maximum(moment / count, 0))
        scale[scale < 1e-8] = 1.0
        mean[self.quaternion_indices] = 0
        scale[self.quaternion_indices] = 1
        self.mean, self.scale = mean, scale
        return self

    def transform(self, sequence) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError("Fit or load the Standardizer before transforming samples")
        values = _validate_array(sequence, len(self.feature_columns))
        result = ((values - self.mean) / self.scale).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Standardized sensor values exceed float32 range")
        return result

    def save(self, path: str | Path) -> None:
        if self.mean is None or self.scale is None:
            raise ValueError("Cannot save an unfitted Standardizer")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "version": 1, "feature_columns": self.feature_columns,
            "quaternion_mode": "start_relative_wxyz", "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("version") != 1 or state.get("quaternion_mode") != "start_relative_wxyz":
            raise ValueError("Unsupported preprocessing artifact version or quaternion mode")
        instance = cls(state["feature_columns"])
        instance.mean = np.asarray(state["mean"], dtype=np.float64)
        instance.scale = np.asarray(state["scale"], dtype=np.float64)
        shape = (len(instance.feature_columns),)
        if (instance.mean.shape != shape or instance.scale.shape != shape
                or not np.isfinite(instance.mean).all() or not np.isfinite(instance.scale).all()
                or np.any(instance.scale <= 0)):
            raise ValueError("Invalid means/scales in preprocessing artifact")
        if (np.any(instance.mean[instance.quaternion_indices] != 0)
                or np.any(instance.scale[instance.quaternion_indices] != 1)):
            raise ValueError("Quaternion channels must remain unscaled")
        return instance
