"""Explicit quaternion preprocessing and saved scalar scaling for model inputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


QUATERNION_COLUMNS = tuple(f"wristMotion_quaternion{axis}" for axis in "WXYZ")


def _quaternion_indices(feature_columns: list[str]) -> list[int]:
    present = [name in feature_columns for name in QUATERNION_COLUMNS]
    if any(present) and not all(present):
        raise ValueError("Select all four quaternion columns together, or omit all four")
    return [feature_columns.index(name) for name in QUATERNION_COLUMNS] if all(present) else []


def _quaternion_norms(values: np.ndarray, indices: list[int]) -> np.ndarray:
    """Return sqrt(w² + x² + y² + z²) per row, rejecting invalid magnitudes."""

    norms = np.linalg.norm(values[:, indices], axis=1)
    if np.any(norms <= 0) or not np.isfinite(norms).all():
        raise ValueError("quaternion must have a finite nonzero norm")
    return norms


def preprocess_quaternions(sequence: np.ndarray, feature_columns: list[str]) -> np.ndarray:
    """Produce normalized, sign-continuous, start-relative WXYZ rotations.
    Uses inverse(q_start) * q_t, matching preprocessing/relative_quaternions.py.
    Applying this again to already start-relative quaternions preserves the rotations.
    """

    columns = list(feature_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("feature_columns must be nonempty and unique")
    indices = _quaternion_indices(columns)
    values = np.asarray(sequence, dtype=np.float64)
    if indices:
        quaternions = values[:, indices]
        norms = _quaternion_norms(values, indices)
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
        raise ValueError("Preprocessed sensor values exceed float32 range")
    return result


class Standardizer:
    """Fit scalar-channel means/scales on training timesteps; leave unit quaternions intact."""

    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, sequences, feature_columns: list[str]) -> "Standardizer":
        columns = list(feature_columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("feature_columns must be nonempty and unique")
        quaternion_indices = _quaternion_indices(columns)
        count = 0
        mean = np.zeros(len(columns), dtype=np.float64)
        moment = np.zeros_like(mean)
        for sequence in sequences:
            values = np.asarray(sequence, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != len(columns) or not len(values):
                raise ValueError("Each sequence must be a nonempty matrix matching feature_columns")
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
        mean[quaternion_indices] = 0
        scale[quaternion_indices] = 1
        self.mean, self.scale = mean, scale
        return self

    def transform(self, sequence) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise ValueError("Fit or load the Standardizer before transforming samples")
        values = np.asarray(sequence, dtype=np.float64)
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
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
        }, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Standardizer":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        instance = cls()
        instance.mean = np.asarray(state["mean"], dtype=np.float64)
        instance.scale = np.asarray(state["scale"], dtype=np.float64)
        if (instance.mean.ndim != 1 or instance.mean.size == 0
                or instance.scale.shape != instance.mean.shape
                or not np.isfinite(instance.mean).all() or not np.isfinite(instance.scale).all()
                or np.any(instance.scale <= 0)):
            raise ValueError("Invalid means/scales in preprocessing artifact")
        return instance
