"""Core contracts for auditing, preprocessing, and grouped splits."""

import csv

import numpy as np
import pytest

from activity_classifier.audit import build_manifest
from activity_classifier.config import DEFAULT_FEATURES
from activity_classifier.preprocessing import Standardizer, preprocess_quaternions
from activity_classifier.sequence import load_sequence
from activity_classifier.split import split_records
from activity_classifier.train import validate_splits


def write_repetition(path, labels=("curl", "curl", "curl"), quaternions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    quaternions = [[1, 0, 0, 0]] * len(labels) if quaternions is None else quaternions
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["secondsElapsed", *DEFAULT_FEATURES, "activity"])
        for index, (activity, quaternion) in enumerate(zip(labels, quaternions)):
            writer.writerow([index * 0.01, *([float(index)] * 9), *quaternion, activity])
    return path


def test_audit_validates_recorded_data_without_preprocessing(tmp_path, monkeypatch):
    import activity_classifier.preprocessing as preprocessing

    def unexpected_preprocessing(*args, **kwargs):
        raise AssertionError("Auditing must not preprocess quaternions")

    monkeypatch.setattr(preprocessing, "preprocess_quaternions", unexpected_preprocessing)
    quaternions = [[2, 0, 0, 2], [-2, 0, 0, -2], [0, 0, 0, 3]]
    source = write_repetition(tmp_path / "recording" / "part_001.csv", quaternions=quaternions)
    original = source.read_bytes()
    write_repetition(tmp_path / "recording" / "part_002.csv", labels=("curl", "press", "curl"))
    write_repetition(tmp_path / "full_recording.csv")
    records = build_manifest(tmp_path)
    assert [row["quality_flag"] for row in records] == ["accepted", "rejected"]
    assert "constant" in records[1]["issues"]
    np.testing.assert_array_equal(load_sequence(source, DEFAULT_FEATURES)[:, 9:], quaternions)
    assert source.read_bytes() == original


def test_quaternion_preprocessing_and_saved_scaling(tmp_path):
    half = np.sqrt(0.5)
    source = write_repetition(tmp_path / "part_001.csv", labels=("curl", "curl"),
                              quaternions=[[half, 0, 0, half], [0, 0, 0, -2]])
    sequence = preprocess_quaternions(load_sequence(source, DEFAULT_FEATURES), DEFAULT_FEATURES)
    np.testing.assert_allclose(sequence[:, 9:], [[1, 0, 0, 0], [half, 0, 0, half]], atol=1e-6)
    np.testing.assert_array_equal(sequence[:, :9], [[0] * 9, [1] * 9])
    standardizer = Standardizer(DEFAULT_FEATURES).fit([sequence])
    transformed = standardizer.transform(sequence)
    np.testing.assert_allclose(transformed[:, :9], [[-1] * 9, [1] * 9])
    np.testing.assert_array_equal(transformed[:, 9:], sequence[:, 9:])
    state = tmp_path / "preprocessing.json"
    standardizer.save(state)
    np.testing.assert_array_equal(Standardizer.load(state).transform(sequence), transformed)


def test_grouped_splits_preserve_classes_and_reject_leakage():
    records = [dict(sample_id=f"{group}-{activity}", path=f"/data/{group}-{activity}.csv",
                    parent_recording_id=f"group-{group}", activity=activity,
                    quality_flag="accepted")
               for group in range(6) for activity in ("curl", "press")]
    result = split_records(records)
    assert result == split_records(records)
    for group in {row["parent_recording_id"] for row in result}:
        assert len({row["split"] for row in result if row["parent_recording_id"] == group}) == 1
    partitions = validate_splits(result, "parent_recording_id")
    for rows in partitions.values():
        assert {row["activity"] for row in rows} == {"curl", "press"}
    partitions["validation"][0]["parent_recording_id"] = partitions["train"][0]["parent_recording_id"]
    with pytest.raises(ValueError, match="Leakage"):
        validate_splits(result, "parent_recording_id")
