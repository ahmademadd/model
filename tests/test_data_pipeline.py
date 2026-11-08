"""Behavior checks for provenance, quaternion preprocessing, and batch padding."""

import csv
from pathlib import Path

import numpy as np
import pytest

from activity_classifier.audit import build_manifest
from activity_classifier.config import DEFAULT_FEATURES
from activity_classifier.dataset import make_dataset
from activity_classifier.manifest import read_manifest, write_manifest
from activity_classifier.preprocessing import Standardizer, load_sequence
from activity_classifier.split import split_records


def write_repetition(path, labels=("curl", "curl", "curl"), times=None, quaternions=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(len(labels)) * 0.01 if times is None else times
    quaternions = [[1, 0, 0, 0]] * len(labels) if quaternions is None else quaternions
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["secondsElapsed", *DEFAULT_FEATURES, "activity"])
        writer.writeheader()
        for i, (time, activity, quaternion) in enumerate(zip(times, labels, quaternions)):
            row = dict(zip(DEFAULT_FEATURES, [float(i)] * 9 + list(quaternion)))
            writer.writerow({"secondsElapsed": time, **row, "activity": activity})
    return path


def test_audit_preserves_rejections_and_ignores_full_recordings(tmp_path):
    write_repetition(tmp_path / "session_a" / "part_001.csv")
    write_repetition(tmp_path / "session_a" / "part_002.csv", labels=("curl", "press", "curl"))
    write_repetition(tmp_path / "session_b" / "part_001.csv", times=[0, 0, 1])
    write_repetition(tmp_path / "full_recording.csv")
    write_repetition(tmp_path / "manifest.csv")
    records = build_manifest(tmp_path)
    assert len(records) == 3
    assert [row["quality_flag"] for row in records] == ["accepted", "rejected", "rejected"]
    assert records[0]["activity"] == "curl"
    assert records[0]["parent_recording_id"] == "session_a"
    assert records[0]["subject_id"] == records[0]["session_id"] == ""
    assert records[0]["duration_seconds"] == pytest.approx(0.02)
    assert "constant" in records[1]["issues"]
    assert "strictly increasing" in records[2]["issues"]


def test_manifest_paths_resolve_from_manifest_directory(tmp_path, monkeypatch):
    source = write_repetition(tmp_path / "data" / "recording" / "part_001.csv")
    records = build_manifest(tmp_path / "data")
    destination = tmp_path / "metadata" / "manifest.csv"
    write_manifest(records, destination)
    with destination.open() as handle:
        assert next(csv.DictReader(handle))["path"] == "../data/recording/part_001.csv"
    monkeypatch.chdir(tmp_path / "data")
    restored = read_manifest(destination)
    assert Path(restored[0]["path"]) == source.resolve()


def grouped_records(groups=6):
    return [dict(sample_id=f"{group}-{activity}", parent_recording_id=f"group-{group}",
                 subject_id="", activity=activity, quality_flag="accepted")
            for group in range(groups) for activity in ("curl", "press")]


def test_split_keeps_groups_together_is_repeatable_and_covers_classes():
    records = grouped_records()
    result = split_records(records)
    assert result == split_records(records)
    assert all("split" not in row for row in records)
    for group in {row["parent_recording_id"] for row in result}:
        assert len({row["split"] for row in result if row["parent_recording_id"] == group}) == 1
    for split in ("train", "validation", "test"):
        assert {row["activity"] for row in result if row["split"] == split} == {"curl", "press"}


def test_split_rejects_missing_groups_and_impossible_training_coverage():
    with pytest.raises(ValueError, match="missing subject_id"):
        split_records(grouped_records(), group_column="subject_id")
    records = [dict(sample_id=str(i), parent_recording_id=str(i), activity=f"activity-{i}", quality_flag="accepted") for i in range(3)]
    with pytest.raises(ValueError, match="every activity in training"):
        split_records(records)


def test_split_reports_missing_holdout_classes_and_keeps_rejected_rows():
    records = grouped_records(4)
    records[0]["activity"] = "rare"
    records.append(dict(sample_id="bad", quality_flag="rejected", issues="bad sensor values"))
    with pytest.warns(UserWarning, match="missing activities"):
        result = split_records(records)
    assert next(row for row in result if row["activity"] == "rare")["split"] == "train"
    assert result[-1]["split"] == ""


def test_quaternion_relative_normalization_and_sign_continuity(tmp_path):
    half = np.sqrt(0.5)
    source = write_repetition(tmp_path / "part_001.csv", labels=("curl", "curl"),
                              quaternions=[[half, 0, 0, half], [0, 0, 0, -2]])
    sequence = load_sequence(source, DEFAULT_FEATURES)
    expected = [[1, 0, 0, 0], [half, 0, 0, half]]
    np.testing.assert_allclose(sequence[:, 9:], expected, atol=1e-6)
    relative_source = write_repetition(tmp_path / "part_002.csv", labels=("curl", "curl"),
                                       quaternions=sequence[:, 9:])
    np.testing.assert_allclose(load_sequence(relative_source, DEFAULT_FEATURES), sequence, atol=1e-6)
    with pytest.raises(ValueError, match="all four"):
        load_sequence(source, DEFAULT_FEATURES[:-1])
    with pytest.raises(ValueError, match="Metadata"):
        load_sequence(source, ["activityEncoded"])


def test_preprocessing_rejects_zero_quaternions_and_nonfinite_values(tmp_path):
    path = write_repetition(tmp_path / "part_001.csv", quaternions=[[0, 0, 0, 0]] * 3)
    with pytest.raises(ValueError, match="nonzero norm"):
        load_sequence(path, DEFAULT_FEATURES)
    write_repetition(path, times=[0, np.nan, 0.02])
    with pytest.raises(ValueError, match="strictly increasing"):
        load_sequence(path, DEFAULT_FEATURES)


def test_standardizer_roundtrip_preserves_quaternions(tmp_path):
    sequence = load_sequence(write_repetition(tmp_path / "part_001.csv"), DEFAULT_FEATURES)
    standardizer = Standardizer(DEFAULT_FEATURES).fit([sequence[:1], sequence[1:]])
    transformed = standardizer.transform(sequence)
    np.testing.assert_allclose(transformed[:, :9].mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(transformed[:, :9].std(axis=0), 1, atol=1e-7)
    np.testing.assert_array_equal(transformed[:, 9:], sequence[:, 9:])
    standardizer.save(tmp_path / "preprocessing.json")
    restored = Standardizer.load(tmp_path / "preprocessing.json")
    np.testing.assert_array_equal(restored.transform(sequence), transformed)
    with pytest.raises(ValueError, match="without training"):
        Standardizer(DEFAULT_FEATURES).fit([])


def test_dataset_rightpads_and_uses_lengths_not_zero_values():
    pytest.importorskip("tensorflow")
    sequences = [np.zeros((2, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32)]
    dataset = make_dataset(sequences, [0, 1], batch_size=2)
    assert dataset.cardinality().numpy() == 1
    inputs, targets = next(iter(dataset))
    assert inputs["sensors"].shape == (2, 4, 3)
    np.testing.assert_array_equal(inputs["valid"].numpy(), [[True, True, False, False], [True] * 4])
    np.testing.assert_array_equal(targets.numpy(), [0, 1])
    np.testing.assert_array_equal(inputs["sensors"].numpy()[0, 2:], 0)


def test_dataset_rejects_empty_arrays_and_noninteger_labels():
    with pytest.raises(ValueError, match="finite, nonempty"):
        make_dataset([np.empty((0, 3))], [0])
    with pytest.raises(ValueError, match="class indices"):
        make_dataset([np.ones((2, 3))], [0.5])
