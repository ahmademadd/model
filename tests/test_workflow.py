"""Exercise a whole saved run using synthetic recordings, not project sensor data."""

import csv
import json

import numpy as np
import pytest

from activity_classifier.audit import build_manifest
from activity_classifier.config import DEFAULT_FEATURES, load_config
from activity_classifier.evaluate import classification_metrics, evaluate_run
from activity_classifier.manifest import write_manifest
from activity_classifier.predict import predict_file
from activity_classifier.train import train, validate_splits


def write_repetition(path, label, length, seed):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(length, len(DEFAULT_FEATURES)))
    values[:, -4:] = [1, 0, 0, 0]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["secondsElapsed", *DEFAULT_FEATURES, "activity"])
        for index, row in enumerate(values):
            writer.writerow([index / 100, *row, label])


def test_complete_training_evaluation_and_inference(tmp_path):
    for partition in ("train", "validation", "test"):
        for index, label in enumerate(("curl", "press")):
            write_repetition(tmp_path / "data" / f"{partition}-{label}" / "part_001.csv",
                             label, 12 + index * 3, index)
    records = build_manifest(tmp_path / "data")
    for row in records:
        row["split"] = row["parent_recording_id"].split("-")[0]
    manifest = tmp_path / "manifest.csv"
    write_manifest(records, manifest)
    config = load_config()
    config["model"].update(filters=4, kernel_size=3, dilations=[1, 2], dense_units=8,
                           dropout=0.0, head_dropout=0.0)
    config["training"].update(epochs=1, batch_size=2, patience=1)
    run_dir = tmp_path / "run"
    validation = train(manifest, run_dir, config)
    assert validation["n_samples"] == 2
    assert (run_dir / "model.keras").is_file()
    assert not (run_dir / "test_metrics.json").exists()
    metrics = evaluate_run(run_dir)
    assert metrics["n_samples"] == 2
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)
    prediction = predict_file(run_dir, tmp_path / "data" / "test-curl" / "part_001.csv")
    assert prediction["activity"] in ("curl", "press")
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
    assert json.loads((run_dir / "class_mapping.json").read_text())["classes"] == ["curl", "press"]


def test_split_validation_rejects_leakage_and_unknown_activity():
    records = [dict(sample_id=str(i), path=f"/data/{i}.csv", parent_recording_id=f"group-{i}",
                    quality_flag="accepted", activity=label, split=split)
               for i, (label, split) in enumerate([
                   ("curl", "train"), ("press", "train"), ("curl", "validation")])]
    records[2]["parent_recording_id"] = "group-0"
    with pytest.raises(ValueError, match="Leakage"):
        validate_splits(records, "parent_recording_id")
    records[2]["parent_recording_id"] = "group-2"
    records[2]["activity"] = "unseen"
    with pytest.raises(ValueError, match="absent from training"):
        validate_splits(records, "parent_recording_id")


def test_metrics_report_missing_activity_support():
    metrics = classification_metrics([0, 0], [[2, 0], [3, 0]], ["curl", "press"])
    assert metrics["missing_activities"] == ["press"]
    assert metrics["macro_f1"] == pytest.approx(0.5)


def test_config_normalizes_scientific_notation_and_rejects_target_features(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("training:\n  learning_rate: 1e-3\n")
    assert load_config(config)["training"]["learning_rate"] == 0.001
    config.write_text("feature_columns: [activityEncoded]\n")
    with pytest.raises(ValueError, match="sensor columns"):
        load_config(config)
