"""One synthetic training, evaluation, and inference smoke test."""

import csv

import numpy as np
import pytest

from activity_classifier.audit import build_manifest
from activity_classifier.config import DEFAULT_FEATURES, load_config
from activity_classifier.evaluate import evaluate_run
from activity_classifier.manifest import write_manifest
from activity_classifier.predict import Predictor
from activity_classifier.train import train


def test_training_evaluation_and_inference(tmp_path):
    rng = np.random.default_rng(42)
    for partition in ("train", "validation", "test"):
        for index, label in enumerate(("curl", "press")):
            path = tmp_path / "data" / f"{partition}-{label}" / "part_001.csv"
            path.parent.mkdir(parents=True)
            values = rng.normal(size=(12 + index * 3, len(DEFAULT_FEATURES)))
            angle = np.linspace(0.6, 1.4, len(values))
            values[:, -4:] = np.column_stack((2 * np.cos(angle / 2), np.zeros(len(values)),
                                              np.zeros(len(values)), 2 * np.sin(angle / 2)))
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["secondsElapsed", *DEFAULT_FEATURES, "activity"])
                for timestep, row in enumerate(values):
                    writer.writerow([timestep / 100, *row, label])
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
    assert train(manifest, run_dir, config)["n_samples"] == 2
    assert not (run_dir / "test_metrics.json").exists()
    metrics = evaluate_run(run_dir)
    assert metrics["n_samples"] == 2
    assert np.asarray(metrics["confusion_matrix"]).shape == (2, 2)
    prediction = Predictor(run_dir).predict_file(
        tmp_path / "data" / "test-curl" / "part_001.csv"
    )
    assert prediction["activity"] in ("curl", "press")
    assert set(prediction["probabilities"]) == {"curl", "press"}
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)
