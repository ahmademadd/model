"""Evaluate a saved model on an explicit partition of its frozen manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .manifest import read_manifest
from .preprocessing import preprocess_quaternions
from .sequence import load_sequence


def classification_metrics(labels, logits, classes: list[str]) -> dict:
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits)
    if len(labels) == 0 or logits.shape != (len(labels), len(classes)) or not np.isfinite(logits).all():
        raise ValueError("Metrics require nonempty labels and finite logits matching the activity vocabulary")
    if np.any(labels < 0) or np.any(labels >= len(classes)):
        raise ValueError("Activity indices are outside the saved vocabulary")
    predictions = logits.argmax(axis=1)
    all_ids = list(range(len(classes)))
    present = sorted(set(labels.tolist()))
    matrix = confusion_matrix(labels, predictions, labels=all_ids)
    recalls = [float(matrix[i, i] / matrix[i].sum()) for i in present]
    return {
        "n_samples": len(labels),
        "classes": classes,
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=all_ids, average="macro", zero_division=0)),
        "macro_f1_scope": "all training activities; missing-support activities contribute zero",
        "balanced_accuracy_present_classes": float(np.mean(recalls)),
        "missing_activities": [classes[i] for i in all_ids if i not in present],
        "per_activity": classification_report(labels, predictions, labels=all_ids,
                                              target_names=classes, output_dict=True, zero_division=0),
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_run(run_dir: str | Path, split: str = "test") -> dict:
    from .config import load_config
    from .dataset import make_dataset
    from .predict import load_bundle
    from .train import validate_splits

    if split not in {"train", "validation", "test"}:
        raise ValueError("Unknown evaluation split")
    root = Path(run_dir)
    config = load_config(root / "config.yaml")
    partitions = validate_splits(read_manifest(root / "manifest.csv"), config["group_column"])
    records = partitions[split]
    if not records:
        raise ValueError(f"No accepted samples in {split} partition")
    model, preprocessing, classes = load_bundle(root)
    class_ids = {label: index for index, label in enumerate(classes)}
    labels = [class_ids[row["activity"]] for row in records]
    sequences = []
    for row in records:
        sequence = load_sequence(row["path"], preprocessing.feature_columns)
        sequence = preprocess_quaternions(sequence, preprocessing.feature_columns)
        sequences.append(preprocessing.transform(sequence))
    dataset = make_dataset(sequences, labels, config["training"]["batch_size"])
    logits = np.concatenate([model(batch, training=False).numpy() for batch, _ in dataset])
    metrics = classification_metrics(labels, logits, classes)
    metrics["split"] = split
    (root / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    args = parser.parse_args()
    try:
        print(json.dumps(evaluate_run(args.run_dir, args.split), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
