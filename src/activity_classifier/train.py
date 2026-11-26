"""Train a repetition classifier from a reviewed, already-split manifest."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .config import load_config, save_config
from .manifest import read_manifest, write_manifest
from .preprocessing import Standardizer, preprocess_quaternions
from .sequence import load_sequence


def validate_splits(records: list[dict], group_column: str) -> dict[str, list[dict]]:
    partitions = {name: [] for name in ("train", "validation", "test")}
    groups: dict[tuple[str, str], str] = {}
    paths: set[str] = set()
    ids: set[str] = set()
    for row in records:
        if row.get("quality_flag") != "accepted":
            continue
        split = row.get("split", "")
        if split not in partitions:
            raise ValueError(f"Accepted sample {row.get('sample_id')} has no valid split")
        if not row.get("activity", "").strip() or not row.get("sample_id", "").strip():
            raise ValueError("Each accepted sample needs a sample_id and activity")
        for column in {"parent_recording_id", group_column}:
            group = row.get(column, "").strip()
            if not group:
                raise ValueError(f"Missing {column} for {row['sample_id']}")
            key = (column, group)
            if key in groups and groups[key] != split:
                raise ValueError(f"Leakage: {column}={group} occurs in multiple splits")
            groups[key] = split
        source = str(Path(row["path"]).resolve())
        if source in paths or row["sample_id"] in ids:
            raise ValueError("Manifest contains a duplicate sample ID or source path")
        paths.add(source)
        ids.add(row["sample_id"])
        partitions[split].append(row)
    if not partitions["train"] or not partitions["validation"]:
        raise ValueError("Training requires nonempty train and validation partitions")
    classes = {row["activity"] for row in partitions["train"]}
    if len(classes) < 2:
        raise ValueError("Activity classification requires at least two training activities")
    for split in ("validation", "test"):
        unseen = {row["activity"] for row in partitions[split]} - classes
        if unseen:
            raise ValueError(f"{split} contains activities absent from training: {sorted(unseen)}")
    return partitions


def train(manifest: str | Path, run_dir: str | Path, config: dict) -> dict:
    import keras

    from .dataset import make_dataset
    from .evaluate import classification_metrics
    from .model import build_model

    records = read_manifest(manifest)
    partitions = validate_splits(records, config["group_column"])
    destination = Path(run_dir)
    if destination.exists():
        raise ValueError(f"Run directory already exists: {destination}; choose a new name")
    classes = sorted({row["activity"] for row in partitions["train"]})
    class_ids = {label: index for index, label in enumerate(classes)}
    features = config["feature_columns"]
    # Deliberately do not read test sensor data during training/model selection.
    train_raw = [load_sequence(row["path"], features) for row in partitions["train"]]
    validation_raw = [load_sequence(row["path"], features) for row in partitions["validation"]]
    train_rotations = [preprocess_quaternions(sequence, features) for sequence in train_raw]
    validation_rotations = [preprocess_quaternions(sequence, features) for sequence in validation_raw]
    preprocessing = Standardizer().fit(train_rotations, features)
    train_x = [preprocessing.transform(sequence) for sequence in train_rotations]
    validation_x = [preprocessing.transform(sequence) for sequence in validation_rotations]
    train_y = [class_ids[row["activity"]] for row in partitions["train"]]
    validation_y = [class_ids[row["activity"]] for row in partitions["validation"]]
    settings = config["training"]
    keras.utils.set_random_seed(config["seed"])
    model = build_model(len(features), len(classes), config["model"])
    model.compile(
        optimizer=keras.optimizers.Adam(settings["learning_rate"], clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    train_ds = make_dataset(train_x, train_y, settings["batch_size"], True, config["seed"])
    validation_ds = make_dataset(validation_x, validation_y, settings["batch_size"])
    destination.mkdir(parents=True, exist_ok=False)
    save_config(config, destination / "config.yaml")
    write_manifest(records, destination / "manifest.csv")
    preprocessing.save(destination / "preprocessing.json")
    (destination / "class_mapping.json").write_text(
        json.dumps({"classes": classes}, indent=2) + "\n", encoding="utf-8"
    )
    (destination / "environment.json").write_text(json.dumps({
        package: version(package) for package in
        ("tensorflow", "keras", "keras-tcn", "numpy", "scikit-learn", "PyYAML")
    }, indent=2) + "\n", encoding="utf-8")
    model.summary(print_fn=lambda line: print(line))
    callbacks = [
        keras.callbacks.ModelCheckpoint(destination / "model.keras", monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=settings["patience"], restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                         patience=max(1, settings["patience"] // 3)),
        keras.callbacks.CSVLogger(destination / "history.csv"),
        keras.callbacks.TerminateOnNaN(),
    ]
    weights = None
    if settings["class_weight"]:
        counts = np.bincount(train_y, minlength=len(classes))
        weights = {i: len(train_y) / (len(classes) * int(count)) for i, count in enumerate(counts)}
    history = model.fit(train_ds, validation_data=validation_ds, epochs=settings["epochs"],
                        callbacks=callbacks, class_weight=weights, verbose=2)
    if any(not np.isfinite(value) for values in history.history.values() for value in values):
        raise RuntimeError("Training produced nonfinite metrics; this run must not be deployed")
    # Load the actual selected checkpoint so reported metrics match deployment.
    model = keras.models.load_model(destination / "model.keras", compile=False)
    logits = np.concatenate([model(batch, training=False).numpy() for batch, _ in validation_ds])
    metrics = classification_metrics(validation_y, logits, classes)
    (destination / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved run to {destination}. Test evaluation is a separate command.")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    try:
        train(args.manifest, args.run_dir, load_config(args.config))
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
