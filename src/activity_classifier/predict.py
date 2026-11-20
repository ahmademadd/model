"""Load a complete run bundle and classify an already-segmented repetition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .preprocessing import Standardizer, preprocess_quaternions
from .sequence import load_sequence


def load_bundle(run_dir: str | Path):
    import keras
    from . import model as _registered_layers  # noqa: F401

    root = Path(run_dir)
    classes = json.loads((root / "class_mapping.json").read_text(encoding="utf-8"))["classes"]
    if (not isinstance(classes, list) or len(classes) < 2
            or any(not isinstance(item, str) or not item for item in classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("Invalid saved activity vocabulary")
    preprocessing = Standardizer.load(root / "preprocessing.json")
    model = keras.models.load_model(root / "model.keras", compile=False)
    if model.output_shape[-1] != len(classes):
        raise ValueError("Model output and activity vocabulary differ")
    sensor_input = next(tensor for tensor in model.inputs if tensor.name.split(":")[0] == "sensors")
    if sensor_input.shape[-1] != len(preprocessing.feature_columns):
        raise ValueError("Model input and saved preprocessing feature counts differ")
    return model, preprocessing, classes


def _predict_file(run_dir: str | Path, path: str | Path) -> dict:
    import keras

    model, preprocessing, classes = load_bundle(run_dir)
    sequence = load_sequence(path, preprocessing.feature_columns)
    sequence = preprocess_quaternions(sequence, preprocessing.feature_columns)
    sequence = preprocessing.transform(sequence)
    inputs = {"sensors": sequence[None, ...], "valid": np.ones((1, len(sequence)), dtype=bool)}
    probabilities = keras.ops.softmax(model(inputs, training=False), axis=-1).numpy()[0]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced nonfinite probabilities")
    return {
        "activity": classes[int(np.argmax(probabilities))],
        "probabilities": {label: float(value) for label, value in zip(classes, probabilities)},
        "n_rows": len(sequence),
    }


def predict(run_dir: str | Path, path: str | Path) -> list[dict]:
    """Predict every CSV in a file or directory, preserving relative file names."""
    source = Path(path)
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(item for item in source.rglob("*.csv") if item.is_file())
        if not files:
            raise ValueError(f"{source}: no CSV files found")
    else:
        raise ValueError(f"{source}: input path is not a file or directory")
    results = []
    for item in files:
        result = _predict_file(run_dir, item)
        result["path"] = item.name if source.is_file() else item.relative_to(source).as_posix()
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="A repetition CSV or directory containing repetition CSVs")
    args = parser.parse_args()
    try:
        results = predict(args.run_dir, args.input)
        print(json.dumps(results[0] if args.input.is_file() else results, indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
