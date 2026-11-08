"""Load a complete run bundle and classify an already-segmented repetition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .preprocessing import Standardizer, load_sequence


def load_bundle(run_dir: str | Path):
    from tensorflow import keras
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


def predict_file(run_dir: str | Path, path: str | Path) -> dict:
    from tensorflow import keras

    model, preprocessing, classes = load_bundle(run_dir)
    sequence = preprocessing.transform(load_sequence(path, preprocessing.feature_columns))
    inputs = {"sensors": sequence[None, ...], "valid": np.ones((1, len(sequence)), dtype=bool)}
    probabilities = keras.ops.softmax(model(inputs, training=False), axis=-1).numpy()[0]
    if not np.isfinite(probabilities).all():
        raise ValueError("Model produced nonfinite probabilities")
    return {
        "activity": classes[int(np.argmax(probabilities))],
        "probabilities": {label: float(value) for label, value in zip(classes, probabilities)},
        "n_rows": len(sequence),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="One complete repetition CSV")
    args = parser.parse_args()
    try:
        print(json.dumps(predict_file(args.run_dir, args.input), indent=2))
    except (ValueError, OSError, KeyError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
