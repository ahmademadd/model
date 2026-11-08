"""Small, validated configuration shared by command-line entry points."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

DEFAULT_FEATURES = [
    *(f"wristMotion_rotationRate{axis}" for axis in "XYZ"),
    *(f"wristMotion_gravity{axis}" for axis in "XYZ"),
    *(f"wristMotion_acceleration{axis}" for axis in "XYZ"),
    *(f"wristMotion_quaternion{axis}" for axis in "WXYZ"),
]

DEFAULT_CONFIG = {
    "seed": 42,
    "feature_columns": DEFAULT_FEATURES,
    "group_column": "parent_recording_id",
    "model": {
        "filters": 64,
        "kernel_size": 5,
        "dilations": [1, 2, 4, 8, 16, 32],
        "stacks": 1,
        "dropout": 0.1,
        "dense_units": 128,
        "head_dropout": 0.2,
    },
    "training": {
        "batch_size": 32,
        "epochs": 150,
        "learning_rate": 0.001,
        "patience": 15,
        "class_weight": False,
    },
}


def load_config(path: str | Path | None = None) -> dict:
    """Load overrides and reject misspellings instead of silently ignoring them."""
    config = deepcopy(DEFAULT_CONFIG)
    if path is not None:
        with Path(path).open(encoding="utf-8") as handle:
            supplied = yaml.safe_load(handle) or {}
        if not isinstance(supplied, dict):
            raise ValueError("Configuration must be a YAML mapping")
        for key, value in supplied.items():
            if key not in config:
                raise ValueError(f"Unknown configuration key: {key}")
            if isinstance(config[key], dict):
                if not isinstance(value, dict) or set(value) - set(config[key]):
                    raise ValueError(f"Invalid configuration keys in {key}")
                config[key].update(value)
            else:
                config[key] = value
    features = config["feature_columns"]
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("feature_columns must be a nonempty list of unique sensor names")
    if set(features) - set(DEFAULT_FEATURES):
        raise ValueError("Only the declared sensor columns may be used as model inputs")
    for key in ("batch_size", "epochs", "patience"):
        value = config["training"][key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.{key} must be a positive integer")
    learning_rate = config["training"]["learning_rate"]
    if isinstance(learning_rate, bool) or not 0 < float(learning_rate) < 1:
        raise ValueError("learning_rate must be between zero and one")
    config["training"]["learning_rate"] = float(learning_rate)
    if not isinstance(config["training"]["class_weight"], bool):
        raise ValueError("class_weight must be true or false")
    if not isinstance(config["seed"], int) or isinstance(config["seed"], bool):
        raise ValueError("seed must be an integer")
    if config["group_column"] not in {"parent_recording_id", "session_id", "subject_id"}:
        raise ValueError("group_column must be parent_recording_id, session_id, or subject_id")
    return config


def save_config(config: dict, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
