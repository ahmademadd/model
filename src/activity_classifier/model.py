"""Build a Keras TCN that assigns one activity to a complete repetition."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

from tensorflow import keras
from tcn import TCN

from .layers import MaskedMeanMaxPooling1D, ValidateSequence


DEFAULT_MODEL_CONFIG = {
    "filters": 64,
    "kernel_size": 5,
    "dilations": [1, 2, 4, 8, 16, 32],
    "stacks": 1,
    "dropout": 0.1,
    "dense_units": 128,
    "head_dropout": 0.2,
}


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}; received {value!r}.")
    return int(value)


def _dropout_rate(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError(f"{name} must be a finite number in [0, 1); received {value!r}.")
    return float(value)


def build_model(n_features: int, n_classes: int, config: dict | None = None):
    """Return an uncompiled, variable-length activity-classification model.

    Inputs are ``sensors`` (float32, B x T x F) and ``valid`` (bool, B x T).
    Output is B x K logits. Use sparse categorical cross-entropy with
    ``from_logits=True`` and apply softmax when producing probabilities.

    Causal forward convolutions and channel-only LayerNorm prevent right-padding
    from influencing real outputs. These settings are deliberately fixed: stock
    keras-tcn does not propagate timestep masks through its convolutions.
    """
    n_features = _positive_int(n_features, "n_features")
    n_classes = _positive_int(n_classes, "n_classes", minimum=2)
    if config is not None and not isinstance(config, Mapping):
        raise ValueError("Model config must be a mapping of model settings.")
    provided = dict(config or {})
    unknown = set(provided) - set(DEFAULT_MODEL_CONFIG)
    if unknown:
        raise ValueError(f"Unsupported model settings: {', '.join(sorted(unknown))}.")
    settings = {**DEFAULT_MODEL_CONFIG, **provided}
    for name in ("filters", "kernel_size", "stacks", "dense_units"):
        settings[name] = _positive_int(settings[name], name)
    for name in ("dropout", "head_dropout"):
        settings[name] = _dropout_rate(settings[name], name)
    dilations = settings["dilations"]
    if not isinstance(dilations, (list, tuple)) or not dilations:
        raise ValueError("dilations must be a nonempty list or tuple of positive integers.")
    dilations = [_positive_int(value, "dilation") for value in dilations]

    sensors = keras.Input(shape=(None, n_features), dtype="float32", name="sensors")
    valid = keras.Input(shape=(None,), dtype="bool", name="valid")
    x = ValidateSequence(name="validate_sequence")([sensors, valid])
    x = TCN(
        nb_filters=settings["filters"],
        kernel_size=settings["kernel_size"],
        nb_stacks=settings["stacks"],
        dilations=dilations,
        padding="causal",
        use_skip_connections=True,
        dropout_rate=settings["dropout"],
        return_sequences=True,
        activation="relu",
        use_layer_norm=True,
        use_batch_norm=False,
        go_backwards=False,
        name="tcn",
    )(x)
    x = MaskedMeanMaxPooling1D(name="masked_mean_max")([x, valid])
    x = keras.layers.Dense(settings["dense_units"], activation="relu", name="features")(x)
    x = keras.layers.Dropout(settings["head_dropout"], name="head_dropout")(x)
    logits = keras.layers.Dense(n_classes, name="activity_logits")(x)
    return keras.Model(inputs={"sensors": sensors, "valid": valid}, outputs=logits,
                       name="activity_tcn")
