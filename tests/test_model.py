"""Core contracts for padding, masked pooling, and model serialization."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from activity_classifier.dataset import make_dataset
from activity_classifier.layers import MaskedMeanMaxPooling1D
from activity_classifier.model import build_model


@pytest.fixture
def model():
    keras.utils.set_random_seed(17)
    return build_model(3, 4, config={
        "filters": 8, "kernel_size": 3, "dilations": [1, 2, 4],
        "dense_units": 8, "dropout": 0.0, "head_dropout": 0.0,
    })


def test_predictions_ignore_padding_and_other_batch_members(model):
    rng = np.random.default_rng(11)
    sequences = [rng.normal(size=(length, 3)).astype("float32") for length in (37, 61)]
    sequences[0][0] = 0  # A real all-zero row must remain valid.
    inputs, labels = next(iter(make_dataset(sequences, [0, 1], batch_size=2)))
    assert inputs["sensors"].shape == (2, 61, 3)
    np.testing.assert_array_equal(inputs["valid"][0], [True] * 37 + [False] * 24)
    np.testing.assert_array_equal(labels, [0, 1])
    single = {"sensors": sequences[0][None, ...], "valid": np.ones((1, 37), dtype=bool)}
    expected = model(single, training=False).numpy()[0]
    np.testing.assert_allclose(model(inputs, training=False).numpy()[0], expected, rtol=1e-5, atol=1e-5)
    padded = {name: value.numpy() for name, value in inputs.items()}
    padded["sensors"][0, 37:] = np.nan
    np.testing.assert_allclose(model(padded, training=False).numpy()[0], expected, rtol=1e-5, atol=1e-5)


def test_pooling_excludes_padding_from_mean_and_negative_maximum():
    features = tf.constant([[[-4.0, 2.0], [-2.0, 6.0], [999.0, 999.0]]])
    result = MaskedMeanMaxPooling1D()([features, tf.constant([[True, True, False]])])
    np.testing.assert_allclose(result.numpy(), [[-3.0, 4.0, -2.0, 6.0]])


def test_save_and_reload_preserve_predictions(model, tmp_path):
    inputs = {"sensors": np.ones((1, 19, 3), dtype="float32"),
              "valid": np.ones((1, 19), dtype=bool)}
    expected = model(inputs, training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(path)
    restored = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(restored(inputs, training=False).numpy(), expected, rtol=1e-5, atol=1e-5)
