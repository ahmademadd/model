"""Behavioral checks for masking, training, and model serialization."""

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import keras

from activity_classifier.layers import MaskedMeanMaxPooling1D
from activity_classifier.model import build_model


@pytest.fixture
def model():
    keras.utils.set_random_seed(17)
    return build_model(
        n_features=3,
        n_classes=4,
        config={
            "filters": 8,
            "kernel_size": 3,
            "dilations": [1, 2, 4],
            "dense_units": 8,
            "dropout": 0.0,
            "head_dropout": 0.0,
        },
    )


def _batch(lengths, n_features=3):
    rng = np.random.default_rng(11)
    sensors = rng.normal(size=(len(lengths), max(lengths), n_features)).astype("float32")
    valid = np.arange(max(lengths))[None, :] < np.asarray(lengths)[:, None]
    return {"sensors": sensors, "valid": valid}


def test_predictions_ignore_padding_and_other_batch_members(model):
    batch = _batch([37, 61])
    single = {"sensors": batch["sensors"][:1, :37], "valid": batch["valid"][:1, :37]}
    expected = model(single, training=False).numpy()[0]
    np.testing.assert_allclose(model(batch, training=False).numpy()[0], expected,
                               rtol=1e-5, atol=1e-5)

    # Even arbitrary/nonfinite padded values must not change the prediction.
    batch["sensors"][0, 37:] = np.nan
    np.testing.assert_allclose(model(batch, training=False).numpy()[0], expected,
                               rtol=1e-5, atol=1e-5)


def test_pooling_ignores_padding_and_handles_negative_features():
    features = tf.constant([[[-4.0, 2.0], [-2.0, 6.0], [999.0, 999.0]]])
    result = MaskedMeanMaxPooling1D()([features, tf.constant([[True, True, False]])])
    np.testing.assert_allclose(result.numpy(), [[-3.0, 4.0, -2.0, 6.0]])


def test_forward_loss_and_gradients_are_finite(model):
    batch = _batch([21, 31, 27])
    labels = tf.constant([0, 2, 1], dtype=tf.int32)
    with tf.GradientTape() as tape:
        logits = model(batch, training=True)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)(labels, logits)
    gradients = tape.gradient(loss, model.trainable_variables)
    assert tuple(logits.shape) == (3, 4)
    assert np.isfinite(loss.numpy())
    assert all(gradient is not None for gradient in gradients)
    assert all(np.isfinite(gradient.numpy()).all() for gradient in gradients)
    before = [variable.numpy().copy() for variable in model.trainable_variables]
    keras.optimizers.Adam(learning_rate=1e-3).apply_gradients(zip(gradients, model.trainable_variables))
    assert any(not np.array_equal(old, variable.numpy())
               for old, variable in zip(before, model.trainable_variables))


def test_save_and_reload_preserve_predictions(model, tmp_path):
    batch = _batch([19, 29])
    expected = model(batch, training=False).numpy()
    path = tmp_path / "model.keras"
    model.save(path)
    restored = keras.models.load_model(path, compile=False)
    np.testing.assert_allclose(restored(batch, training=False).numpy(), expected,
                               rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("valid", [
    [[False, False, False]],
    [[True, False, True]],
])
def test_invalid_masks_fail_in_eager_and_graph_execution(model, valid):
    batch = {"sensors": tf.ones((1, 3, 3)), "valid": tf.constant(valid)}
    with pytest.raises(tf.errors.InvalidArgumentError):
        model(batch, training=False)
    graph_call = tf.function(lambda inputs: model(inputs, training=False))
    with pytest.raises(tf.errors.InvalidArgumentError):
        graph_call(batch)


def test_nonfinite_real_measurements_are_rejected(model):
    batch = _batch([5])
    batch["sensors"][0, 2, 0] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError):
        model(batch, training=False)


@pytest.mark.parametrize("config", [
    {"padding": "same"},
    {"filters": 0},
    {"dropout": 1.0},
    {"dilations": []},
    {"dilations": [1, -2]},
])
def test_invalid_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        build_model(3, 4, config)
