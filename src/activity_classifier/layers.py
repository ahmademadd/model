"""Serializable TensorFlow layers for right-padded sensor sequences."""

import tensorflow as tf
from tensorflow import keras


@keras.utils.register_keras_serializable(package="activity_classifier")
class ValidateSequence(keras.layers.Layer):
    """Validate a nonempty right-padded sequence and zero its padding.

    ``valid`` describes whole timesteps, not missing individual measurements.
    Only real sensor measurements must be finite; padded values are discarded.
    """

    def call(self, inputs):
        sensors, valid = inputs
        cleaned = tf.where(valid[..., tf.newaxis], sensors, tf.zeros_like(sensors))
        checks = [
            tf.debugging.assert_equal(
                tf.shape(sensors)[:2],
                tf.shape(valid),
                message="Sensor and validity-mask batch/time dimensions must match.",
            ),
            tf.debugging.assert_positive(
                tf.reduce_sum(tf.cast(valid, tf.int32), axis=1),
                message="Every sample must contain at least one valid timestep.",
            ),
            tf.debugging.assert_equal(
                tf.reduce_any(tf.logical_and(tf.logical_not(valid[:, :-1]), valid[:, 1:])),
                False,
                message="Validity masks must be right-padded: True values then False values.",
            ),
            tf.debugging.assert_all_finite(
                cleaned, "Sensor measurements at valid timesteps must be finite."
            ),
        ]
        with tf.control_dependencies([check for check in checks if check is not None]):
            return tf.identity(cleaned)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


@keras.utils.register_keras_serializable(package="activity_classifier")
class MaskedMeanMaxPooling1D(keras.layers.Layer):
    """Concatenate the mean and maximum over real timesteps of each sample."""

    def call(self, inputs):
        features, valid = inputs
        weights = tf.cast(valid[..., tf.newaxis], features.dtype)
        lengths = tf.reduce_sum(weights, axis=1)
        check = tf.debugging.assert_positive(
            lengths, message="Cannot pool a sample without valid timesteps."
        )
        with tf.control_dependencies([] if check is None else [check]):
            masked_features = tf.where(
                valid[..., tf.newaxis], features, tf.zeros_like(features)
            )
            means = tf.reduce_sum(masked_features, axis=1) / lengths
            maxima = tf.reduce_max(
                tf.where(
                    valid[..., tf.newaxis],
                    features,
                    tf.cast(float("-inf"), features.dtype),
                ),
                axis=1,
            )
            return tf.concat([means, maxima], axis=-1)

    def compute_output_shape(self, input_shape):
        features_shape, _ = input_shape
        channels = features_shape[-1]
        return (features_shape[0], None if channels is None else 2 * channels)
