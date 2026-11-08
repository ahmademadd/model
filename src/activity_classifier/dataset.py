"""Create right-padded TensorFlow batches with explicit validity masks."""

from __future__ import annotations

import numpy as np


def make_dataset(sequences: list[np.ndarray], labels: list[int], batch_size: int = 32,
                 shuffle: bool = False, seed: int = 42):
    if not sequences or len(sequences) != len(labels):
        raise ValueError("Provide matching, nonempty sequences and labels")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    arrays = [np.asarray(sequence, dtype=np.float32) for sequence in sequences]
    if arrays[0].ndim != 2 or arrays[0].shape[1] == 0:
        raise ValueError("Sequences must have shape (timesteps, features)")
    width = arrays[0].shape[1]
    for values in arrays:
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width or not np.isfinite(values).all():
            raise ValueError("Sequences must be finite, nonempty, and share the same feature count")
    for label in labels:
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)) or not 0 <= label <= np.iinfo(np.int32).max:
            raise ValueError("Labels must be nonnegative int32 class indices")

    # Audit, splitting, and preprocessing remain usable without TensorFlow installed.
    import tensorflow as tf

    def examples():
        for values, label in zip(arrays, labels):
            yield {"sensors": values, "valid": np.ones(len(values), dtype=bool)}, np.int32(label)

    dataset = tf.data.Dataset.from_generator(examples, output_signature=(
        {"sensors": tf.TensorSpec((None, width), tf.float32),
         "valid": tf.TensorSpec((None,), tf.bool)}, tf.TensorSpec((), tf.int32),
    ))
    # from_generator otherwise hides the known number of examples from Keras.
    dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(arrays)))
    if shuffle:
        dataset = dataset.shuffle(len(arrays), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.padded_batch(
        batch_size, padded_shapes=({"sensors": (None, width), "valid": (None,)}, ()),
        padding_values=({"sensors": np.float32(0), "valid": False}, np.int32(0)),
    )
    return dataset.prefetch(tf.data.AUTOTUNE)
