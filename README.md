# Activity classifier

Classify one complete wrist-motion repetition into one activity using Keras TCN.

Each repetition is a separate CSV with a variable number of rows. Inputs are the
13 sensor channels (rotation rate, gravity, acceleration, WXYZ quaternion); the
only target is `activity`. The model is a causal TCN with channel LayerNorm,
explicit masked mean/max pooling, and one multiclass head.

## Setup

Use one Python 3.11 environment, `.venv-tcn`, for the classifier, notebooks,
and visualizations. The TensorFlow pin does not support Python 3.14.

```sh
python3.11 -m venv .venv-tcn
source .venv-tcn/bin/activate
python -m pip install -e '.[dev,notebooks]'
python -m pytest -q
```

On this Mac, Python 3.11 is also available at `/opt/homebrew/bin/python3.11`.
The shared environment already exists on this checkout; activate it rather than
creating another environment. In VS Code, select `.venv-tcn/bin/python` as the
Python interpreter and **Python (model, 3.11)** as the notebook kernel. If the
kernel is not listed, register it inside the environment:

```sh
python -m ipykernel install --sys-prefix --name model --display-name "Python (model, 3.11)"
```

[pyproject.toml](pyproject.toml) pins TensorFlow 2.19.1, Keras 3.10.0, and
keras-tcn 3.5.6. Version 3.5.6 is used because the published 3.5.7 wheel declares
`tensorflow-macos` as an unconditional dependency, which limits portability.
The TensorFlow backend is required. GPU plugins are optional and are not installed
by this scaffold. [pyproject.toml](pyproject.toml) is the single source of dependency
declarations: the `dev` extra adds test tools, and the `notebooks` extra adds
notebook/visualization dependencies. The setup command installs both extras into
the same environment.

## Structure

| Component | Responsibility |
|---|---|
| [Configuration](configs/activity_tcn.yaml) | Training and model settings |
| [Audit](src/activity_classifier/audit.py) | Validate CSVs and catalog repetitions |
| [Manifest](src/activity_classifier/manifest.py) | Read/write sample metadata and portable paths |
| [Splits](src/activity_classifier/split.py) | Keep complete recording/subject/session groups together |
| [Sequence loading](src/activity_classifier/sequence.py) | Read and validate recorded sensor values without transformations |
| [Preprocessing](src/activity_classifier/preprocessing.py) | Explicit quaternion preprocessing and saved scalar scaling |
| [Dataset](src/activity_classifier/dataset.py) | Right-pad batches and generate explicit masks |
| [Layers](src/activity_classifier/layers.py) | Input validation and serializable masked pooling |
| [Model](src/activity_classifier/model.py) | Import `TCN` and build the activity classifier |
| [Training](src/activity_classifier/train.py) | Train, checkpoint, and save a complete run bundle |
| [Evaluation](src/activity_classifier/evaluate.py) | Per-class metrics and confusion matrix |
| [Prediction](src/activity_classifier/predict.py) | Classify one new repetition |
| [Tests](tests/) | Masking, preprocessing, leakage, serialization, end-to-end workflow |

Existing [preprocessing tools](preprocessing/) and [visualizations](visualization/)
remain available. Use [notebooks](notebooks/) for exploration and the package for
reusable training/inference behavior. The broader design is in the
[modeling plan](TIME_SERIES_MODEL_PLAN.md).

## 1. Catalog repetition files

Run from the project root with the new environment active:

```sh
activity-audit \
  --input data/post_processing_relative_quaternions \
  --output artifacts/manifest.csv
```

By default, only `part_*.csv` files are scanned, avoiding full recordings and
existing manifest files. If your repetition files use another naming scheme,
pass `--pattern '*.csv'` only to a directory containing individual repetitions.

The manifest includes the source path, activity, row count, duration, quality
status, and parent-recording group. Paths are relative to the manifest location.
Invalid samples are retained as `rejected` with reasons. The audit checks finite
sensors, nonzero quaternion norms, increasing timestamps, required columns, and
one constant nonempty activity per CSV. Length outside 300–400 is not by itself
an error. Inference CSVs do not need an activity column.

Auditing uses `sequence.load_sequence()` to read and validate recorded values.
It does not normalize quaternions, change their reference frame or signs, or
standardize channels. Those transformations are explicit model-input steps:
`load_sequence()` → `preprocess_quaternions()` → `Standardizer.transform()`.

**Review the manifest before splitting.** `accepted` means these automatic checks
passed; it does not prove the repetition boundaries or activity annotation are
correct. Duplicate file contents, sensor saturation, sampling-rate consistency,
and segmentation quality still need review. Do not include alternate processed
versions of the same recording as independent samples.

Parent recording IDs are initialized from the containing directory. Confirm that
this matches the actual source recording; merge IDs for segments or variants
derived from the same source. Subject and session IDs start blank and must be
filled from verified provenance, not guessed from activity names or set numbers.

## 2. Create grouped partitions

```sh
activity-split \
  --manifest artifacts/manifest.csv \
  --output artifacts/split_manifest.csv \
  --group-column parent_recording_id
```

The default approximately 70/15/15 allocation is best-effort because groups are
indivisible. Every activity must remain in training. Missing validation/test
classes produce warnings. Impossible allocations fail instead of dividing a
recording or silently discarding activities. Review class/group coverage.

To evaluate new people or sessions, fill those IDs and choose `subject_id` or
`session_id` instead. Set `group_column` in the training config to the same field.
Training also checks that no parent recording crosses partitions. At least two
activities and nonempty training/validation partitions are required to train.

## 3. Train

```sh
activity-train \
  --manifest artifacts/split_manifest.csv \
  --config configs/activity_tcn.yaml \
  --run-dir runs/tcn_001
```

Choose a fresh run directory for every experiment. The training command:

1. Validates split membership, groups, class coverage, unique IDs and paths.
2. Loads accepted training and validation sequences (not test sensor files).
3. Converts quaternions to normalized, sign-continuous, start-relative WXYZ
   rotations with the explicit `preprocess_quaternions()` call; already-relative
   inputs remain compatible. Loading and auditing do not perform this conversion.
4. Fits scalar-channel means/scales on training timesteps only. Quaternion
   components retain their unit-rotation representation.
5. Creates right-padded batches in `(batch, time, features)` order.
6. Trains with sparse categorical cross-entropy from logits, using validation
   loss for early stopping and best-checkpoint selection.
7. Saves the best model and validation macro-F1, per-activity metrics, support,
   and confusion matrix. Compare experiments using those validation results.

The run contains `model.keras`, `config.yaml`, `manifest.csv`,
`preprocessing.json`, `class_mapping.json`, `environment.json`, `history.csv`,
and `validation_metrics.json`. Sensor files are referenced, not copied: retain
them or update manifest paths if moving the run. Artifact JSON schemas record
preprocessing and label order so inference does not recompute statistics.

The initial loader holds sequences in memory. A streaming file-backed loader,
large-scale hyperparameter search, automatic segmentation, and data augmentation
are future extensions, not prerequisites for this scaffold.

## 4. Evaluate after model selection

```sh
activity-evaluate --run-dir runs/tcn_001 --split test
```

This uses the frozen manifest and writes `test_metrics.json` in the run. It
includes accuracy, macro-F1 over the full training vocabulary, per-class scores,
missing-support classes, and the confusion matrix (rows=true, columns=predicted).
Missing-support classes contribute zero to the reported macro-F1; balanced
accuracy is separately reported over classes actually present. Do not repeatedly
tune models against test results.

## 5. Predict a new repetition

```sh
activity-predict --run-dir runs/tcn_001 --input /path/to/repetition.csv
```

The CLI loads the saved model and preprocessing, creates an all-valid mask for
the single repetition, and returns the selected activity and probabilities.
The activity need not be provided at inference. Probabilities are over the known
activities; they do not establish detection of unfamiliar activities.

From Python:

```python
from activity_classifier.predict import predict_file

result = predict_file("runs/tcn_001", "/path/to/repetition.csv")
print(result["activity"])
```

For a long-running service, call `load_bundle()` once and reuse the model and
preprocessing rather than reloading for each request.

## Model contract and masking

```python
from activity_classifier.model import build_model

model = build_model(n_features=13, n_classes=6)
# Inputs: {"sensors": float32[B, T, 13], "valid": bool[B, T]}
# Output: float32[B, 6] logits. Each sample has one integer class target.
```

The encoder uses 64 filters, two kernel-5 convolutions per residual block,
dilations `[1, 2, 4, 8, 16, 32]`, and one stack: receptive field 505 timesteps.
Pooled features feed Dense(128, ReLU), dropout, and one logit per activity.

Stock Keras TCN does not apply timestep masks inside its convolutions. This
implementation deliberately uses **causal forward convolutions, right-padding,
and channel-only LayerNorm** so padded future timesteps cannot affect real
outputs. Pooling explicitly excludes padding. Noncausal padding, BatchNorm,
reversed sequences, and masks with internal gaps are not configuration options.
Handle missing measurements separately before batching.

All-empty sequences and nonfinite real sensor values are rejected. Tests verify
padding/batch invariance, negative-valued masked maxima, preprocessing round
trips, group leakage checks, normal `.keras` save/reload, and one synthetic
training-through-prediction smoke test. Import `activity_classifier.model`
before loading a model manually to register the custom layers and TCN.
