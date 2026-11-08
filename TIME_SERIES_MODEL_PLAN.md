# Plan: classify activity from one complete repetition

## 1. Confirmed task and recommendation

Each input sample contains one complete repetition, typically 300–400 rows of multivariate sensor data. The only prediction is its activity. This is **single-label, multiclass sequence classification**: one activity per repetition, selected from a vocabulary of K activities.

Use a compact residual temporal convolutional network (TCN) with one classification head as the first neural model. Compare it with a feature-based linear model and boosted trees before selecting the final model. Variable sample lengths and the number of activity classes are separate concerns: padding and masking handle length; the classification head handles classes.

**Implemented Keras starting point:** the [project workflow](README.md) uses stock `keras-tcn` with causal forward convolutions, channel-only LayerNorm, explicit right-padding masks at pooling, and one activity head. This library-specific design supersedes the noncausal candidate below for the initial implementation; noncausal masking remains a separate future experiment.

No weight, set, repetition-count, or auxiliary prediction heads are required. Every repetition gets one target, even when the CSV repeats that target in every row. Do not treat the 300–400 rows as independent training examples.

## 2. Input and target contract

- Input: one sensor sequence with shape `(T, F)`, where T is the number of timesteps and F is the number of selected sensor channels.
- Target: one integer activity index in `[0, K - 1]`.
- Model output: K unnormalized scores, called logits.
- Training loss: multiclass cross-entropy, computed directly from logits and the integer target.
- Inference: softmax converts scores to activity probabilities; argmax selects one activity.

Example:

```text
Input: 347 timesteps × 13 sensor channels
Target: activity index 2

Output vocabulary: [activity_A, activity_B, activity_C, activity_D]
Probabilities:     [    0.03,       0.04,       0.91,       0.02]
Prediction: activity_C
```

The example uses placeholder class names and probabilities. The real class mapping must come from a versioned activity dictionary.

Use one consistent mapping between activity names and contiguous class indices. Check the existing activity encoding rather than assuming it is contiguous or stable. Exclude both `activity` and `activityEncoded` from inputs. Exclude `weight`, `set`, and `reps` as predictors for the initial sensor-only model; retain them only as metadata for audits where useful.

## 3. Project context and dataset audit

The inspected processed CSV contains three rotation-rate channels, three gravity channels, three acceleration channels, four quaternion components, elapsed time, and label/metadata columns. This supports an initial 13-channel sensor configuration. Validate units and coordinate frames before implementation.

The existing [quaternion preprocessing](preprocessing/README.md) normalizes quaternions, aligns consecutive signs, and expresses orientation relative to the start of each motion. These observations do not establish overall data quality or sample counts; a full audit remains necessary.

Create a manifest with one row per repetition containing:

| Field | Purpose |
|---|---|
| `sample_id`, `path`, `activity` | Sample identity and sole prediction target |
| `parent_recording_id`, `set_id` | Link related repetitions and alternate processed copies |
| `subject_id`, `session_id`, `device_id` | Grouping and generalization analysis |
| `start_time`, `end_time`, `n_rows`, `duration_seconds` | Boundary and timing validation |
| `quality_flag`, `preprocessing_version`, `split` | Reproducibility and auditability |

Audit independent group counts, class coverage, repetition boundaries, sequence lengths, sampling intervals, missing values, duplicate timestamps, sensor saturation, and overlapping or duplicate samples. Confirm that each segment contains exactly one complete repetition of its labeled activity. Review incomplete repetitions, transitions, and idle periods separately.

Repeated rows or many repetitions from one set do not establish broad class coverage. Record class counts at both repetition and subject/session/recording levels. Inspect whether activities are confounded with particular people, loads, devices, or collection sessions.

## 4. Splitting and leakage prevention

A repetition is an independent prediction unit, but repetitions from the same set or recording are statistically related. Do not randomly distribute neighboring repetitions from one recording across training and test sets.

| Intended use | Appropriate split | Interpretation |
|---|---|---|
| Classify repetitions from new people | Hold out entire subjects | Population generalization |
| Classify new sessions from known people | Hold out complete sessions | Session generalization |
| Personalized future use | Train on earlier sessions, test on later sessions | Personalization over time |

At minimum, keep every repetition and processed copy from the same parent recording in one partition. Where recordings share a session, use session grouping when the deployment goal requires it. If the dataset contains only one person, subject generalization cannot be measured.

Use an approximately 70/15/15 training/validation/test allocation only if enough independent groups and class coverage exist. Otherwise use grouped cross-validation, with nested grouping for unbiased model selection when feasible. Preserve grouping even when it prevents perfectly balanced class counts, and report classes absent from evaluation.

Fit preprocessing statistics using training data only. Choose architecture, hyperparameters, and any rejection threshold using validation data. Evaluate the selected model on the final test set without tuning against its results. Augmentation happens after splitting and only on training examples.

## 5. Variable-length repetitions

Keep each complete repetition at its observed length by default. Right-pad to the longest sequence in each batch and pass a boolean valid-position mask.

```text
Sample lengths: 310, 347, 389
Batch sensor tensor: (3, 389, F)
Batch validity mask: (3, 389)
Keras TCN input: (3, 389, F), with no transpose
```

Zero-pad after normalization. Zero is also a valid sensor value, so the mask is mandatory. Mask padded activations throughout convolution blocks and exclude them from pooling. Use channel-only LayerNorm at each timestep, or an alternative whose statistics exclude padding. A conventional pooling layer or normalization across padded time positions can contaminate results.

For the implemented causal Keras encoder, right-padding cannot affect earlier valid outputs, so internal masking is unnecessary under these constraints; the explicit mask is applied at input validation and pooling. A noncausal encoder would need masking within its convolution blocks.

Compare the following policies if necessary:

| Policy | Advantage | Tradeoff |
|---|---|---|
| Preserve lengths and use masks | Retains duration and complete motion | More careful neural implementation |
| Resample to a common physical sampling rate, then pad | Consistent temporal meaning across recordings | Interpolation and gap handling required |
| Resample each repetition to a fixed number of points | Simple fixed-size batches and phase alignment | Changes timing information and suppresses speed differences |
| Truncate or crop | Fixed compute | Can remove a meaningful movement phase; not the default |

Row-count variation is not necessarily sampling-rate variation. Inspect timestamps first. Avoid splitting a complete repetition into arbitrary short windows unless a later deployment requirement justifies changing the task.

## 6. Sensor preprocessing

1. Validate timestamps, feature order, units, and synchronization. Establish explicit duplicate and gap policies.
2. Resample to a common physical frequency only when needed. Anti-alias before downsampling; interpolate only short, defensible gaps.
3. Standardize ordinary sensor channels with training-set statistics. Consider robust scaling if outliers are substantial.
4. Keep quaternion geometry valid: unit normalization, consistent sign convention, and rotation-aware interpolation such as SLERP if resampling orientation.
5. Compare the existing start-relative orientation with absolute orientation. Relative orientation reduces dependence on starting pose but discards absolute orientation that may distinguish activities.
6. Audit the coordinate frames of acceleration, gravity, and rotation rate. Changing quaternion coordinates alone does not make all channels invariant to sensor orientation.
7. Consider duration and vector magnitudes as explicit optional features, evaluated through ablations.

Avoid per-repetition scaling by default: it can erase useful motion amplitude and posture information. Sensor noise or rotation augmentation must preserve activity identity and physical consistency. Reversing time or heavy cropping can change the meaning of a repetition.

## 7. Recommended architecture

```text
One repetition: (T, F) + valid mask
    ↓
Training-fitted preprocessing
    ↓
1×1 projection: F → 64 channels
    ↓
6 residual temporal convolution blocks
    - two convolutions per block
    - kernel size 5
    - dilations 1, 2, 4, 8, 16, 32
    - channel-only normalization, activation, dropout
    - padding masked through the blocks
    ↓
Masked mean pooling + masked max pooling
    ↓
Dense projection to a 128-value embedding
    ↓
Single linear activity head: 128 → K
    ↓
K activity logits
```

At stride 1, this configuration has a theoretical receptive field of `1 + 2 × (5 - 1) × (1 + 2 + 4 + 8 + 16 + 32) = 505` timesteps. It is a starting configuration spanning typical 300–400-row inputs, not a guarantee of useful learned context. Compare a smaller model if independent training data are limited.

Masked mean divides the sum of valid activations by valid length. Masked max excludes padding entirely, including when valid activations are negative. Reject empty sequences.

Because the complete repetition is supplied, noncausal temporal convolutions may use context on both sides of each timestep. No streaming-state machinery is necessary for the current task. The fixed-size pooled embedding allows different input lengths without changing the activity head.

## 8. Model alternatives and tradeoffs

| Model | Benefits | Costs | Role |
|---|---|---|---|
| Majority predictor | Reveals class imbalance | Ignores sensor data | Required reference |
| Summary features + linear classifier | Fast, interpretable, relatively low data demand | Compresses temporal shape | First learned baseline |
| Summary features + boosted trees | Strong nonlinear baseline | Feature engineering; loses some temporal order | Main non-neural competitor |
| Residual TCN | Efficient temporal pattern learning; handles masked variable lengths | Masking and receptive-field design need care | First neural candidate |
| GRU/LSTM | Explicit sequential state | Sequential processing; additional state-handling concerns | Follow-up if ordered phases are poorly captured |
| InceptionTime-style CNN | Learns multiple temporal scales | More implementation and tuning effort | Follow-up neural comparison |
| Transformer | Flexible temporal interactions | More tuning and data demand; unnecessary complexity may not pay off | Later only if simpler models leave a measured gap |

Baseline features can include channel statistics, RMS, quantiles, energy, slopes, selected autocorrelations, cross-axis relationships, and frequency summaries using the actual sampling rate. Include a duration-only baseline to check whether duration is acting as a shortcut.

Do not select a model solely from architecture reputation. Compare models using the same group-safe evaluation and choose the simpler model when the performance difference is not practically convincing.

## 9. Training configuration

Use multiclass cross-entropy on logits, with one integer target per repetition. Do not apply softmax before a standard cross-entropy implementation that already incorporates it.

Illustrative initial settings:

| Setting | Initial choice |
|---|---|
| Optimizer | AdamW |
| Learning rate | `1e-3`; compare `3e-4` |
| Weight decay | `1e-4` to `1e-2` |
| Batch size | 16–64 repetitions, subject to memory and dataset size |
| Dropout | 0.1–0.3 |
| Epoch cap | 100–200, with early stopping |
| Early stopping | Validation macro-F1, patience approximately 15 epochs |
| Schedule | Plateau reduction or cosine decay |
| Gradient clipping | Initial maximum norm 1.0 |

For imbalanced activities, compare unweighted cross-entropy with training-derived class weights. Oversampling one rare recording does not replace collecting independent examples. Keep augmentation conservative: small realistic noise, label-preserving boundary adjustments, and physically consistent orientation perturbations where appropriate.

Compare final candidates over several random seeds when resources allow. Record the class mapping, manifest, preprocessing version, split, seed, configuration, and checkpoint.

## 10. Evaluation and verification

Use macro-F1 as the primary validation metric, supported by balanced accuracy, overall accuracy, per-activity precision/recall/F1, class support, and a confusion matrix. Macro-F1 gives each activity equal weight; overall accuracy can hide poor performance on rare activities.

Report results by subject, session, device, repetition length, duration, and available load metadata. Bootstrap independent subjects or sessions for uncertainty estimates rather than treating correlated repetitions as independent. Report unsupported classes and limitations from small group counts.

Verify:

- The same repetition produces equivalent predictions when only batch padding changes.
- No activity or metadata columns enter the sensor tensor accidentally.
- Related repetitions and processed copies cannot cross partitions.
- Training and inference preprocessing produce consistent results.
- Empty samples, invalid sensors, and samples outside the typical length range have defined handling.
- The model improves meaningfully over majority, duration-only, and feature-based baselines.

Softmax probabilities are relative scores over known activities and are not automatically calibrated. An unfamiliar activity may still receive a confident known-class prediction. If abstention is needed, tune and evaluate it with representative validation examples; do not assume a confidence threshold solves unknown-activity detection.

## 11. Scaling to more activity classes

Adding activities expands the single output head, not the number of full models. A linear head from a 128-value embedding to K activities has `128 × K + K = 129K` parameters.

| Activities | Head parameters |
|---|---:|
| 10 | 1,290 |
| 50 | 6,450 |
| 100 | 12,900 |

These counts exclude the encoder and embedding projection. Keep one output per activity; a sample still receives exactly one selected activity.

When adding a class:

1. Collect representative labeled repetitions from independent people/sessions as appropriate.
2. Extend the versioned activity vocabulary and output layer.
3. Optionally freeze the encoder while initializing the new output, then fine-tune jointly.
4. Include old-class examples during training to reduce forgetting.
5. Reevaluate all activities, the confusion matrix, and calibration; softmax scores can change for existing classes too.

Expanding the head does not teach the model a new activity automatically. The main scaling limits are independent data coverage and whether similar activities are distinguishable from the available sensors. Two activities with nearly identical wrist motion may require additional sensor placement or context.

## 12. Deployment and implementation phases

Deployment receives an already segmented complete repetition. Package feature order and units, preprocessing, model weights, class mapping, and input-quality policies together. Return the selected activity, optional activity probabilities, and any validated quality/rejection flag.

Measure latency and memory on the actual target device. Total user-visible delay includes collecting the repetition, detecting its endpoint if applicable, and running inference. A future continuous-stream system would require a separately evaluated segmentation stage; it is outside the current classifier's responsibility.

Implementation sequence:

1. Build and audit the repetition manifest; verify activity labels and parent IDs.
2. Freeze group-safe splits and preprocessing conventions.
3. Train majority, duration-only, and feature-based baselines.
4. Implement the masked TCN and its single activity head.
5. Compare model size, orientation representation, and length-handling options using validation data.
6. Select the model, evaluate once on the final test set, and package the complete inference pipeline.

Remaining decisions are the activity vocabulary, independent subject/session counts, intended generalization population, actual sampling characteristics, and deployment hardware/latency budget. The prediction target and prediction unit are settled: **one complete repetition → one activity**.
