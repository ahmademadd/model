# Quaternion preprocessing

Run from the project root:

```sh
python3 preprocessing/relative_quaternions.py
```

Reads motion CSVs recursively from `chopped_data` and writes processed copies to
`preprocessing/data`, preserving the directory layout and copying manifests.
Each CSV is treated as a separate motion. Original files are preserved.

Quaternions use scalar-first `(W, X, Y, Z)` Hamilton multiplication. Each sample
is normalized, then transformed with `q_relative = inverse(q_start) * q_sample`.
This expresses orientation in the first sample's reference frame; every motion
starts at the identity quaternion `(1, 0, 0, 0)`. Consecutive quaternion signs
are aligned to avoid jumps between equivalent `q` and `-q` representations.
All other values, column order, and row order are preserved. Invalid, missing,
nonfinite, or zero-norm quaternions raise an error with the file and CSV line.

Optional paths: `--input path/to/chopped_data --output path/to/processed_data`.
Rerunning replaces corresponding processed files using the original inputs.
