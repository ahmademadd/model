# Quaternion preprocessing

Run from the project root:

```sh
python3 preprocessing/relative_quaternions.py --input data/chopped_data
```

Reads motion CSVs recursively from the input directory and writes processed copies to
`data/post_processing_relative_quaternions/<input-directory-name>/`, preserving
the directory layout within the input. Manifest CSVs are skipped.
Each CSV is treated as a separate motion. Original files are preserved.

Quaternions use scalar-first `(W, X, Y, Z)` Hamilton multiplication. Each sample
is normalized, then transformed with `q_relative = inverse(q_start) * q_sample`.
This expresses orientation in the first sample's reference frame; every motion
starts at the identity quaternion `(1, 0, 0, 0)`. Consecutive quaternion signs
are aligned to avoid jumps between equivalent `q` and `-q` representations.
All other values, column order, and row order are preserved. Invalid, missing,
nonfinite, or zero-norm quaternions raise an error with the file and CSV line.

Use `--output path/to/processed_data` to change the output parent directory.
For example, `--input path/to/motion_001 --output path/to/processed_data` writes
files under `path/to/processed_data/motion_001/`.
Rerunning replaces corresponding processed files using the original inputs.
