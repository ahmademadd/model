#!/usr/bin/env python3
"""Express each chopped motion's orientations in its starting reference frame."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QUATERNION_COLUMNS = tuple(f"wristMotion_quaternion{axis}" for axis in "WXYZ")


def normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quaternion must contain finite numbers")
    norm = math.hypot(*values)
    if norm == 0:
        raise ValueError("quaternion must have a nonzero norm")
    return tuple(value / norm for value in values)


def multiply(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    """Hamilton product, with scalar-first (W, X, Y, Z) components."""
    w, x, y, z = a
    v, i, j, k = b
    return (w*v - x*i - y*j - z*k,
            w*i + x*v + y*k - z*j,
            w*j - x*k + y*v + z*i,
            w*k + x*j - y*i + z*v)


def preprocess_file(source: Path, destination: Path) -> int:
    with source.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if fields is None or not set(QUATERNION_COLUMNS).issubset(fields):
            raise ValueError(f"{source}: missing quaternion columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: no motion samples")

    reference_inverse = None
    previous = None
    for line, row in enumerate(rows, start=2):
        try:
            quaternion = normalize(tuple(float(row[key]) for key in QUATERNION_COLUMNS))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{source}:{line}: invalid quaternion ({error})") from error
        if reference_inverse is None:
            w, x, y, z = quaternion
            reference_inverse = (w, -x, -y, -z)
            relative = (1.0, 0.0, 0.0, 0.0)
        else:
            relative = normalize(multiply(reference_inverse, quaternion))
        # q and -q represent the same rotation; avoid artificial sign jumps.
        if previous is not None and sum(a*b for a, b in zip(previous, relative)) < 0:
            relative = tuple(-value for value in relative)
        previous = relative
        for key, value in zip(QUATERNION_COLUMNS, relative):
            row[key] = repr(value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "chopped_data")
    parser.add_argument("--output", type=Path, default=ROOT / "preprocessing" / "data")
    args = parser.parse_args()
    source_root, output_root = args.input.resolve(), args.output.resolve()
    if source_root == output_root or source_root in output_root.parents:
        parser.error("output must be outside the input directory")
    sources = sorted(source_root.rglob("*.csv"))
    motions = [path for path in sources if path.name != "manifest.csv"]
    if not motions:
        parser.error(f"no motion CSVs found under {source_root}")
    total = 0
    for source in motions:
        total += preprocess_file(source, output_root / source.relative_to(source_root))
    for source in sources:
        if source.name == "manifest.csv":
            destination = output_root / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    print(f"Preprocessed {len(motions)} motions ({total} samples) into {output_root}")


if __name__ == "__main__":
    main()
