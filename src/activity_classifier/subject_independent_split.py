"""Deterministic, best-effort stratification while preserving whole provenance groups."""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import permutations
from pathlib import Path
import warnings

import numpy as np

from .manifest import read_manifest, write_manifest


def split_records(records: list[dict], group_column: str = "parent_recording_id",
                  seed: int = 42, val_fraction: float = 0.15,
                  test_fraction: float = 0.15) -> list[dict]:
    """Keep every activity in training and require nonempty requested holdout splits.

    Fractions are targets for sample counts; indivisible recording/subject groups
    can make exact proportions or full holdout class coverage impossible. Rejected
    samples are preserved with an empty split. This is a greedy allocation, not an
    optimizer guaranteeing the globally best class balance.
    """
    fractions = np.array([1 - val_fraction - test_fraction, val_fraction, test_fraction], dtype=float)
    if not np.isfinite(fractions).all() or fractions[0] <= 0 or np.any(fractions[1:] < 0):
        raise ValueError("Split fractions must be finite, nonnegative, and sum to less than one")
    accepted = [row for row in records if row.get("quality_flag") == "accepted"]
    if not accepted:
        raise ValueError("No accepted samples are available to split")
    for row in accepted:
        if not str(row.get(group_column, "")).strip():
            raise ValueError(f"Sample {row.get('sample_id')} is missing {group_column}; fill verified provenance first")
        if not str(row.get("activity", "")).strip():
            raise ValueError(f"Accepted sample {row.get('sample_id')} has no activity")
    groups = sorted({row[group_column] for row in accepted})
    labels = sorted({row["activity"] for row in accepted})
    group_index = {name: index for index, name in enumerate(groups)}
    label_index = {name: index for index, name in enumerate(labels)}
    counts = np.zeros((len(groups), len(labels)), dtype=float)
    for row in accepted:
        counts[group_index[row[group_column]], label_index[row["activity"]]] += 1
    total = counts.sum(axis=0)
    holdouts = [index for index in (1, 2) if fractions[index] > 0]
    if len(groups) < 1 + len(holdouts):
        raise ValueError(f"Need at least {1 + len(holdouts)} distinct {group_column} groups for the requested splits")
    targets = fractions[:, None] * total

    def cost(allocation):
        class_error = np.mean(((allocation - targets) / total) ** 2)
        size_error = np.sum(((allocation.sum(axis=1) / total.sum()) - fractions) ** 2)
        return float(class_error + size_error)

    allocation = np.zeros((3, len(labels)))
    allocation[0] = total
    assignments = np.zeros(len(groups), dtype=int)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    if holdouts:
        # Reserving one group per requested holdout first also detects impossible
        # train-class coverage. There are at most two holdout groups to reserve.
        best = None
        best_cost = np.inf
        for candidates in permutations(order, len(holdouts)):
            remaining = total - counts[list(candidates)].sum(axis=0)
            if np.any(remaining <= 0):
                continue
            trial = np.zeros_like(allocation)
            trial[0] = remaining
            for destination, candidate in zip(holdouts, candidates):
                trial[destination] = counts[candidate]
            score = cost(trial)
            if score < best_cost:
                best, best_cost = (candidates, trial), score
        if best is None:
            raise ValueError("Cannot create nonempty holdouts while keeping every activity in training. Collect independent groups or reduce the requested holdout splits.")
        candidates, allocation = best
        for destination, candidate in zip(holdouts, candidates):
            assignments[candidate] = destination

        while True:
            best_move = None
            best_cost = cost(allocation)
            for candidate in order:
                if assignments[candidate] or np.any(allocation[0] - counts[candidate] <= 0):
                    continue
                for destination in holdouts:
                    trial = allocation.copy()
                    trial[0] -= counts[candidate]
                    trial[destination] += counts[candidate]
                    score = cost(trial)
                    if score < best_cost - 1e-12:
                        best_move, best_cost = (candidate, destination, trial), score
            if best_move is None:
                break
            candidate, destination, allocation = best_move
            assignments[candidate] = destination

    split_names = ("train", "validation", "test")
    for index in holdouts:
        missing = [label for label, n in zip(labels, allocation[index]) if n == 0]
        if missing:
            warnings.warn(f"{split_names[index]} is missing activities {missing}; group constraints limit coverage", UserWarning, stacklevel=2)
    result = []
    for record in records:
        row = dict(record)
        row["split"] = split_names[assignments[group_index[row[group_column]]]] if row.get("quality_flag") == "accepted" else ""
        result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--group-column", default="parent_recording_id",
                        choices=("parent_recording_id", "subject_id", "session_id"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    try:
        records = read_manifest(args.manifest)
        result = split_records(records, args.group_column, args.seed, args.val_fraction, args.test_fraction)
        if args.output.resolve() in {Path(row["path"]) for row in result}:
            raise ValueError("Output must not overwrite a source repetition")
        write_manifest(result, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    for split, count in sorted(Counter(row["split"] or "rejected" for row in result).items()):
        groups = {row[args.group_column] for row in result if row["split"] == split}
        print(f"{split}: {count} samples, {len(groups)} groups")
    print(f"Split manifest: {args.output.resolve()}")


if __name__ == "__main__":
    main()
