"""Persisted outer test folds with independent inner validation partitions."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)


def validate_splits(folds, dataset, expected_folds):
    dataset.require_labels()
    if len(folds) != expected_folds:
        raise ValueError("Split file has the wrong number of folds")
    universe = set(range(len(dataset)))
    tests = []
    for fold in folds:
        if set(fold) != {"train", "validation", "test"}:
            raise ValueError("Each fold must define train, validation, and test")
        sets = []
        for name in ("train", "validation", "test"):
            indices = fold[name]
            if not indices or any(type(i) is not int for i in indices):
                raise ValueError("Split indices must be nonempty lists of integers")
            chosen = set(indices)
            if len(chosen) != len(indices) or not chosen <= universe:
                raise ValueError("Split contains duplicate or invalid indices")
            if set(dataset.labels[indices]) != {0, 1}:
                raise ValueError(
                    "Every partition must contain both classes; provide more subjects/groups"
                )
            sets.append(chosen)
        a, b, c = sets
        if a & b or a & c or b & c or a | b | c != universe:
            raise ValueError(
                "Train, validation, and test must form disjoint complete partitions"
            )
        if dataset.groups is not None:
            group_sets = [set(dataset.groups[list(indices)]) for indices in sets]
            if any(group_sets[i] & group_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
                raise ValueError("A group appears in more than one partition")
        tests.extend(fold["test"])
    if len(tests) != len(dataset) or set(tests) != universe:
        raise ValueError("Every subject must occur in exactly one outer test fold")
    return folds


def create_splits(dataset, config):
    dataset.require_labels()
    y, groups = dataset.labels, dataset.groups
    if min(np.bincount(y, minlength=2)) < config.folds:
        raise ValueError("Each class needs at least as many subjects as outer folds")
    splitter = (
        StratifiedKFold(config.folds, shuffle=True, random_state=config.seed)
        if groups is None
        else StratifiedGroupKFold(config.folds, shuffle=True, random_state=config.seed)
    )
    folds = []
    for number, (development, test) in enumerate(
        splitter.split(np.zeros(len(y)), y, groups)
    ):
        seed = config.seed + number + 1
        if groups is None:
            train, validation = train_test_split(
                development,
                test_size=config.validation_fraction,
                stratify=y[development],
                random_state=seed,
            )
        else:
            candidates = GroupShuffleSplit(
                128, test_size=config.validation_fraction, random_state=seed
            )
            choices = []
            for local_train, local_val in candidates.split(
                development, y[development], groups[development]
            ):
                train, validation = development[local_train], development[local_val]
                if len(set(y[train])) == len(set(y[validation])) == 2:
                    error = abs(
                        len(validation) / len(development) - config.validation_fraction
                    )
                    error += abs(y[validation].mean() - y[development].mean())
                    choices.append((error, train, validation))
            if not choices:
                raise ValueError(
                    "Cannot form group-disjoint validation data with both classes"
                )
            _, train, validation = min(choices, key=lambda item: item[0])
        folds.append(
            {
                "train": train.tolist(),
                "validation": validation.tolist(),
                "test": test.tolist(),
            }
        )
    return validate_splits(folds, dataset, config.folds)


def load_or_create_splits(dataset, config, output, source=None):
    path = Path(source) if source else Path(output) / "splits.json"
    fingerprint = dataset.fingerprint()
    if path.exists():
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version") != 1
            or payload.get("fingerprint") != fingerprint
        ):
            raise ValueError(
                "Split file does not match input data values/order or schema"
            )
        folds = validate_splits(payload["folds"], dataset, config.folds)
    elif source:
        raise FileNotFoundError(path)
    else:
        folds = create_splits(dataset, config)
        payload = {
            "schema_version": 1,
            "fingerprint": fingerprint,
            "seed": config.seed,
            "subject_ids": dataset.subject_ids.tolist(),
            "folds": folds,
        }
    destination = Path(output) / "splits.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return folds
