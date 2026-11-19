#!/usr/bin/env python3
"""Does a representation know the child better than the diagnosis?

For window vectors with their child and label, split into training and test children:

- **identification**: each training child's windows are split in two halves (alternate
  windows); a nearest-centroid classifier (cosine, on standardised vectors) learns the children
  from one half and names the child of every window of the other half. Chance is one over the
  number of children.
- **diagnosis**: a logistic regression on the training children's mean vectors predicts the
  label of each test child's mean vector, scored with the AUC.

High identification with chance diagnosis means the representation memorises children. Run on
the frozen embeddings and on the fine-tuned features of the same fold, it shows what
fine-tuning changes. The fine-tuned features are the ``*_features.npz`` that a stage 17 run
with ``save_features: true`` writes per fold (``features``, ``sample_id``, ``y``, ``group``,
``split`` in train / valid / test).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, normalize


def identification_accuracy(X: np.ndarray, child: np.ndarray) -> tuple[float, float]:
    """Top-1 child identification of held-out windows, and its chance level.

    Children with fewer than two windows are left out.
    """
    child = np.asarray(child)
    order = np.argsort(child, kind="stable")
    X, child = np.asarray(X, dtype=float)[order], child[order]
    rank = np.concatenate([np.arange(n) for n in np.unique(child, return_counts=True)[1]])
    names, counts = np.unique(child, return_counts=True)
    keep = np.isin(child, names[counts >= 2])
    X, child, rank = X[keep], child[keep], rank[keep]
    fit, held = rank % 2 == 0, rank % 2 == 1
    scaler = StandardScaler().fit(X[fit])
    Z = normalize(scaler.transform(X))
    names = np.unique(child)
    centroids = normalize(np.stack([Z[fit & (child == name)].mean(axis=0) for name in names]))
    predicted = names[(Z[held] @ centroids.T).argmax(axis=1)]
    return float((predicted == child[held]).mean()), 1.0 / len(names)


def diagnosis_auc(X: np.ndarray, child: np.ndarray, y: np.ndarray, train: np.ndarray) -> float:
    """AUC of a logistic regression from training to test children's mean vectors."""
    X, child, y, train = map(np.asarray, (X, child, y, train))

    def means(rows):
        names = np.unique(child[rows])
        return (
            np.stack([X[rows & (child == name)].mean(axis=0) for name in names]),
            np.array([y[rows & (child == name)][0] for name in names]),
        )

    X_train, y_train = means(train)
    X_test, y_test = means(~train)
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
        scaler.transform(X_train), y_train
    )
    return float(roc_auc_score(y_test, model.predict_proba(scaler.transform(X_test))[:, 1]))


def check(X: np.ndarray, child: np.ndarray, y: np.ndarray, train: np.ndarray) -> dict:
    """Identification (training children) and diagnosis (training -> test children)."""
    train = np.asarray(train, dtype=bool)
    accuracy, chance = identification_accuracy(np.asarray(X)[train], np.asarray(child)[train])
    return {
        "identification_accuracy": accuracy,
        "identification_chance": chance,
        "diagnosis_auc": diagnosis_auc(X, child, y, train),
        "n_train_children": int(np.unique(np.asarray(child)[train]).size),
        "n_test_children": int(np.unique(np.asarray(child)[~train]).size),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "features", nargs="+", type=Path, help="*_features.npz files (one per fold)."
    )
    parser.add_argument("--output", type=Path, help="JSON with one entry per file.")
    args = parser.parse_args()
    results = {}
    for path in args.features:
        data = np.load(path, allow_pickle=True)
        split = data["split"].astype(str)
        # validation children chose the stopping epoch, so they are neither
        # training nor held-out children
        keep = split != "valid"
        results[str(path)] = check(
            data["features"][keep], data["group"][keep], data["y"][keep], split[keep] == "train"
        )
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
