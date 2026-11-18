from types import SimpleNamespace

import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.analysis.continued_pretraining import child_folds, combine


def test_child_folds_cover_each_child_once_in_any_order():
    labels = pd.Series(np.arange(20) % 2, index=[f"c{i:02d}" for i in range(20)])
    folds = child_folds(labels)
    tests = np.concatenate([test for _, test in folds])
    assert sorted(tests) == list(range(20))
    shuffled = child_folds(labels.sample(frac=1, random_state=1))
    assert all((a[1] == b[1]).all() for a, b in zip(folds, shuffled))


def test_combine_scores_original_and_continued_per_model(tmp_path):
    rng = np.random.default_rng(0)
    for fold in range(2):
        out = tmp_path / "cbramod" / f"fold{fold}"
        out.mkdir(parents=True)
        y = np.arange(10) % 2
        pd.DataFrame(
            {
                "child": [f"f{fold}c{i}" for i in range(10)],
                "y": y,
                "original": rng.random(10),
                "continued": y + rng.random(10),
            }
        ).to_csv(out / "scores.csv", index=False)
    args = SimpleNamespace(output_dir=tmp_path, n_folds=2, n_permutations=9)
    summary = combine(args)
    assert summary["cbramod_continued"]["auc"] == 1.0
    assert summary["cbramod_original"]["n_children"] == 20
