import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from eeg_adhd_epilepsy.analysis.child_level import (
    CONDITIONS,
    attention_scores,
    child_blocks,
    linear_scores,
    run,
)


def _windows(n_children=60, n_windows=12, d=8, seed=0):
    """Children whose label shifts the first feature of their EC windows only."""
    rng = np.random.default_rng(seed)
    y = pd.Series(np.arange(n_children) % 2, index=[f"c{i:02d}" for i in range(n_children)])
    X, child, condition = [], [], []
    for name, label in y.items():
        for cond in CONDITIONS:
            x = rng.normal(size=(n_windows, d))
            if cond == CONDITIONS[1]:
                x[:, 0] += 1.5 * label
            X.append(x)
            child += [name] * n_windows
            condition += [cond] * n_windows
    return np.vstack(X).astype(np.float32), np.array(child), np.array(condition), y


def test_child_blocks_are_condition_means_and_their_difference():
    X = np.array([[1.0], [3.0], [10.0], [0.0]])
    blocks = child_blocks(X, ["a", "a", "a", "b"], [*CONDITIONS[:1] * 2, CONDITIONS[1], "EC"], "m")
    assert list(blocks) == ["m_EO", "m_EC", "m_EC-EO"]
    assert list(blocks["m_EO"].index) == ["a"]  # b has no EO window
    assert blocks["m_EO"].iloc[0, 0] == 2.0 and blocks["m_EC-EO"].iloc[0, 0] == 8.0


def test_linear_and_attention_scores_find_a_child_level_signal():
    X, child, condition, y = _windows()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(y, y))
    blocks = child_blocks(X, child, condition, "m")
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, linear_scores(blocks, y, folds, n_components=None)) > 0.8
    scores, weights = attention_scores(
        X, child, condition, y, folds, seeds=(0,), n_components=4, max_epochs=30, patience=10
    )
    assert scores.notna().all() and roc_auc_score(y, scores) > 0.8
    # one weight per test window, summing to one per child
    assert len(weights) == len(X)
    assert np.allclose(weights.groupby("child")["attention"].sum(), 1.0, atol=1e-5)


def test_run_scores_every_method_on_the_same_children():
    X, child, condition, y = _windows(n_children=40)
    windows = {"m": (X, child, condition)}
    blocks = child_blocks(X, child, condition, "m")
    descriptors = {"summary": child_blocks(X[:, :3], child, condition, "desc")}
    table, summary, attention = run(y, windows, blocks, descriptors, n_permutations=9, seeds=(0,))
    assert {"mean_m", "attention_m", "descriptors_summary", "early_fusion_summary"} <= set(table)
    assert {"late_fusion_embeddings", "late_fusion_summary"} <= set(summary)
    assert table.drop(columns="y").notna().all().all() and len(table) == 40
    assert summary["mean_m"]["n_children"] == 40 and "m" in attention
