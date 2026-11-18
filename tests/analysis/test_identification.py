import numpy as np
import pytest

from eeg_adhd_epilepsy.analysis.identification import check


def test_identification_separates_child_memory_from_diagnosis():
    rng = np.random.default_rng(0)
    n_children, windows, dims = 40, 10, 16
    child = np.repeat(np.arange(n_children), windows)
    y = np.repeat(np.arange(n_children) % 2, windows)
    train = child < 30
    noise = rng.normal(size=(len(child), dims))
    # each child has its own offset, unrelated to the label: known children, no diagnosis
    fingerprint = np.repeat(rng.normal(scale=3, size=(n_children, dims)), windows, axis=0)
    memory = check(fingerprint + noise, child, y, train)
    assert memory["identification_accuracy"] > 0.9
    assert memory["identification_chance"] == pytest.approx(1 / 30)
    assert abs(memory["diagnosis_auc"] - 0.5) < 0.3
    assert (memory["n_train_children"], memory["n_test_children"]) == (30, 10)
    # a label offset and no child offset: diagnosis, children not told apart
    label = np.outer(y, np.ones(dims)) * 2
    diagnosis = check(label + noise, child, y, train)
    assert diagnosis["diagnosis_auc"] == pytest.approx(1.0)
    assert diagnosis["identification_accuracy"] < 0.2
