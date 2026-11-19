import numpy as np
import pandas as pd
import pytest

from eeg_adhd_epilepsy.analysis.ensemble import child_scores, ensemble, evaluate


def _predictions(scores, labels, rows_per_child=3):
    """Epoch rows of a fit: each child's rows share its label and probability."""
    children = np.repeat(np.arange(len(labels)), rows_per_child)
    proba = np.repeat(scores, rows_per_child)
    return pd.DataFrame(
        {
            "group_id": children,
            "y_true": np.repeat(labels, rows_per_child),
            "y_proba_0": 1 - proba,
            "y_proba_1": proba,
        }
    )


def test_ensemble_scores_children_on_mean_ranks():
    labels = np.array([0, 0, 0, 1, 1, 1])
    # two fits on different scales; the second ranks child 2 above child 3
    a = child_scores(_predictions(np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8]), labels))
    b = child_scores(_predictions(np.array([0.51, 0.52, 0.56, 0.55, 0.57, 0.58]), labels))
    assert a.shape == (6, 3) and a["score"].tolist() == pytest.approx(
        [0.1, 0.2, 0.3, 0.6, 0.7, 0.8]
    )
    children = ensemble({"a": a, "b": b})
    assert children["n_fits"].tolist() == [2] * 6
    assert children["score"].tolist() == pytest.approx([1 / 6, 2 / 6, 3.5 / 6, 3.5 / 6, 5 / 6, 1])
    result = evaluate(children, n_permutations=200, n_bootstrap=50)
    assert result["auc"] == pytest.approx(17 / 18)  # one tied pair counts half
    assert result["balanced_accuracy_median_split"] == pytest.approx(5 / 6)
    assert result["auc_p"] < 0.1 and result["n_children"] == 6
    with pytest.raises(ValueError, match="two labels"):
        ensemble({"a": a, "c": a.assign(y=1 - a["y"])})
