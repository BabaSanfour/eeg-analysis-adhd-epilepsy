import numpy as np
import pandas as pd

from eeg_adhd_epilepsy.analysis.medication_contrast import COVARIATES, matched_draw, run


def _children(seed=0):
    """Controls, unmedicated ADHD shifted far from them, medicated ADHD shifted a little."""
    rng = np.random.default_rng(seed)
    groups = [("control", 200, 0.0), ("unmedicated", 40, 1.5)]
    groups += [("Methylphenidate", 110, 0.3), ("Lisdexamfetamine", 40, 0.3)]
    rows, X = [], []
    for name, n, shift in groups:
        for _ in range(n):
            rows.append(
                {
                    "adhd": "0" if name == "control" else "1",
                    "psychostimulant": "1"
                    if name in ("Methylphenidate", "Lisdexamfetamine")
                    else "0",
                    "psychostimulant_category": name,
                    "epilepsy": str(rng.integers(2)),
                    "autism": "0",
                    "age_group": rng.choice(["5-8", "9-12"]),
                    "sex": rng.choice(["M", "F"]),
                }
            )
            x = rng.normal(size=6)
            x[0] += shift
            X.append(x)
    meta = pd.DataFrame(rows, index=[f"c{i:03d}" for i in range(len(rows))])
    X = np.array(X)
    blocks = {
        "m_EO": pd.DataFrame(X + rng.normal(scale=0.5, size=X.shape), index=meta.index),
        "m_EC": pd.DataFrame(X + rng.normal(scale=0.5, size=X.shape), index=meta.index),
    }
    return blocks, meta


def test_matched_draw_copies_the_reference_make_up():
    _, meta = _children()
    reference = meta[meta.psychostimulant_category == "unmedicated"]
    pool = meta[meta.psychostimulant == "1"]
    chosen = matched_draw(pool, reference, np.random.default_rng(0))
    assert chosen.is_unique and chosen.isin(pool.index).all()
    pd.testing.assert_series_equal(
        meta.loc[chosen].groupby(list(COVARIATES)).size(),
        reference.groupby(list(COVARIATES)).size(),
    )


def test_run_finds_that_medication_brings_children_towards_controls():
    blocks, meta = _children()
    summary, draws = run(blocks, meta, n_draws=5, n_repeats=1)
    unmedicated = summary["unmedicated_vs_control"]
    assert unmedicated["reference_auc"] > 0.8 > unmedicated["draws_auc_95"][1]
    assert abs(unmedicated["covariates_auc"] - 0.5) < 0.15  # covariates carry nothing here
    assert summary["medicated_vs_unmedicated"]["auc_median"] > 0.6
    assert 0.0 < summary["axis"]["normalisation_index_matched_median"] < 0.5
    # both drugs shift equally: lisdexamfetamine is inside the methylphenidate draws
    drugs = summary["lisdexamfetamine_vs_control"]
    assert drugs["draws_auc_95"][0] - 0.1 < drugs["reference_auc"] < drugs["draws_auc_95"][1] + 0.1
    assert len(draws) == 5 and "normalisation_index" in draws
