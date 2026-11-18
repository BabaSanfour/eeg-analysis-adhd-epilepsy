import numpy as np
import pandas as pd
import yaml

from eeg_adhd_epilepsy.analysis.medication_contrast import (
    COVARIATES,
    _drug_transfer,
    matched_draw,
    run,
    widen_cohort,
)


def _children(seed=0, drug_shift=(0.3, 0.3)):
    """Controls, unmedicated ADHD shifted far from them, medicated ADHD shifted a little
    (methylphenidate, lisdexamfetamine)."""
    rng = np.random.default_rng(seed)
    groups = [("control", 200, 0.0), ("unmedicated", 40, 1.5)]
    groups += [("Methylphenidate", 110, drug_shift[0]), ("Lisdexamfetamine", 40, drug_shift[1])]
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
    assert unmedicated["reference_auc"]["median"] > 0.8 > unmedicated["drawn_auc"]["95"][1]
    assert unmedicated["share_of_draws_reference_higher"] == 1.0
    assert unmedicated["covariates_auc"]["median"] < 0.65  # matched: covariates carry nothing
    assert summary["medicated_vs_unmedicated"]["auc_median"] > 0.6
    assert 0.0 < summary["axis"]["normalisation_index"]["median"] < 0.5
    # unmedicated children lie beyond the medicated ones on the medicated-vs-control axis
    assert summary["transfer"]["adhd_index"]["median"] > 1.0
    assert "methylphenidate_on_lisdexamfetamine_axis_index" in summary["drug_transfer"]
    # both drugs shift equally: no consistent difference between them
    assert abs(summary["lisdexamfetamine_vs_control"]["difference"]["median"]) < 0.1
    assert len(draws) == 5 and "normalisation_index" in draws


def test_widen_cohort_keeps_every_adhd_child_and_the_family_filters(tmp_path):
    source = tmp_path / "family.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "dataset_name": "pooled_02",
                "group_filters": [
                    {"adhd": [0], "psychostimulant": [0], "epilepsy": [0]},
                    {"adhd": [1], "psychostimulant": [1], "epilepsy": [0]},
                ],
            }
        )
    )
    out = yaml.safe_load(widen_cohort(source, tmp_path).read_text())
    assert out["group_filters"] == [
        {"adhd": [0], "psychostimulant": [0], "epilepsy": [0]},
        {"adhd": [1], "epilepsy": [0]},
    ]
    assert out["dataset_name"] == "contrast_pooled_02"


def test_run_skips_contrasts_whose_groups_are_too_small():
    blocks, meta = _children()
    meta = meta.drop(meta.index[meta.psychostimulant_category == "unmedicated"][5:])
    summary, _ = run(blocks, meta, n_draws=2, n_repeats=1)
    assert "skipped" in summary["unmedicated_vs_control"] and "skipped" in summary["axis"]
    assert "skipped" in summary["transfer"]
    assert "difference" in summary["lisdexamfetamine_vs_control"]


def test_drug_transfer_tells_a_shared_signal_from_a_drug_specific_one():
    def indices(drug_shift):
        blocks, meta = _children(drug_shift=drug_shift)
        drug = meta["psychostimulant_category"]
        ids = {name: meta.index[drug == name] for name in drug.unique()}
        summary, _ = _drug_transfer(
            blocks,
            meta,
            ids["Lisdexamfetamine"],
            ids["Methylphenidate"],
            ids["control"],
            n_draws=3,
            n_repeats=1,
            seed=0,
            n_jobs=1,
        )
        return summary["methylphenidate_on_lisdexamfetamine_axis_index"]["median"]

    # the same shift for both drugs transfers; methylphenidate without an effect does not
    assert indices((1.5, 1.5)) > 0.6
    assert indices((0.0, 1.5)) < 0.3
