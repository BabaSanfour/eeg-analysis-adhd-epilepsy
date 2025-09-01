import pandas as pd
from eeg_psychostimulant.analysis import design


def test_filter_by_sex():
    df = pd.DataFrame({"Sex": ["F", "M", "F"]})
    filtered = design.filter_by_sex(df, "F")
    assert filtered.shape[0] == 2
    assert all(filtered["Sex"] == "F")


def test_filter_diagnosis_without():
    df = pd.DataFrame({"ADHD": ["1", "0", "0 (potentiel)"]})
    filtered = design.filter_diagnosis(df, "ADHD", "without")
    assert filtered["ADHD"].tolist() == ["0"]


def test_build_analysis_dataset(tmp_path):
    demo = pd.DataFrame(
        [
            {
                "Study ID": 1,
                "Sex": "F",
                "Age": 10,
                "ADHD": "1",
                "TSA": "0",
                "Epilepsy": "0",
                "Psychostimulant (y/n)": 1,
            },
            {
                "Study ID": 2,
                "Sex": "M",
                "Age": 13,
                "ADHD": "0",
                "TSA": "1",
                "Epilepsy": "0",
                "Psychostimulant (y/n)": 0,
            },
            {
                "Study ID": 3,
                "Sex": "F",
                "Age": 15,
                "ADHD": "0 (potentiel)",
                "TSA": "0",
                "Epilepsy": "1",
                "Psychostimulant (y/n)": 1,
            },
        ]
    )
    features = pd.DataFrame({"Study ID": [1, 2, 3], "feat": [0.1, 0.2, 0.3]})
    demo_path = tmp_path / "demo.csv"
    feat_path = tmp_path / "feat.csv"
    demo.to_csv(demo_path, index=False)
    features.to_csv(feat_path, index=False)

    filters = {
        "sex": "F",
        "age_groups": "combined",
        "ADHD": "with",
        "TSA": "combined",
        "Epilepsy": "without",
    }
    result = design.build_analysis_dataset(
        str(demo_path), str(feat_path), "general", filters
    )
    assert result["Study ID"].tolist() == [1]
    assert result["target"].tolist() == [1]
