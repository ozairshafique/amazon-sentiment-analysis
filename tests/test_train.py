from src.train import load_data, build_pipelines
import pytest
import pandas as pd

def test_build_piplines():
    piplines = build_pipelines()
    assert isinstance(piplines, dict)
    assert "logreg" in piplines
    assert hasattr(piplines["logreg"], "predict")
    assert hasattr(piplines["logreg"], "predict_proba")

def test_load_data_missing_columns(tmp_path):
    # Create a CSV with missing 'reviewText' column
    df = pd.DataFrame({
        "other_column": [1, 2, 3, 4]
    })

    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    with pytest.raises(ValueError):
        load_data(file_path)

def test_load_data_valid(tmp_path):
    df = pd.DataFrame({
        "reviewText" : ["Great product!", "Terrible experience.", "Okay, not bad.", "Loved it!"],
        "overall": [5, 1, 2, 5]
    })

    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    X, y = load_data(file_path)
    assert len(X) == 4
    assert set(y.unique()).issubset({0,1})