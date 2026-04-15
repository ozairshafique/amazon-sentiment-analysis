from src.train import load_data, build_pipelines
import src.train as train_module
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

def test_load_data_invalid_sentiment(tmp_path):
    df = pd.DataFrame({
        "reviewText" : ["Great product!", "Terrible experience.", "Okay, not bad.", "Loved it!"],
        "overall": [5, 1, 3, 2]  # 3 is neutral and should be dropped
    })

    file_path = tmp_path / "tests_data.csv"
    df.to_csv(file_path, index=False)

    X, y = load_data(file_path)
    assert len(X) == 3  # One neutral row should be dropped
    assert set(y.unique()).issubset({0, 1})

def test_test_train_end_to_end(tmp_path, monkeypatch):
    df = pd.DataFrame({
        'reviewText' : ["Great product!", "Terrible experience.", "Okay, not bad.", "Loved it!", "Worst ever!", "Average product."],
        'overall': [5, 4, 4, 5, 2, 2]
    })
    file_path = tmp_path / "test_data.csv"
    df.to_csv(file_path, index=False)

    monkeypatch.setattr(train_module, "DATA_PATH", file_path)
    monkeypatch.setattr(train_module, "MODEL_DIR", tmp_path)

    train_module.train()
    assert(tmp_path/"logreg_model.pkl").exists()
    assert(tmp_path/"linear_svc_model.pkl").exists()
