from src.model_loader import predict, _load, AVAILABLE_MODELS
import pytest
from src import model_loader


def test_predict():
    result = predict("This producgt is amazing!")

    assert result['sentiment'] in ['positive', 'negative']
    assert 0.0 <= result['confidence'] <= 1.0

def test_predict_empty():
    with pytest.raises(ValueError):
        predict("")

def test_model_not_found():
    model_name = "svm"
    assert model_name not in AVAILABLE_MODELS
    with pytest.raises(ValueError):
        _load(model_name)

def test_load_model():
    model = _load("logreg")
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")

def test_load_model_file_not_found(tmp_path, monkeypatch):

    model_loader._load.cache_clear()  # Clear lru_cache to ensure _load is called again
    monkeypatch.setattr(model_loader, "MODEL_DIR", tmp_path)
    excepted_path = tmp_path / "logreg_model.pkl"

    if excepted_path.exists():
        excepted_path.unlink()

    with pytest.raises(FileNotFoundError):
        model_loader._load("logreg")