from src.model_loader import predict
import pytest

def test_predict():
    result = predict("This producgt is amazing!")

    assert result['sentiment'] in ['positive', 'negative']
    assert 0.0 <= result['confidence'] <= 1.0

def test_predict_empty():
    with pytest.raises(ValueError):
        predict("")

