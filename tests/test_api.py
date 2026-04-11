from api.main import app
from fastapi.testclient import TestClient
import pytest

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()['status'] == "ok"

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "version" in response.json()
    assert "models" in response.json()

def test_predict_postive():
    response = client.post("/predict", json={
        "text" : "This product is amazing! I loved it.",
        "model_name": "logreg"
    })
    data = response.json()
    assert response.status_code == 200
    assert data['sentiment'] == "positive"
    assert 0.0 <= data['confidence'] <= 1.0

def test_predict_negative():
    response = client.post("/predict", json= {
        "text": "Worst product ever. Completely broken, total waste of money. Do not buy. Very disappointed",
        "model_name":"logreg"
    })
    data = response.json()
    assert response.status_code == 200
    assert data['sentiment'] == "negative"
    assert "confidence" in data


def test_predict_empty_text():
    response = client.post("/predict", json={
        "text": "",
        "model_name": "logreg"
    })
    assert response.status_code == 422

def test_invalid_model():
    response = client.post("/predict", json= {
        "text": "This product is amazing! I like it.",
        "model_name": "invalid_model"
    })

    assert response.status_code == 422

def test_predict_batch():
    response = client.post("/predict/batch", json= [
            {"text": "This product is amazing! I like it.", "model_name": "logreg"},
            {"text": "Worst product ever. Completely broken, total waste of money. Do not buy. Very disappointed", "model_name": "logreg"}
        ])

    data = response.json()
    assert response.status_code == 200
    assert data['total'] >= 2
    assert len(data['results']) == 2