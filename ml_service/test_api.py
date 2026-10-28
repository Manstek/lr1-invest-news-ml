"""Тесты API ML-сервиса. Запуск из корня проекта: python -m pytest ml_service"""
from fastapi.testclient import TestClient

from ml_service.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_distribution():
    r = client.post("/predict", json={"text": "Net sales increased by 18 % and operating profit doubled"})
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in {"negative", "neutral", "positive"}
    assert abs(sum(data["probabilities"].values()) - 1) < 1e-3
    assert data["confidence"] == max(data["probabilities"].values())


def test_predict_negative_example():
    r = client.post("/predict", json={"text": "Operating loss widened and the company cut 500 jobs"})
    assert r.json()["label"] == "negative"


def test_predict_validation():
    assert client.post("/predict", json={"text": ""}).status_code == 422
    assert client.post("/predict", json={}).status_code == 422
