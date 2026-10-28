"""ML-сервис классификации тональности финансовых новостей.

Отдельный компонент ИС: загружает обученную модель и отдаёт предсказания по HTTP (REST/JSON).
Запуск из корня проекта:  uvicorn ml_service.main:app --port 8001
Документация API (Swagger): http://127.0.0.1:8001/docs
"""
import json
import logging
import os
import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "ml" / "models"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ml_service")

model = joblib.load(MODEL_DIR / "sentiment_model.joblib")
meta = json.loads((MODEL_DIR / "model_meta.json").read_text(encoding="utf-8"))
LABELS_RU = meta["labels_ru"]
log.info("Модель загружена: %s", meta["model_version"])

app = FastAPI(
    title="ML-сервис тональности финансовых новостей",
    description="Классифицирует заголовок новости: negative / neutral / positive.",
    version="1.0",
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=3, max_length=2000, examples=[
        "Company X raises its full-year profit forecast after strong quarterly sales"
    ])


class PredictResponse(BaseModel):
    label: str = Field(..., description="Класс: negative | neutral | positive")
    label_ru: str
    confidence: float = Field(..., description="Вероятность предсказанного класса")
    probabilities: dict[str, float]
    model_version: str
    elapsed_ms: float


@app.get("/health")
def health():
    """Проверка доступности сервиса и версии модели."""
    return {"status": "ok", "model_version": meta["model_version"], "model": meta["model_title"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Определяет тональность текста новости."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Пустой текст")
    t0 = time.perf_counter()
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    best = int(proba.argmax())
    label = classes[best]
    elapsed = (time.perf_counter() - t0) * 1000
    log.info("predict label=%s conf=%.3f len=%d", label, proba[best], len(text))
    return PredictResponse(
        label=label,
        label_ru=LABELS_RU[label],
        confidence=round(float(proba[best]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(classes, proba)},
        model_version=meta["model_version"],
        elapsed_ms=round(elapsed, 2),
    )
