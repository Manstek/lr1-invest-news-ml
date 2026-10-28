"""HTTP-клиент ML-сервиса. Вся сетевая работа с интеллектуальным компонентом — только здесь."""
import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

from django.conf import settings

log = logging.getLogger(__name__)


class MLServiceError(Exception):
    """ML-сервис недоступен или вернул ошибку."""


@dataclass
class Prediction:
    label: str
    confidence: float
    probabilities: dict
    model_version: str


def _request(method: str, path: str, payload: dict | None = None) -> dict:
    url = settings.ML_SERVICE_URL.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=settings.ML_SERVICE_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
        log.warning("ML-сервис недоступен: %s %s -> %s", method, url, exc)
        raise MLServiceError(str(exc)) from exc


def predict(text: str) -> Prediction:
    """POST /predict — тональность текста новости."""
    data = _request("POST", "/predict", {"text": text})
    return Prediction(
        label=data["label"],
        confidence=data["confidence"],
        probabilities=data["probabilities"],
        model_version=data["model_version"],
    )


def health() -> dict | None:
    """GET /health — статус сервиса (None, если сервис недоступен)."""
    try:
        return _request("GET", "/health")
    except MLServiceError:
        return None
