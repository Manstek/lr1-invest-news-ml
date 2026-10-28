"""Проверка итоговой модели на 27 демонстрационных новостях, не участвовавших в обучении.
Запуск из корня проекта: python -m ml.evaluate_demo"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["negative", "neutral", "positive"]

model = joblib.load(ROOT / "ml" / "models" / "sentiment_model.joblib")
demo = pd.read_csv(ROOT / "data" / "demo_news.csv")
demo["pred"] = model.predict(demo["text"])
res = {
    "n": len(demo),
    "correct": int((demo["pred"] == demo["true_sentiment"]).sum()),
    "accuracy": round(accuracy_score(demo["true_sentiment"], demo["pred"]), 4),
    "true_counts": demo["true_sentiment"].value_counts().reindex(LABELS).to_dict(),
    "confusion_matrix": confusion_matrix(demo["true_sentiment"], demo["pred"], labels=LABELS).tolist(),
}
(ROOT / "ml" / "reports" / "demo_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(res, ensure_ascii=False))
