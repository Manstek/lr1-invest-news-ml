"""Обучение и оценка моделей классификации тональности финансовых новостей.

Запуск из корня проекта:  python -m ml.train
Результат: ml/models/sentiment_model.joblib, ml/models/model_meta.json,
           ml/reports/*.json|png
"""
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

from ml.features import SpacyMeanEmbedding, normalize_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "all-data.csv"
DEMO = ROOT / "data" / "demo_news.csv"
MODELS = ROOT / "ml" / "models"
REPORTS = ROOT / "ml" / "reports"
LABELS = ["negative", "neutral", "positive"]
LABELS_RU = {"negative": "негативная", "neutral": "нейтральная", "positive": "позитивная"}
SEED = 42

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def load_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(DATA, encoding="latin-1", header=None, names=["sentiment", "text"])
    stats = {"rows_raw": len(df)}
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    # одинаковый текст с разными метками -> противоречие, удаляем такие тексты целиком
    conflict = df.groupby("text")["sentiment"].nunique()
    conflict_texts = set(conflict[conflict > 1].index)
    stats["conflicting_texts"] = len(conflict_texts)
    df = df[~df["text"].isin(conflict_texts)]
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    stats["duplicates_removed"] = before - len(df)
    # новости, отложенные для демонстрации работы ИС, в обучении и оценке не участвуют
    demo = pd.read_csv(DEMO, encoding="utf-8")
    df = df[~df["text"].isin(set(demo["text"]))].reset_index(drop=True)
    stats["demo_excluded"] = len(demo)
    stats["rows_clean"] = len(df)
    stats["class_counts"] = df["sentiment"].value_counts().reindex(LABELS).to_dict()
    words = df["text"].str.split().str.len()
    stats["words_mean"] = round(float(words.mean()), 1)
    stats["words_median"] = int(words.median())
    stats["words_max"] = int(words.max())
    return df, stats


def tfidf_features():
    return TfidfVectorizer(
        preprocessor=normalize_text, ngram_range=(1, 2), min_df=2, sublinear_tf=True
    )


def build_candidates():
    lr = dict(class_weight="balanced", max_iter=3000)
    return {
        "baseline": (
            Pipeline([("clf", DummyClassifier(strategy="most_frequent"))]),
            {},
        ),
        "tfidf_lr": (
            Pipeline([("tfidf", tfidf_features()), ("clf", LogisticRegression(**lr))]),
            {"clf__C": [1, 3, 10, 30]},
        ),
        "emb_lr": (
            Pipeline(
                [
                    ("emb", SpacyMeanEmbedding()),
                    ("scale", StandardScaler()),
                    ("clf", LogisticRegression(**lr)),
                ]
            ),
            {"clf__C": [0.03, 0.1, 0.3, 1]},
        ),
        "tfidf_emb_lr": (
            Pipeline(
                [
                    (
                        "features",
                        FeatureUnion(
                            [
                                ("tfidf", tfidf_features()),
                                (
                                    "emb",
                                    Pipeline([("vec", SpacyMeanEmbedding()), ("scale", StandardScaler())]),
                                ),
                            ]
                        ),
                    ),
                    ("clf", LogisticRegression(**lr)),
                ]
            ),
            {"clf__C": [0.03, 0.1, 0.3, 1]},
        ),
    }


MODEL_TITLES = {
    "baseline": "Базовая (самый частый класс)",
    "tfidf_lr": "TF-IDF + логистическая регрессия",
    "emb_lr": "Эмбеддинги GloVe + логистическая регрессия",
    "tfidf_emb_lr": "TF-IDF + эмбеддинги + логистическая регрессия",
}


def plot_class_distribution(stats):
    counts = [stats["class_counts"][l] for l in LABELS]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    bars = ax.bar([LABELS_RU[l] for l in LABELS], counts, color=["#c0392b", "#7f8c8d", "#27ae60"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 30, str(c), ha="center")
    ax.set_ylabel("Количество новостей")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORTS / "class_distribution.png", dpi=200)
    plt.close(fig)


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    draw_confusion(cm, path)
    return cm.tolist()


def draw_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ConfusionMatrixDisplay(np.asarray(cm), display_labels=[LABELS_RU[l] for l in LABELS]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_xlabel("Предсказанный класс", labelpad=8)
    ax.set_ylabel("Истинный класс", labelpad=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    df, stats = load_dataset()
    print("Данные:", json.dumps(stats, ensure_ascii=False))
    plot_class_distribution(stats)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["sentiment"], test_size=0.2, stratify=df["sentiment"], random_state=SEED
    )
    stats["train_size"], stats["test_size"] = len(X_train), len(X_test)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, (pipe, grid) in build_candidates().items():
        t0 = time.time()
        if grid:
            search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=1)
            search.fit(X_train, y_train)
            model, best_params, cv_f1 = search.best_estimator_, search.best_params_, search.best_score_
        else:
            model, best_params, cv_f1 = pipe.fit(X_train, y_train), {}, None
        fit_time = time.time() - t0
        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_ms = (time.time() - t0) / len(X_test) * 1000
        rep = classification_report(y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0)
        results[name] = {
            "title": MODEL_TITLES[name],
            "best_params": best_params,
            "cv_f1_macro": round(cv_f1, 4) if cv_f1 is not None else None,
            "test_accuracy": round(accuracy_score(y_test, y_pred), 4),
            "test_f1_macro": round(f1_score(y_test, y_pred, average="macro"), 4),
            "test_f1_weighted": round(f1_score(y_test, y_pred, average="weighted"), 4),
            "per_class": {
                l: {k: round(rep[l][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                for l in LABELS
            },
            "fit_time_s": round(fit_time, 1),
            "predict_ms_per_text": round(pred_ms, 3),
        }
        fitted[name] = (model, y_pred)
        print(name, json.dumps(results[name], ensure_ascii=False))

    # выбор модели: лучшая по macro-F1 на кросс-валидации (тестовая выборка только для итоговой оценки)
    best = max((n for n in results if results[n]["cv_f1_macro"] is not None),
               key=lambda n: results[n]["cv_f1_macro"])
    best_model, best_pred = fitted[best]
    results[best]["confusion_matrix"] = plot_confusion(y_test, best_pred, REPORTS / "confusion_matrix.png")
    for n in results:
        if n != best and n != "baseline":
            results[n]["confusion_matrix"] = plot_confusion(
                y_test, fitted[n][1], REPORTS / f"confusion_matrix_{n}.png"
            )

    # примеры ошибок лучшей модели — для анализа в отчёте
    errors = pd.DataFrame({"text": X_test, "true": y_test, "pred": best_pred})
    errors = errors[errors["true"] != errors["pred"]].head(15)
    errors.to_csv(REPORTS / "errors_sample.csv", index=False)

    # финальная модель дообучается на всех данных с подобранными гиперпараметрами
    final_model = build_candidates()[best][0].set_params(**results[best]["best_params"])
    final_model.fit(df["text"], df["sentiment"])
    version = f"{best}-{datetime.now():%Y%m%d}"
    joblib.dump(final_model, MODELS / "sentiment_model.joblib")
    meta = {
        "model_version": version,
        "model_name": best,
        "model_title": MODEL_TITLES[best],
        "labels": LABELS,
        "labels_ru": LABELS_RU,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "train_rows": len(df),
        "test_metrics": {k: results[best][k] for k in ("test_accuracy", "test_f1_macro", "test_f1_weighted")},
        "dataset": "Sentiment Analysis for Financial News (FinancialPhraseBank), Kaggle",
    }
    (MODELS / "model_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS / "metrics.json").write_text(
        json.dumps({"dataset": stats, "best": best, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("Лучшая модель:", best, "| версия", version)


if __name__ == "__main__":
    main()
