"""Признаки для классификации тональности финансовых новостей.

Модуль общий для обучения (ml/train.py) и ML-сервиса (ml_service/main.py):
сохранённая модель ссылается на эти классы, поэтому при загрузке модели
модуль должен быть доступен для импорта.
"""
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

_spacy_cache = {}


def normalize_text(text: str) -> str:
    """Простая нормализация: нижний регистр, числа -> токен <num>, лишние пробелы."""
    text = str(text).lower()
    text = re.sub(r"\d+([.,]\d+)?", " <num> ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _load_spacy(model_name: str):
    if model_name not in _spacy_cache:
        import spacy

        # для эмбеддингов нужны только векторы слов -> отключаем парсер, NER и т.д.
        _spacy_cache[model_name] = spacy.load(
            model_name, exclude=["tagger", "parser", "ner", "lemmatizer", "attribute_ruler", "senter"]
        )
    return _spacy_cache[model_name]


class SpacyMeanEmbedding(BaseEstimator, TransformerMixin):
    """Эмбеддинг текста = среднее предобученных векторов слов (GloVe, 300 измерений)
    из модели spaCy en_core_web_md. Пропускаются только знаки препинания: стоп-слова
    (up, down, not, ...) важны для тональности, их удаление ухудшало macro-F1."""

    def __init__(self, model_name: str = "en_core_web_md"):
        self.model_name = model_name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        nlp = _load_spacy(self.model_name)
        dim = nlp.vocab.vectors_length
        out = np.zeros((len(X), dim), dtype=np.float32)
        for i, doc in enumerate(nlp.pipe([str(t) for t in X], batch_size=256)):
            vecs = [t.vector for t in doc if t.has_vector and not t.is_punct]
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out
