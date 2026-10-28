"""Прикладная логика ИС, использующая результат интеллектуальной классификации."""
from django.conf import settings
from django.utils import timezone

from . import ml_client
from .models import News, Sentiment


def classify_news(news: News) -> bool:
    """Отправляет текст новости в ML-сервис и сохраняет результат в БД.

    Возвращает False, если сервис недоступен: новость остаётся в БД
    без тональности и может быть классифицирована позже.
    """
    try:
        pred = ml_client.predict(news.headline)
    except ml_client.MLServiceError:
        return False

    news.ml_sentiment = pred.label
    news.ml_confidence = pred.confidence
    news.ml_probabilities = pred.probabilities
    news.ml_model_version = pred.model_version
    news.ml_classified_at = timezone.now()
    apply_business_rules(news)
    news.save()
    return True


def apply_business_rules(news: News) -> None:
    """Правило: уверенный негатив по бумаге из портфеля -> «Требует внимания»
    и перевод новой новости в статус «На анализе»."""
    is_negative = news.final_sentiment == Sentiment.NEGATIVE
    in_portfolio = news.security is not None and news.security.in_portfolio
    # решение аналитика считается уверенным, ответ модели — от порога
    confident = bool(news.analyst_sentiment) or (
        (news.ml_confidence or 0) >= settings.ML_ALERT_THRESHOLD)
    news.is_alert = is_negative and in_portfolio and confident
    if news.is_alert and news.status == News.Status.NEW:
        news.status = News.Status.IN_REVIEW


def classify_pending() -> tuple[int, int]:
    """Классифицирует все новости, которые ещё не обработаны моделью."""
    done = failed = 0
    for news in News.objects.filter(ml_sentiment__isnull=True):
        if classify_news(news):
            done += 1
        else:
            failed += 1
            break  # сервис недоступен — нет смысла продолжать
    return done, failed
