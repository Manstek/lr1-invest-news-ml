from django.conf import settings
from django.db import models
from django.utils import timezone


class Sentiment(models.TextChoices):
    NEGATIVE = "negative", "Негативная"
    NEUTRAL = "neutral", "Нейтральная"
    POSITIVE = "positive", "Позитивная"


class Security(models.Model):
    """Ценная бумага (эмитент), по которой отслеживаются новости."""

    ticker = models.CharField("Тикер", max_length=12, unique=True)
    name = models.CharField("Эмитент", max_length=120)
    sector = models.CharField("Отрасль", max_length=80, blank=True)
    in_portfolio = models.BooleanField("В портфеле", default=True)

    class Meta:
        verbose_name = "Ценная бумага"
        verbose_name_plural = "Ценные бумаги"
        ordering = ["ticker"]

    def __str__(self):
        return f"{self.ticker} — {self.name}"


class News(models.Model):
    """Новость — объект ИС, который классифицирует ML-сервис."""

    class Status(models.TextChoices):
        NEW = "new", "Новая"
        IN_REVIEW = "in_review", "На анализе"
        DONE = "done", "Обработана"
        ARCHIVED = "archived", "В архиве"

    headline = models.TextField("Текст новости")
    security = models.ForeignKey(
        Security, verbose_name="Ценная бумага", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="news",
    )
    source = models.CharField("Источник", max_length=120, blank=True)
    published_at = models.DateTimeField("Дата публикации", default=timezone.now)
    status = models.CharField("Статус", max_length=12, choices=Status.choices, default=Status.NEW)
    analyst = models.ForeignKey(
        settings.AUTH_USER_MODEL, verbose_name="Аналитик", null=True, blank=True,
        on_delete=models.SET_NULL, related_name="assigned_news",
    )

    # результат интеллектуальной классификации (заполняет ML-сервис)
    ml_sentiment = models.CharField("Тональность (ML)", max_length=10, choices=Sentiment.choices, null=True, blank=True)
    ml_confidence = models.FloatField("Уверенность модели", null=True, blank=True)
    ml_probabilities = models.JSONField("Вероятности классов", null=True, blank=True)
    ml_model_version = models.CharField("Версия модели", max_length=60, blank=True)
    ml_classified_at = models.DateTimeField("Классифицирована", null=True, blank=True)

    # решение аналитика имеет приоритет над моделью
    analyst_sentiment = models.CharField(
        "Тональность (аналитик)", max_length=10, choices=Sentiment.choices, null=True, blank=True
    )
    is_alert = models.BooleanField("Требует внимания", default=False)

    created_at = models.DateTimeField("Создана", auto_now_add=True)
    updated_at = models.DateTimeField("Изменена", auto_now=True)

    class Meta:
        verbose_name = "Новость"
        verbose_name_plural = "Новости"
        ordering = ["-is_alert", "-published_at"]

    def __str__(self):
        return self.headline[:80]

    @property
    def final_sentiment(self):
        """Итоговая тональность: исправление аналитика или ответ модели."""
        return self.analyst_sentiment or self.ml_sentiment

    def get_final_sentiment_display(self):
        value = self.final_sentiment
        return Sentiment(value).label if value else "не определена"

    @property
    def is_classified(self):
        return self.ml_sentiment is not None
