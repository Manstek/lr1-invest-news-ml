"""Тесты интеграции ИС с ML-сервисом (ML-сервис подменяется заглушкой)."""
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from . import ml_client
from .models import News, Security
from .services import classify_news

NEG = ml_client.Prediction(label="negative", confidence=0.9,
                           probabilities={"negative": 0.9, "neutral": 0.05, "positive": 0.05},
                           model_version="test")


class ClassificationIntegrationTests(TestCase):
    def setUp(self):
        self.sec = Security.objects.create(ticker="TST", name="Test Oyj", in_portfolio=True)

    @patch("news.services.ml_client.predict", return_value=NEG)
    def test_result_saved_and_alert_rule(self, _):
        news = News.objects.create(headline="Profit fell sharply", security=self.sec)
        self.assertTrue(classify_news(news))
        news.refresh_from_db()
        self.assertEqual(news.ml_sentiment, "negative")
        self.assertEqual(news.ml_model_version, "test")
        self.assertTrue(news.is_alert)
        self.assertEqual(news.status, News.Status.IN_REVIEW)

    @patch("news.services.ml_client.predict", return_value=NEG)
    def test_no_alert_outside_portfolio(self, _):
        self.sec.in_portfolio = False
        self.sec.save()
        news = News.objects.create(headline="Profit fell sharply", security=self.sec)
        classify_news(news)
        self.assertFalse(news.is_alert)

    @patch("news.services.ml_client.predict", side_effect=ml_client.MLServiceError("down"))
    def test_service_unavailable_news_still_saved(self, _):
        resp = self.client.post(reverse("news_create"), {
            "headline": "Profit fell sharply", "security": self.sec.pk,
            "source": "test", "published_at": "2026-10-28T12:00",
        }, follow=True)
        news = News.objects.get()
        self.assertIsNone(news.ml_sentiment)
        self.assertContains(resp, "ML-сервис недоступен")

    @patch("news.services.ml_client.predict", return_value=NEG)
    def test_analyst_override_has_priority(self, _):
        news = News.objects.create(headline="Profit fell sharply", security=self.sec)
        classify_news(news)
        self.client.post(reverse("news_detail", args=[news.pk]),
                         {"status": "done", "analyst": "", "analyst_sentiment": "neutral"})
        news.refresh_from_db()
        self.assertEqual(news.final_sentiment, "neutral")
        self.assertFalse(news.is_alert)
