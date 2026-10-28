"""Наполнение БД демонстрационными данными: бумаги портфеля, аналитики и 27 новостей,
отложенных из датасета (в обучении модели не участвовали).

python manage.py seed_demo          — создать данные и классифицировать новости через ML-сервис
python manage.py seed_demo --no-ml  — создать данные без обращения к ML-сервису
"""
import csv
from datetime import timedelta
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from news.models import News, Security
from news.services import classify_news

DEMO_CSV = Path(__file__).resolve().parents[4] / "data" / "demo_news.csv"

SECURITIES = [
    ("NOKIA", "Nokia Oyj", "Телекоммуникационное оборудование", True),
    ("STERV", "Stora Enso Oyj", "Лесная и целлюлозно-бумажная промышленность", True),
    ("UPM", "UPM-Kymmene Oyj", "Лесная и целлюлозно-бумажная промышленность", True),
    ("KNEBV", "Kone Oyj", "Машиностроение", True),
    ("NDA FI", "Nordea Bank Abp", "Банки", True),
    ("FIA1S", "Finnair Oyj", "Авиаперевозки", False),
]
ANALYSTS = [("ivanov", "Пётр", "Иванов"), ("petrova", "Анна", "Петрова")]


class Command(BaseCommand):
    help = "Создаёт демонстрационные данные ИС"

    def add_arguments(self, parser):
        parser.add_argument("--no-ml", action="store_true", help="не вызывать ML-сервис")

    def handle(self, *args, **opts):
        User = get_user_model()
        if not User.objects.filter(username="admin").exists():
            User.objects.create_superuser("admin", "admin@example.com", "admin12345",
                                          first_name="Александр", last_name="Морозов")
        analysts = []
        for username, first, last in ANALYSTS:
            u, _ = User.objects.get_or_create(username=username, defaults={"first_name": first, "last_name": last})
            analysts.append(u)

        secs = {}
        for ticker, name, sector, in_portfolio in SECURITIES:
            secs[ticker], _ = Security.objects.update_or_create(
                ticker=ticker, defaults={"name": name, "sector": sector, "in_portfolio": in_portfolio})

        News.objects.all().delete()
        now = timezone.now().replace(second=0, microsecond=0)
        with open(DEMO_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ok = 0
        for i, row in enumerate(rows):
            news = News.objects.create(
                headline=row["text"],
                security=secs[row["ticker"]],
                source="FinancialPhraseBank",
                published_at=now - timedelta(hours=7 * i + 1),
                analyst=analysts[i % 2] if i % 3 == 0 else None,
                status=News.Status.DONE if i % 5 == 4 else News.Status.NEW,
            )
            if not opts["no_ml"]:
                ok += classify_news(news)
        self.stdout.write(self.style.SUCCESS(
            f"Бумаг: {len(secs)}, аналитиков: {len(analysts)}, новостей: {len(rows)}, классифицировано: {ok}"))
