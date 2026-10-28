from django.contrib import messages
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import ml_client
from .forms import NewsCreateForm, NewsProcessForm
from .models import News, Security, Sentiment
from .services import apply_business_rules, classify_news, classify_pending


def news_list(request):
    qs = News.objects.select_related("security", "analyst")
    sentiment = request.GET.get("sentiment", "")
    status = request.GET.get("status", "")
    security = request.GET.get("security", "")
    if sentiment == "none":
        qs = qs.filter(ml_sentiment__isnull=True, analyst_sentiment__isnull=True)
    elif sentiment:
        qs = qs.filter(Q(analyst_sentiment=sentiment) | Q(analyst_sentiment__isnull=True, ml_sentiment=sentiment))
    if status:
        qs = qs.filter(status=status)
    if security:
        qs = qs.filter(security__ticker=security)
    return render(request, "news/list.html", {
        "news_list": qs,
        "securities": Security.objects.all(),
        "sentiments": Sentiment.choices,
        "statuses": News.Status.choices,
        "f": {"sentiment": sentiment, "status": status, "security": security},
        "ml_health": ml_client.health(),
        "alerts": News.objects.filter(is_alert=True).exclude(status__in=["done", "archived"]).count(),
    })


def news_create(request):
    form = NewsCreateForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        news = form.save()
        if classify_news(news):
            messages.success(request, f"Новость сохранена. Тональность по оценке модели: "
                                      f"{news.get_ml_sentiment_display().lower()} ({news.ml_confidence:.0%}).")
        else:
            messages.warning(request, "Новость сохранена, но ML-сервис недоступен — тональность будет определена позже.")
        return redirect("news_detail", pk=news.pk)
    return render(request, "news/create.html", {"form": form})


def news_detail(request, pk):
    news = get_object_or_404(News.objects.select_related("security", "analyst"), pk=pk)
    form = NewsProcessForm(request.POST or None, instance=news)
    if request.method == "POST" and form.is_valid():
        news = form.save(commit=False)
        apply_business_rules(news)
        news.save()
        messages.success(request, "Изменения сохранены.")
        return redirect("news_detail", pk=news.pk)
    probs = []
    if news.ml_probabilities:
        probs = [(Sentiment(k).label, k, v) for k, v in sorted(news.ml_probabilities.items(), key=lambda kv: -kv[1])]
    return render(request, "news/detail.html", {"news": news, "form": form, "probs": probs})


@require_POST
def news_reclassify(request, pk):
    news = get_object_or_404(News, pk=pk)
    if classify_news(news):
        messages.success(request, "Новость заново классифицирована ML-сервисом.")
    else:
        messages.error(request, "ML-сервис недоступен.")
    return redirect("news_detail", pk=pk)


@require_POST
def classify_all(request):
    done, failed = classify_pending()
    if failed:
        messages.error(request, f"ML-сервис недоступен. Классифицировано: {done}.")
    else:
        messages.success(request, f"Классифицировано новостей: {done}.")
    return redirect("news_list")


def portfolio(request):
    """Сводка тональности новостей по бумагам портфеля."""
    def cnt(value):
        return Count("news", filter=Q(news__analyst_sentiment=value) |
                     Q(news__analyst_sentiment__isnull=True, news__ml_sentiment=value))

    rows = Security.objects.annotate(
        pos=cnt(Sentiment.POSITIVE), neu=cnt(Sentiment.NEUTRAL), neg=cnt(Sentiment.NEGATIVE),
        alerts=Count("news", filter=Q(news__is_alert=True)),
    )
    for r in rows:
        total = r.pos + r.neu + r.neg
        r.total = total
        r.index = round((r.pos - r.neg) / total, 2) if total else None
    return render(request, "news/portfolio.html", {"rows": rows})
