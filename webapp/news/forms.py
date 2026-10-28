from django import forms
from django.contrib.auth import get_user_model

from .models import News, Sentiment


class NewsCreateForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["security"].empty_label = "— не указана —"

    class Meta:
        model = News
        fields = ["headline", "security", "source", "published_at"]
        widgets = {
            "headline": forms.Textarea(attrs={"rows": 3, "placeholder": "Заголовок или текст новости (англ.)"}),
            "published_at": forms.DateTimeInput(attrs={"type": "datetime-local"}, format="%Y-%m-%dT%H:%M"),
        }


class NewsProcessForm(forms.ModelForm):
    """Обработка новости аналитиком: статус, исполнитель, исправление тональности."""

    class Meta:
        model = News
        fields = ["status", "analyst", "analyst_sentiment"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["analyst"].queryset = get_user_model().objects.filter(is_active=True).order_by("last_name")
        self.fields["analyst"].label_from_instance = lambda u: u.get_full_name() or u.username
        self.fields["analyst"].empty_label = "— не назначен —"
        self.fields["analyst_sentiment"].required = False
        self.fields["analyst_sentiment"].choices = [("", "— как определила модель —"), *Sentiment.choices]
