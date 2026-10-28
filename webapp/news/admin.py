from django.contrib import admin

from .models import News, Security


@admin.register(Security)
class SecurityAdmin(admin.ModelAdmin):
    list_display = ("ticker", "name", "sector", "in_portfolio")


@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = ("headline", "security", "status", "ml_sentiment", "ml_confidence", "analyst_sentiment", "is_alert")
    list_filter = ("status", "ml_sentiment", "is_alert", "security")
    search_fields = ("headline",)
