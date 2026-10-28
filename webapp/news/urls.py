from django.urls import path

from . import views

urlpatterns = [
    path("", views.news_list, name="news_list"),
    path("news/add/", views.news_create, name="news_create"),
    path("news/<int:pk>/", views.news_detail, name="news_detail"),
    path("news/<int:pk>/reclassify/", views.news_reclassify, name="news_reclassify"),
    path("news/classify-all/", views.classify_all, name="classify_all"),
    path("portfolio/", views.portfolio, name="portfolio"),
]
