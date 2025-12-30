# app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),  # root
    path('search/', views.search, name='search'),
    path('autocomplete/', views.autocomplete_tickers, name='autocomplete_tickers'),
    path('predict/<str:ticker_value>/', views.predict, {"number_of_days": 1}, name='predict_default'),
    path('predict/<str:ticker_value>/<int:number_of_days>/', views.predict, name='predict'),
]
