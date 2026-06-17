# movie_api/config/urls.py
from django.urls import path, include

urlpatterns = [
    path('api/', include('recommender.urls')),
]