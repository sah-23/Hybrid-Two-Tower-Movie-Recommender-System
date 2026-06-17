# movie_api/recommender/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('movies/', views.movies_list),
    path('recommend/', views.recommend),
    path('v2/movies/',      views.movies_list_25m),
    path('v2/recommend/',   views.recommend_25m),
    path('similar/',  views.similar_movies),
    path('tmdb/search/', views.tmdb_search),
    path('tmdb/similar/', views.tmdb_similar),
]