from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.predict_emotion, name='predict_emotion'),
]
