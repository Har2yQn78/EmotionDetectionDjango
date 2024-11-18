from django.urls import path
from .views import predict_emotion

app_name = 'emotion'

urlpatterns = [
    path('detect/', predict_emotion, name='detect'),
]
