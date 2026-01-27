from django.urls import path
from .views import upload_audio

urlpatterns = [
    path("", upload_audio, name="upload"),
]