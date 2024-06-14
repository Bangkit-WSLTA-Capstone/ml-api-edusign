from django.urls import path
from translation.views import translate_video

app_name = 'translation'

urlpatterns = [
    path('', translate_video, name='translate'),
]