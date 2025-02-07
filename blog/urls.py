from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload, name='upload'),
    path('image/<int:id>/', views.image_detail, name='image_detail'),
]