from django.urls import path
from . import views
from categorizer.views import index

urlpatterns = [
    path('', views.index, name='index'),
    path('history/', views.history, name='history'),
    path('', index, name='index'),
]