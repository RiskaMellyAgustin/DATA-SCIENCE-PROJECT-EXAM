from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  
    path('predict/', views.predict_student, name='predict_student'),  
]
