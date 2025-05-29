# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.dashboard, name='melly_dashboard'),  # ini home dashboard
#     path('predict/', views.predict_dropout, name='predict_dropout'),
#     path('predict-dropout/', views.predict_dropout, name='predict_dropout'),
#     path('predict-grade/', views.predict_grade_redirect, name='predict_grade'),  # sementara redirect
# ]
from django.urls import path
from . import views
from home.views import landing_page


app_name = 'mellyapp'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('landing/', landing_page, name='landing'),
    path('predict/', views.predict_dropout, name='predict_dropout'),
    # path('predict-grade/', views.predict_grade_redirect, name='predict_grade_redirect'),
    path('predict-anomaly/', views.predict_anomaly, name='predict_anomaly'),
    path('predict/dropout-rf/', views.predict_dropout_rf, name='predict_dropout_rf'),
    path('predict/clustering-anomaly/', views.predict_cluster_anomaly, name='predict_cluster_anomaly'),
    path('predict/dropout-exist/', views.predict_dropout_exist, name='predict_dropout_exist'),
    path('search-student/', views.search_student, name='search_student'),


]
