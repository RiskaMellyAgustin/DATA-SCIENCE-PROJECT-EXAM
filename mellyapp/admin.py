# from django.contrib import admin
# # mellyapp/admin.py

# from django.contrib import admin
# from .models import DropoutPredictionLog

# @admin.register(DropoutPredictionLog)
# class DropoutPredictionAdmin(admin.ModelAdmin):
#     list_display = ['gender', 'age', 'total_activities_done', 'prediction', 'created_at']
#     search_fields = ['gender', 'prediction']
#     list_filter = ['prediction', 'gender']


# === admin.py ===
from django.contrib import admin
from .models import DropoutPredictionLog
from django.urls import path
from django.shortcuts import redirect
from django.contrib import messages
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import os

@admin.register(DropoutPredictionLog)
class DropoutPredictionLogAdmin(admin.ModelAdmin):
    list_display = ('gender', 'age', 'total_activities_done', 'total_duration_minutes', 'prediction', 'created_at')
    change_list_template = "admin/retrain_changelist.html"

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('retrain-model/', self.admin_site.admin_view(self.retrain_model), name="retrain_model"),
        ]
        return custom_urls + urls

    def retrain_model(self, request):
        queryset = DropoutPredictionLog.objects.all()
        if not queryset.exists():
            self.message_user(request, "Tidak ada data untuk melatih ulang.", level=messages.WARNING)
            return redirect("..")

        df = pd.DataFrame(list(queryset.values('gender', 'age', 'total_activities_done', 'total_duration_minutes', 'prediction')))
        df['dropout'] = df['prediction'].apply(lambda x: 1 if x == 'Dropout' else 0)
        df = df.drop(columns=['prediction'])

        X = df.drop(columns=['dropout'])
        y = df['dropout']

        categorical_features = ['gender']
        numerical_features = ['age', 'total_activities_done', 'total_duration_minutes']

        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(drop='first'), categorical_features),
            ('num', StandardScaler(), numerical_features)
        ])

        pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('clf', XGBClassifier(
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ))
        ])

        pipeline.fit(X, y)

        model_path = os.path.join(os.path.dirname(__file__), 'ml', 'dropout', 'xgboost_dropout_model.pkl')
        joblib.dump(pipeline, model_path)
        self.message_user(request, "Model berhasil dilatih ulang dan disimpan.", level=messages.SUCCESS)
        return redirect("..")

from .models import (
    DropoutPredictionLog, AnomalyDetectionLog,
    DropoutRegressionLog, ClusterAnomalyLog
)

# @admin.register(DropoutPredictionLog)
# class DropoutPredictionLogAdmin(admin.ModelAdmin):
#     list_display = ('gender', 'age', 'total_activities_done', 'total_duration_minutes', 'prediction', 'created_at')

@admin.register(AnomalyDetectionLog)
class AnomalyDetectionLogAdmin(admin.ModelAdmin):
    list_display = ('gender', 'age', 'total_activities_done', 'total_duration_minutes', 'anomaly_score', 'prediction', 'created_at')

@admin.register(DropoutRegressionLog)
class DropoutRegressionLogAdmin(admin.ModelAdmin):
    list_display = ('gender', 'age', 'total_activities_done', 'total_duration_minutes', 'dropout_score', 'prediction', 'created_at')

@admin.register(ClusterAnomalyLog)
class ClusterAnomalyLogAdmin(admin.ModelAdmin):
    list_display = ('gender', 'age', 'total_activities_done', 'total_duration_minutes', 'cluster_id', 'cluster_label', 'created_at')


# Register your models here.
