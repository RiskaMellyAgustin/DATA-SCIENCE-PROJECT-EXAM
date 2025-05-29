from django.shortcuts import render, redirect
from .forms import DropoutPredictionForm
import joblib
import os
import pandas as pd
from .models import (
    DropoutPredictionLog,
    AnomalyDetectionLog,
    DropoutRegressionLog,
    ClusterAnomalyLog,
    DropoutExistLog
)

# =========================== LOAD SEMUA MODEL SEKALI SAJA ===========================
model_path = os.path.join('mellyapp', 'ml', 'dropout', 'xgboost_dropout_model.pkl')
model = joblib.load(model_path)

anomaly_model_path = os.path.join('mellyapp', 'ml', 'anomaly', 'isolation_forest_model.pkl')
anomaly_model = joblib.load(anomaly_model_path)

rf_model_path = os.path.join('mellyapp', 'ml', 'dropout', 'randomforest_dropout_model.pkl')
rf_model = joblib.load(rf_model_path)

cluster_model_path = os.path.join('mellyapp', 'ml', 'clustering', 'kmeans_anomaly_model.pkl')
cluster_model = joblib.load(cluster_model_path)

exist_model_path = r'D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT - Copy (5) - Copy/ds_project/mellyapp/ml/Exist_Data/xgboost_dropout_Exist_model.pkl'
exist_model = joblib.load(exist_model_path)

# =========================== DASHBOARD ===========================
def dashboard(request):
    return render(request, 'mellyapp/dashboard.html')


# =========================== XGBOOST (Dropout Classifier) ===========================
def predict_dropout(request):
    if request.method == 'POST':
        form = DropoutPredictionForm(request.POST)
        if form.is_valid():
            gender = form.cleaned_data['gender'].capitalize()
            age = form.cleaned_data['age']
            total_activities = form.cleaned_data['total_activities_done']
            total_duration = form.cleaned_data['total_duration_minutes']

            input_df = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'total_activities_done': total_activities,
                'total_duration_minutes': total_duration,
            }])

            prediction = model.predict(input_df)[0]

            DropoutPredictionLog.objects.create(
                gender=gender,
                age=age,
                total_activities_done=total_activities,
                total_duration_minutes=total_duration,
                prediction='Dropout' if prediction == 1 else 'Aktif'
            )

            return render(request, 'mellyapp/predict_result.html', {
                'gender': gender,
                'age': age,
                'total_activities': total_activities,
                'total_duration': total_duration,
                'prediction': prediction,
            })
    else:
        form = DropoutPredictionForm()

    return render(request, 'mellyapp/predict.html', {'form': form})


# =========================== ISOLATION FOREST (Anomaly) ===========================
def predict_anomaly(request):
    if request.method == 'POST':
        form = DropoutPredictionForm(request.POST)
        if form.is_valid():
            gender = form.cleaned_data['gender'].capitalize()
            age = form.cleaned_data['age']
            total_activities = form.cleaned_data['total_activities_done']
            total_duration = form.cleaned_data['total_duration_minutes']

            input_df = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'total_activities_done': total_activities,
                'total_duration_minutes': total_duration,
            }])

            pred = anomaly_model.predict(input_df)[0]  # -1 = anomaly
            score = anomaly_model.decision_function(input_df)[0]

            AnomalyDetectionLog.objects.create(
                gender=gender,
                age=age,
                total_activities_done=total_activities,
                total_duration_minutes=total_duration,
                anomaly_score=score,
                prediction='Anomali' if pred == -1 else 'Normal'
            )

            return render(request, 'mellyapp/anomaly_result.html', {
                'gender': gender,
                'age': age,
                'total_activities': total_activities,
                'total_duration': total_duration,
                'anomaly': pred,
                'score': score,
            })
    else:
        form = DropoutPredictionForm()

    return render(request, 'mellyapp/anomaly_predict.html', {'form': form})


# =========================== RANDOM FOREST REGRESSION ===========================
def predict_dropout_rf(request):
    if request.method == 'POST':
        form = DropoutPredictionForm(request.POST)
        if form.is_valid():
            gender = form.cleaned_data['gender'].capitalize()
            age = form.cleaned_data['age']
            total_activities = form.cleaned_data['total_activities_done']
            total_duration = form.cleaned_data['total_duration_minutes']

            input_df = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'total_activities_done': total_activities,
                'total_duration_minutes': total_duration,
            }])

            dropout_score = rf_model.predict(input_df)[0]

            DropoutRegressionLog.objects.create(
                gender=gender,
                age=age,
                total_activities_done=total_activities,
                total_duration_minutes=total_duration,
                dropout_score=dropout_score,
                prediction='Dropout' if dropout_score >= 0.5 else 'Aktif'
            )

            return render(request, 'mellyapp/predict_result_rf.html', {
                'gender': gender,
                'age': age,
                'total_activities': total_activities,
                'total_duration': total_duration,
                'dropout_score': dropout_score,
            })
    else:
        form = DropoutPredictionForm()

    return render(request, 'mellyapp/predict_rf.html', {'form': form})


# =========================== KMEANS CLUSTERING ===========================
def predict_cluster_anomaly(request):
    if request.method == 'POST':
        form = DropoutPredictionForm(request.POST)
        if form.is_valid():
            gender = form.cleaned_data['gender'].capitalize()
            age = form.cleaned_data['age']
            total_activities = form.cleaned_data['total_activities_done']
            total_duration = form.cleaned_data['total_duration_minutes']

            input_df = pd.DataFrame([{
                'gender': gender,
                'age': age,
                'total_activities_done': total_activities,
                'total_duration_minutes': total_duration,
            }])

            cluster_id = cluster_model.predict(input_df)[0]

            cluster_labels = {
                0: "🟢 Normal (Tidak Menunjukkan Anomali)",
                1: "🟡 Risiko Sedang (Perlu Perhatian)",
                2: "🔴 Anomali Tinggi (Aktivitas Janggal)"
            }

            label = cluster_labels.get(cluster_id, "Unknown Cluster")

            ClusterAnomalyLog.objects.create(
                gender=gender,
                age=age,
                total_activities_done=total_activities,
                total_duration_minutes=total_duration,
                cluster_id=cluster_id,
                cluster_label=label
            )

            return render(request, 'mellyapp/cluster_result.html', {
                'gender': gender,
                'age': age,
                'total_activities': total_activities,
                'total_duration': total_duration,
                'cluster_id': cluster_id,
                'cluster_desc': label,
            })
    else:
        form = DropoutPredictionForm()

    return render(request, 'mellyapp/cluster_predict.html', {'form': form})


# =========================== DROP OUT EXIST DATA (XGBOOST) ===========================
from django.shortcuts import render
from .forms import StudentSearchForm
from .models import Student, Enrollment, StudentActivityLog, DropoutExistLog
import joblib
import pandas as pd

# Load model exist (sekali load)
exist_model_path = r'D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT - Copy (5) - Copy/ds_project/mellyapp/ml/Exist_Data/xgboost_dropout_Exist_model.pkl'
exist_model = joblib.load(exist_model_path)

def predict_dropout_exist(request):
    result = None

    if request.method == 'POST':
        form = StudentSearchForm(request.POST)
        if form.is_valid():
            student_id = form.cleaned_data['student_id']

            try:
                # Ambil data student berdasarkan stu_id
                student = Student.objects.get(stu_id=student_id)

                # Ambil data enrollment student
                enrollment = Enrollment.objects.filter(student__stu_id=student_id).first()

                # Ambil data aktivitas
                activities = StudentActivityLog.objects.filter(student__stu_id=student_id)

                if not enrollment or activities.count() == 0:
                    return render(request, 'mellyapp/predict_exist.html', {'form': form, 'error': 'Data tidak lengkap untuk prediksi.'})

                # Hitung total aktivitas & durasi
                total_activities = activities.values('activity_id').distinct().count()
                total_duration = sum([
                    (a.activity_end - a.activity_start).total_seconds() / 60 for a in activities if a.activity_start and a.activity_end
                ])

                # Siapkan input model
                input_df = pd.DataFrame([{
                    'gender': student.gender.capitalize(),
                    'age': student.get_age(),
                    'total_activities_done': total_activities,
                    'total_duration_minutes': total_duration
                }])

                # Prediksi
                prediction = exist_model.predict(input_df)[0]
                probability = exist_model.predict_proba(input_df)[0][1]

                # Simpan log
                DropoutExistLog.objects.create(
                    gender=student.gender,
                    age=student.get_age(),
                    total_activities_done=total_activities,
                    total_duration_minutes=total_duration,
                    prediction='Dropout' if prediction == 1 else 'Active',
                    probability=probability
                )

                result = {
                    'gender': student.gender,
                    'age': student.get_age(),
                    'total_activities': total_activities,
                    'total_duration': total_duration,
                    'prediction': 'Dropout' if prediction == 1 else 'Active',
                    'probability': probability,
                    'name': student.name,
                    'stu_id': student.stu_id
                }

            except Student.DoesNotExist:
                return render(request, 'mellyapp/predict_exist.html', {'form': form, 'error': 'Mahasiswa tidak ditemukan.'})

    else:
        form = StudentSearchForm()

    return render(request, 'mellyapp/predict_exist.html', {'form': form, 'result': result})

# views.py

from django.shortcuts import render, get_object_or_404
from .forms import StudentSearchForm
from .models import Student

def search_student(request):
    student = None
    if request.method == 'POST':
        form = StudentSearchForm(request.POST)
        if form.is_valid():
            student_id = form.cleaned_data['student_id']
            student = get_object_or_404(Student, student_id=student_id)
    else:
        form = StudentSearchForm()
    return render(request, 'search_student.html', {'form': form, 'student': student})
