import joblib
import pandas as pd
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from django.shortcuts import render
from .forms import DataInputForm
from .models import StudentPrediction  # Import model untuk menyimpan prediksi

# Load semua model dan scaler
xgb_model = joblib.load('valenciaapp/xgboost_student_segment_model.pkl')
kmeans_model = joblib.load('valenciaapp/kmeans_student_segment_model.pkl')
scaler = joblib.load('valenciaapp/scaler.pkl')
feature_names = joblib.load('valenciaapp/feature_names.pkl')


def index(request):
    return render(request, 'index.html')


def predict_student(request):
    if request.method == 'POST':
        form = DataInputForm(request.POST)
        if form.is_valid():
            input_data = form.cleaned_data.copy()

            # Mapping checkbox ke one-hot encoding
            input_data['dominant_activity_type_Forum'] = 1 if input_data.get('dominant_activity_type_forum') else 0
            input_data['dominant_activity_type_Group Assignment'] = 1 if input_data.get('dominant_activity_type_group_assignment') else 0
            input_data['dominant_activity_type_Individual Assignment'] = 1 if input_data.get('dominant_activity_type_individual_assignment') else 0

            # Simpan nilai-nilai asli sebelum pop untuk disimpan di database
            original_data = {
                'activity_count': input_data['activity_count'],
                'login_days': input_data['login_days'],
                'total_study_time': input_data['total_study_time'],
                'avg_time_per_activity': input_data['avg_time_per_activity'],
                'avg_active_hour': input_data['avg_active_hour'],
                'grade_mean': input_data['grade_mean'],
                'grade_stddev': input_data['grade_stddev'],
                'dominant_activity_type_forum': input_data['dominant_activity_type_Forum'],
                'dominant_activity_type_group_assignment': input_data['dominant_activity_type_Group Assignment'],
                'dominant_activity_type_individual_assignment': input_data['dominant_activity_type_Individual Assignment'],
            }

            # Hapus kolom checkbox boolean asli
            for key in ['dominant_activity_type_forum', 'dominant_activity_type_group_assignment', 'dominant_activity_type_individual_assignment']:
                input_data.pop(key, None)

            # Simpan algoritma dan hapus dari dict untuk preprocessing
            algorithm = input_data.pop('algorithm')

            # Buat DataFrame dari input user
            input_df = pd.DataFrame([input_data])

            # Pastikan semua kolom fitur ada
            for col in feature_names:
                if col not in input_df.columns:
                    input_df[col] = 0

            # Urutkan kolom agar cocok dengan saat training
            input_df = input_df[feature_names]

            # Scaling
            input_scaled = scaler.transform(input_df)

            # Inisialisasi hasil prediksi dan grafik
            predicted_segment = "Unknown"
            fig = None

            # Prediksi berdasarkan algoritma
            if algorithm == 'xgboost':
                prediction = xgb_model.predict(input_scaled)
                label_map = {
                     0: 'Active Learners',
                     1: 'Balanced Learners',   
                     2: 'Passive Learners'
                }

                predicted_segment = label_map.get(prediction[0], "Unknown")

                # Plot segment prediction
                fig, ax = plt.subplots()
                segments = ['Active Learners', 'Balanced Learners', 'Passive Learners']
                values = [0, 0, 0]
                values[prediction[0]] = 1
                ax.bar(segments, values, color=['green', 'blue', 'red'])
                ax.set_title(f"Student Segment Breakdown (Predicted: {predicted_segment})")
                plt.tight_layout()

            elif algorithm == 'kmeans':
                prediction = kmeans_model.predict(input_scaled)
                cluster_label = prediction[0]
                cluster_names = {
                    0: "Passive Learner",
                    1: "Balanced Learner",
                    2: "Active Learner"
                }
                predicted_segment = cluster_names.get(cluster_label, f"Cluster {cluster_label}")
                
                fig, ax = plt.subplots()
                clusters = [cluster_names[i] for i in range(kmeans_model.n_clusters)]
                values = [0] * kmeans_model.n_clusters
                values[prediction[0]] = 1  # Set the predicted cluster value to 1
                ax.bar(clusters, values, color='skyblue')
                ax.set_title(f"Cluster Assignment (Predicted: {predicted_segment})")
                plt.tight_layout()

            # Konversi gambar ke base64
            if fig:
                buf = BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                image_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
            else:
                image_base64 = None

            # Simpan hasil ke database
            StudentPrediction.objects.create(
                activity_count=original_data['activity_count'],
                login_days=original_data['login_days'],
                total_study_time=original_data['total_study_time'],
                avg_time_per_activity=original_data['avg_time_per_activity'],
                avg_active_hour=original_data['avg_active_hour'],
                grade_mean=original_data['grade_mean'],
                grade_stddev=original_data['grade_stddev'],
                dominant_activity_type_forum=original_data['dominant_activity_type_forum'],
                dominant_activity_type_group_assignment=original_data['dominant_activity_type_group_assignment'],
                dominant_activity_type_individual_assignment=original_data['dominant_activity_type_individual_assignment'],
                algorithm=algorithm,
                predicted_segment=predicted_segment
            )

            return render(request, 'result.html', {
                'prediction_segment': predicted_segment,
                'image': image_base64,
                'algorithm': algorithm
            })

        else:
            return render(request, 'predict.html', {'form': form, 'errors': form.errors})
    else:
        form = DataInputForm()
        return render(request, 'predict.html', {'form': form})
