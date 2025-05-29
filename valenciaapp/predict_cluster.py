import pandas as pd
import joblib

# Load model dan scaler
kmeans = joblib.load('ds_project/valenciaapp/kmeans_student_segment_model.pkl')
scaler = joblib.load('ds_project/valenciaapp/scaler.pkl')
feature_names = joblib.load('ds_project/valenciaapp/feature_names.pkl')

def predict_student_cluster(input_data: dict):
    """
    input_data contoh:
    {
      'activity_count': 12,
      'login_days': 20,
      'total_study_time': 1600,
      'avg_time_per_activity': 75,
      'avg_active_hour': 15,
      'grade_mean': 80,
      'grade_stddev': 5
    }
    """
    # Buat DataFrame dari input
    input_df = pd.DataFrame([input_data])

    # Pastikan semua fitur ada, jika tidak tambahkan dengan nilai 0
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_names]

    # Standarisasi fitur input
    input_scaled = scaler.transform(input_df.values)

    # Prediksi cluster
    cluster_label = kmeans.predict(input_scaled)[0]

    # Label cluster bisa kamu definisikan sendiri sesuai hasil interpretasi cluster
    cluster_names = {
    0: "Passive Learner",
    1: "Balanced Learner",
    2: "Active Learner"
    }
    return cluster_label, cluster_names.get(cluster_label, "Unknown Cluster")

# Contoh penggunaan
if __name__ == "__main__":
    sample_input = {
        'activity_count': 12,
        'login_days': 20,
        'total_study_time': 1600,
        'avg_time_per_activity': 75,
        'avg_active_hour': 15,
        'grade_mean': 80,
        'grade_stddev': 5
    }
    label, name = predict_student_cluster(sample_input)
    print(f"Prediksi cluster: {label} - {name}")
