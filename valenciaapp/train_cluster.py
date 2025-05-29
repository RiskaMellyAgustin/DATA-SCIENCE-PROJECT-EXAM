import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import os

# Load data fitur
df = pd.read_csv('student_learning_features.csv')

features = [
    'activity_count',
    'login_days',
    'total_study_time',
    'avg_time_per_activity',
    'avg_active_hour',
    'grade_mean',
    'grade_stddev'
]

X = df[features].fillna(0)

# Standarisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Training KMeans clustering
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)

# Buat folder simpan model jika belum ada
save_dir = 'ds_project/valenciaapp'
os.makedirs(save_dir, exist_ok=True)

# Simpan model dan scaler
joblib.dump(kmeans, os.path.join(save_dir, 'kmeans_student_segment_model.pkl'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
joblib.dump(features, os.path.join(save_dir, 'feature_names.pkl'))

print("The training is complete and the model has been saved.")
