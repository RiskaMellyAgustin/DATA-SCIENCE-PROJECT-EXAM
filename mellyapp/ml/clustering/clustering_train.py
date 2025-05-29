import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib

# Load data
df = pd.read_csv('D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/dropout/DATASET DROPOUT 4.csv')
X = df.drop(columns=['dropout', 'grade'])

categorical_features = ['gender']
numerical_features = ['age', 'total_activities_done', 'total_duration_minutes']

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# Pipeline + KMeans
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# Fit model
pipeline.fit(X)

# Get cluster labels
X_transformed = pipeline.named_steps['preprocess'].transform(X)
cluster_labels = pipeline.named_steps['kmeans'].labels_

# Evaluasi
inertia = pipeline.named_steps['kmeans'].inertia_
silhouette = silhouette_score(X_transformed, cluster_labels)
db_score = davies_bouldin_score(X_transformed, cluster_labels)

print("\n🔍 Evaluasi Clustering:")
print(f"- Inertia (jumlah kuadrat jarak dalam cluster): {inertia:.2f}")
print(f"- Silhouette Score (semakin dekat ke 1 lebih baik): {silhouette:.3f}")
print(f"- Davies-Bouldin Score (semakin kecil lebih baik): {db_score:.3f}")

# Save model
model_path = 'D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/clustering/kmeans_anomaly_model.pkl'
joblib.dump(pipeline, model_path)
print(f"\n✅ Model clustering disimpan di {model_path}")
