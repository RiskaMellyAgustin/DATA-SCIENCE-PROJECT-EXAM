import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import joblib  # Untuk menyimpan model

# 1. Load dataset
df = pd.read_csv('D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/dropout/DATASET DROPOUT 4.csv')

# 2. Drop kolom target dari fitur input
X = df.drop(columns=['dropout', 'grade'])

# 3. Definisikan fitur kategorikal dan numerikal
categorical_features = ['gender']
numerical_features = ['age', 'total_activities_done', 'total_duration_minutes']

# 4. Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# 5. Pipeline IsolationForest
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('model', IsolationForest(contamination=0.1, random_state=42))
])

# 6. Fit model
pipeline.fit(X)

# 7. Dapatkan skor anomali
scores = pipeline.named_steps['model'].decision_function(
    pipeline.named_steps['preprocess'].transform(X)
)

# 8. Visualisasi distribusi skor
plt.figure(figsize=(10, 5))
plt.hist(scores, bins=50)
plt.title("Distribusi Skor Anomali (semakin kecil = semakin aneh)")
plt.xlabel("Outlier Score")
plt.ylabel("Jumlah Mahasiswa")
plt.grid(True)
plt.tight_layout()
plt.show()

# 9. Simpan model ke .pkl
model_path = 'D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/anomaly/isolation_forest_model.pkl'
joblib.dump(pipeline, model_path)
print(f"✅ Model berhasil disimpan ke: {model_path}")
