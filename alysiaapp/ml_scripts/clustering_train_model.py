import pandas as pd
import numpy as np
import joblib
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ========== LOAD DATA ==========
df = pd.read_csv('alysiaapp/ml_scripts/dataset_clean.csv')

# ========== KONVERSI JAM KE KATEGORI WAKTU ==========
def waktu_kategori(h):
    if 3 <= h <= 10:
        return 'Pagi'
    elif 11 <= h <= 17:
        return 'Siang'
    else:
        return 'Malam'

if 'hour' not in df.columns:
    raise KeyError("❌ Kolom 'hour' tidak ditemukan di dataset.")
df['hour'] = df['hour'].astype(int)
df['waktu_label'] = df['hour'].apply(waktu_kategori)

# ========== HITUNG DURASI DAN HARI ==========
if 'activity_start' in df.columns and 'activity_end' in df.columns:
    df['activity_start'] = pd.to_datetime(df['activity_start'])
    df['activity_end'] = pd.to_datetime(df['activity_end'])
    df['duration_minutes'] = (df['activity_end'] - df['activity_start']).dt.total_seconds() / 60
    df['day_of_week'] = df['activity_start'].dt.dayofweek
    print("✅ Kolom duration_minutes & day_of_week berhasil dihitung.")
elif 'duration_minutes' not in df.columns or 'day_of_week' not in df.columns:
    raise KeyError("❌ Tidak ada kolom activity_start & activity_end untuk menghitung durasi dan hari.")

# ========== ENCODE KATEGORI WAKTU BELAJAR ==========
le_time = LabelEncoder()
df['time_of_day_encoded'] = le_time.fit_transform(df['waktu_label'])
print("✅ 'time_of_day_encoded' column created.")

# ========== FITUR YANG DIGUNAKAN ==========
features = ['gender', 'age', 'grade', 'activity_type', 'duration_minutes', 'day_of_week']
X = df[features]

# ========== PIPELINE CLUSTERING ==========
categorical_cols = ['gender', 'activity_type']
numerical_cols = ['age', 'grade', 'duration_minutes', 'day_of_week']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numerical_cols)
])

pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('clustering', KMeans(n_clusters=3, random_state=42))
])

pipeline.fit(X)

# ========== SIMPAN MODEL ==========
model_path = os.path.join(os.path.dirname(__file__), "clustering_model.pkl")
joblib.dump(pipeline, model_path)
print(f"📁 Menyimpan model ke: {os.path.abspath(model_path)}")

# ========== TAMBAHKAN LABEL CLUSTER KE DATA ==========
df['cluster'] = pipeline.named_steps['clustering'].labels_

# ========== PCA UNTUK VISUALISASI DAN KLASIFIKASI ==========
df_encoded = df[features].copy()
le_gender = LabelEncoder()
le_activity = LabelEncoder()
le_day = LabelEncoder()

df_encoded['gender'] = le_gender.fit_transform(df_encoded['gender'])
df_encoded['activity_type'] = le_activity.fit_transform(df_encoded['activity_type'])
df_encoded['day_of_week'] = le_day.fit_transform(df_encoded['day_of_week'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded)

pca_final = PCA(n_components=6, random_state=42)
X_pca_final = pca_final.fit_transform(X_scaled)

# ========== KLASIFIKASI PER CLUSTER ==========
X_classifier = X_pca_final
y_classifier = df['time_of_day_encoded']
clusters = df['cluster']
unique_clusters = sorted(clusters.unique())

print("\n=== Klasifikasi Waktu Belajar per Cluster ===")
for cluster_id in unique_clusters:
    print(f"\n--- Cluster {cluster_id} ---")
    cluster_indices = clusters[clusters == cluster_id].index
    X_cluster = X_classifier[cluster_indices]
    y_cluster = y_classifier[cluster_indices]

    if len(np.unique(y_cluster)) < 2:
        print("⚠️ Hanya ada satu kelas di cluster ini, melewatkan klasifikasi.")
        continue

    if len(X_cluster) < 10:
        print(f"⚠️ Cluster {cluster_id} terlalu kecil ({len(X_cluster)} sampel). Melewatkan klasifikasi.")
        continue

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_cluster, y_cluster, test_size=0.3, random_state=42, stratify=y_cluster
        )
    except ValueError:
        print("⚠️ Stratifikasi gagal, menggunakan split acak.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_cluster, y_cluster, test_size=0.3, random_state=42
        )

    if cluster_id in [0, 2]:
        print("⚙️ Menggunakan RandomForestClassifier")
        classifier = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    else:
        classifier = LogisticRegression(random_state=42, solver='liblinear')

    try:
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=le_time.inverse_transform(np.unique(y_test))))
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 Akurasi: {acc:.4f}")
    except ValueError as e:
        print(f"❌ Gagal melatih klasifikasi cluster {cluster_id}: {e}")

# ========== SIMPAN ENCODER, SCALER, PCA ==========
joblib.dump(le_gender, os.path.join(os.path.dirname(__file__), "le_gender.pkl"))
joblib.dump(le_activity, os.path.join(os.path.dirname(__file__), "le_activity.pkl"))
joblib.dump(le_day, os.path.join(os.path.dirname(__file__), "le_day.pkl"))
joblib.dump(scaler, os.path.join(os.path.dirname(__file__), "clustering_scaler.pkl"))
joblib.dump(pca_final, os.path.join(os.path.dirname(__file__), "clustering_pca.pkl"))

# ========== EVALUASI CLUSTERING ==========
X_preprocessed = pipeline.named_steps['preprocessing'].transform(X)
sil_score = silhouette_score(X_preprocessed, df['cluster'])

# ========== VISUALISASI CLUSTER ==========
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='hour', y='grade', hue='cluster', palette='Set2')
plt.title("Visualisasi Cluster: Jam vs Nilai")
plt.xlabel("Jam Aktivitas")
plt.ylabel("Grade")
plt.grid(True)
os.makedirs("alysiaapp/visuals", exist_ok=True)
plt.savefig("alysiaapp/visuals/cluster_plot.png")
print("📊 Visualisasi disimpan sebagai 'cluster_plot.png'")
