import joblib
import pandas as pd
import os

# Fungsi konversi jam ke kategori waktu (jika dibutuhkan di UI/validasi)
def waktu_kategori(h):
    if 3 <= h <= 10:
        return 'Morning'
    elif 11 <= h <= 17:
        return 'Afternoon'
    else:
        return 'Night'

# Fungsi memuat model clustering
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "clustering_model.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Error: File model tidak ditemukan di path: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        print("✅ Model clustering berhasil dimuat.")
        return model
    except Exception as e:
        print(f"❌ Error saat load model dari '{model_path}': {e}")
        return None

# Fungsi prediksi cluster dan waktu belajar
def predict_best_study_time(gender, age, target_grade, activity_type, duration, day_of_week, model=None):
    if model is None:
        model = load_model()
        if model is None:
            return None, "Model tidak tersedia atau gagal dimuat"

    # Buat dataframe input sesuai urutan fitur saat pelatihan
    df_input = pd.DataFrame([{
        'gender': gender,
        'age': int(age),
        'grade': float(target_grade),
        'activity_type': activity_type,
        'duration_minutes': float(duration),
        'day_of_week': int(day_of_week)
    }])

    try:
        cluster = model.predict(df_input)[0]
    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return None, "Prediksi gagal"

    # Mapping cluster ke waktu belajar (kamu bisa update mapping ini jika perlu)
    cluster_map = {0: 'Pagi', 1: 'Siang', 2: 'Malam'}
    best_time = cluster_map.get(cluster, 'Tidak diketahui')

    return cluster, best_time

# Debug manual (run langsung file ini)
if __name__ == "__main__":
    cluster, waktu = predict_best_study_time("Female", 21, 85, "quiz", 50, 2)
    if cluster is not None:
        print(f"✅ Rekomendasi waktu belajar terbaik: {waktu} (Cluster {cluster})")
    else:
        print(f"❌ {waktu}")
