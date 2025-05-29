import pandas as pd
import joblib

# Load model, scaler, dan fitur yang digunakan saat training
model = joblib.load('ds_project/valenciaapp/xgboost_student_segment_model.pkl')
scaler = joblib.load('ds_project/valenciaapp/scaler.pkl')
feature_names = joblib.load('ds_project/valenciaapp/feature_names.pkl')
label_encoder = joblib.load('ds_project/valenciaapp/label_encoder.pkl')

# Contoh input user yang ingin diprediksi
# Pastikan input ini punya semua fitur yang model butuhkan, atau kamu harus sesuaikan
input_data = {
    'feature1': 10,
    'feature2': 5,
    'feature3': 1,
    # tambahkan semua fitur sesuai feature_names.pkl, tanpa fitur yang tidak dikenal seperti dominant_activity_type_Forum, dst.
}

# Ubah input_data ke DataFrame
input_df = pd.DataFrame([input_data])

# Filter kolom supaya sesuai dengan fitur saat training
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Scale input data
input_scaled = scaler.transform(input_df)

# Prediksi menggunakan model
prediction = model.predict(input_scaled)

print("Predicted class:", prediction[0])
