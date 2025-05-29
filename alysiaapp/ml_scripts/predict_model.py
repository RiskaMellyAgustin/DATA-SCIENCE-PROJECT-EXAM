import joblib
import pandas as pd
import os
import warnings

# Path absolut
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'waktu_belajar_model.pkl')
encoder_path = os.path.join(BASE_DIR, 'label_encoder.pkl')

# Load model dan encoder
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    warnings.warn(f"Model file not found: {model_path}")
    model = None

if os.path.exists(encoder_path):
    le_target = joblib.load(encoder_path)
else:
    warnings.warn(f"Label encoder file not found: {encoder_path}")
    le_target = None

# Fungsi prediksi label
def predict_waktu_belajar(input_data):
    if model is None or le_target is None:
        return "Model atau encoder tidak tersedia."
    df_input = pd.DataFrame([input_data])
    pred_encoded = model.predict(df_input)[0]
    pred_label = le_target.inverse_transform([pred_encoded])[0]
    return pred_label

# Fungsi prediksi probabilitas
def predict_proba_waktu_belajar(input_data):
    if model is None or le_target is None:
        return None
    df_input = pd.DataFrame([input_data])
    proba = model.predict_proba(df_input)[0]
    labels = le_target.inverse_transform(model.named_steps['classifier'].classes_)
    return {label: float(p) for label, p in zip(labels, proba)}  # 👈 Konversi ke float Python

