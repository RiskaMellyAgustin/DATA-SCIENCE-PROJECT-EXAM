import joblib
import pandas as pd
import os

# Load model dan expected columns
model = joblib.load(os.path.join('ruthapp', 'ml', 'xgb_classifier_model.pkl'))
expected_columns = joblib.load(os.path.join('ruthapp', 'ml', 'expected_columns.pkl'))

# Fungsi preprocessing
def preprocess_input(data):
    df = pd.DataFrame([data])
    df['duration_per_activity'] = df['total_duration_minutes'] / (df['total_activities'] + 1e-5)
    df = pd.get_dummies(df)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    return df

# Fungsi konversi score ke label efektivitas
def to_effectiveness_label(score):
    if score < 60:
        return "Low"
    elif score < 75:
        return "Medium"
    elif score < 90:
        return "High"
    else:
        return "Excellent"

# Fungsi prediksi efektivitas
def predict_effectiveness(input_data):
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]
    score = round(prediction, 2)
    label = to_effectiveness_label(score)
    return label  # Bisa juga return score, label jika ingin tampilkan keduanya
