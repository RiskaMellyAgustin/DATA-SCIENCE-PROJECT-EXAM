import joblib
import pandas as pd

# Load model dan scaler
kmeans = joblib.load('valenciaapp/kmeans_model.pkl')
scaler = joblib.load('valenciaapp/kmeans_scaler.pkl')
feature_names = joblib.load('valenciaapp/cluster_feature_names.pkl')

def predict_cluster(input_data: dict) -> int:
    """
    Receive input from the form as a dictionary and return the clustering results.

    """
    df = pd.DataFrame([input_data])

    # One-hot encoding manual
    df['dominant_activity_type_Forum'] = 1 if input_data.get('dominant_activity_type_forum') else 0
    df['dominant_activity_type_Group Assignment'] = 1 if input_data.get('dominant_activity_type_group_assignment') else 0
    df['dominant_activity_type_Individual Assignment'] = 1 if input_data.get('dominant_activity_type_individual_assignment') else 0

    # Tambahkan kolom yang mungkin belum ada
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]  # Urutkan sesuai fitur saat training

    # Scaling
    scaled = scaler.transform(df)

    # Prediksi cluster
    cluster = kmeans.predict(scaled)
    return int(cluster[0])
