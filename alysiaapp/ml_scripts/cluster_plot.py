# ml_scripts/cluster_plot.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import os

def generate_cluster_plot(input_point, save_path="static/cluster_plot.png"):
    # Load model clustering dan PCA
    model_path = os.path.join(os.path.dirname(__file__), "clustering_model.pkl")
    pca_path = os.path.join(os.path.dirname(__file__), "pca_model.pkl")
    data_path = os.path.join(os.path.dirname(__file__), "dataset_cleaning.csv")

    model = joblib.load(model_path)
    pca = joblib.load(pca_path)
    df = pd.read_csv(data_path)

    # Ambil fitur yang sama seperti saat training
    features = ['gender', 'age', 'grade', 'activity_type', 'duration_minutes', 'day_of_week']
    df_plot = df[features].copy()

    # One-hot encoding atau label encoding jika diperlukan
    df_plot_encoded = pd.get_dummies(df_plot)
    input_df = pd.DataFrame([input_point])
    input_encoded = pd.get_dummies(input_df)

    # Samakan kolom input agar sesuai dengan data pelatihan
    input_encoded = input_encoded.reindex(columns=df_plot_encoded.columns, fill_value=0)

    # PCA transform
    df_pca = pca.transform(df_plot_encoded)
    input_pca = pca.transform(input_encoded)

    # Tambah label cluster
    cluster_labels = model.predict(df_plot_encoded)
    df_plot_pca = pd.DataFrame(df_pca, columns=["PC1", "PC2"])
    df_plot_pca["cluster"] = cluster_labels

    # Plot hasil
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_plot_pca, x="PC1", y="PC2", hue="cluster", palette="tab10", alpha=0.6)
    plt.scatter(input_pca[0, 0], input_pca[0, 1], color='red', s=120, marker='X', label='Input User')
    plt.title("Visualisasi Clustering dan Input")
    plt.legend()
    plt.tight_layout()

    # Simpan gambar ke folder static
    os.makedirs("static", exist_ok=True)
    plt.savefig(save_path)
    plt.close()
