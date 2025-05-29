# alysiaapp/ml_scripts/preprocessing.py
import pandas as pd

# Load data mentah
df = pd.read_csv("alysiaapp/ml_scripts/dataset.csv")

# Konversi kolom waktu
df['activity_start'] = pd.to_datetime(df['activity_start'], errors='coerce')
df['activity_end'] = pd.to_datetime(df['activity_end'], errors='coerce')

# Buang baris yang gagal konversi waktu
df = df.dropna(subset=['activity_start', 'activity_end'])

# Tambah fitur jam dan durasi
df['hour'] = df['activity_start'].dt.hour
df['duration'] = (df['activity_end'] - df['activity_start']).dt.total_seconds() / 60

# Label waktu
def waktu_kategori(h):
    if 3 <= h <= 10:
        return 'Morning'
    elif 11 <= h <= 17:
        return 'Afternoon'
    else:
        return 'Night'

df['waktu_label'] = df['hour'].apply(waktu_kategori)

# Simpan data bersih
df.to_csv("alysiaapp/ml_scripts/dataset_clean.csv", index=False)
print("✅ Preprocessing selesai dan file disimpan sebagai dataset_clean.csv")
