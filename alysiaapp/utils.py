import pandas as pd

def get_study_time_distribution():
    df = pd.read_csv('alysiaapp/ml_scripts/alysiaapp/ml_scripts/dataset_clean.csv')
    
    # Hitung distribusi label waktu
    distribusi = df['waktu_label'].value_counts().reindex(['Morning', 'Afternoon', 'Night'], fill_value=0)

    return {
        'labels': list(distribusi.index),
        'values': [int(v) for v in distribusi.values]  # Fix untuk JSON serialization
    }
