import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# === Fungsi untuk memetakan waktu ke kategori waktu belajar ===
def map_time_of_day(hour):
    if 3 <= hour <= 10:
        return 'Morning'
    elif 11 <= hour <= 17:
        return 'Afternoon'
    else:
        return 'Night'

# === Load dataset ===
df = pd.read_csv('alysiaapp/ml_scripts/dataset_clean.csv')

# === Label encoder untuk target ===
le_target = LabelEncoder()

# === Preprocessing tambahan ===
df['activity_start'] = pd.to_datetime(df['activity_start'])
df['activity_end'] = pd.to_datetime(df['activity_end'])
df['duration_minutes'] = (df['activity_end'] - df['activity_start']).dt.total_seconds() / 60
df['day_of_week'] = df['activity_start'].dt.dayofweek  # 0 = Senin
df['waktu_belajar'] = df['activity_start'].dt.hour.apply(map_time_of_day)
df['waktu_belajar'] = le_target.fit_transform(df['waktu_belajar'])

# === Pilih fitur dan target ===
features = ['gender', 'age', 'grade', 'activity_type', 'duration_minutes', 'day_of_week']
target = 'waktu_belajar'
X = df[features]
y = df[target]

# === Kolom kategorikal dan numerik ===
categorical_cols = ['gender', 'activity_type']
numeric_cols = ['age', 'grade', 'duration_minutes', 'day_of_week']

# === Preprocessing pipeline ===
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', 'passthrough', numeric_cols)
])

# === Model pipeline ===
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    learning_rate=0.05,
    n_estimators=700))
])



# === Train/test split & training ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# === Evaluasi ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi model:", accuracy)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# === Simpan model dan encoder ===
model_path = 'alysiaapp/ml_scripts/waktu_belajar_model.pkl'
encoder_path = 'alysiaapp/ml_scripts/label_encoder.pkl'

joblib.dump(model, model_path)
print(f"Model berhasil disimpan di: {model_path}")

joblib.dump(le_target, encoder_path)
print(f"Label encoder berhasil disimpan di: {encoder_path}")
