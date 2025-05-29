import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load data
df = pd.read_csv('D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/dropout/DATASET DROPOUT 4.csv')

# 2. Fitur dan target
X = df.drop(columns=['dropout', 'grade'])
y = df['dropout']

# 3. Kolom kategorikal dan numerikal
categorical_features = ['gender']
numerical_features = ['age', 'total_activities_done', 'total_duration_minutes']

# 4. Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first'), categorical_features),
    ('num', StandardScaler(), numerical_features)
])

# 5. Pipeline model dengan RandomForestClassifier
pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('clf', RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ))
])

# 6. Split train-test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 7. Fit model
pipeline.fit(X_train, y_train)

# 8. Predict dan evaluasi
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 9. Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy (5-fold): {cv_scores.mean():.4f}")

# 10. Simpan model
model_path = 'D:/1.UNI 5th Semester/6. Advance DataBase/ADVANCE_DB PROJECT/ds_project/mellyapp/ml/dropout/randomforest_dropout_model.pkl'
joblib.dump(pipeline, model_path)
print(f"✅ Model berhasil disimpan ke: {model_path}")


