import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
df = pd.read_csv('student_learning_features.csv')

# Define custom segmenting logic
def categorize_segment(row):
    if row['activity_count'] > 10 and row['total_study_time'] > 1500 and row['grade_mean'] > 70:
        return 'Active Learners'
    elif row['activity_count'] < 5 and row['total_study_time'] < 500 and row['grade_mean'] < 50:
        return 'Passive Learners'
    else:
        return 'Balanced Learners'

df['segment_label'] = df.apply(categorize_segment, axis=1)

# Encode labels
label_encoder = LabelEncoder()
df['segment_label_encoded'] = label_encoder.fit_transform(df['segment_label'])

# Feature and label selection
X = df.drop(['stu_id', 'segment_label', 'segment_label_encoded', 'dominant_activity_type'], axis=1)
y = df['segment_label_encoded']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model and tools
save_dir = 'ds_project/valenciaapp'
os.makedirs(save_dir, exist_ok=True)

joblib.dump(model, os.path.join(save_dir, 'xgboost_student_segment_model.pkl'))
joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
joblib.dump(label_encoder, os.path.join(save_dir, 'label_encoder.pkl'))
joblib.dump(list(X.columns), os.path.join(save_dir, 'feature_names.pkl'))
