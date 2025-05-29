import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    numeric_cols = [
        "quiz_count",
        "individual_assignment_count",
        "group_assignment_count",
        "forum_count",
        "quiz_avg_duration",
        "individual_assignment_avg_duration",
        "group_assignment_avg_duration",
        "forum_avg_duration",
        "age",
    ]

    for col in numeric_cols:
        df[col] = df[col].replace("NULL", np.nan)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    count_cols = [
        "quiz_count",
        "individual_assignment_count",
        "group_assignment_count",
        "forum_count",
    ]
    duration_cols = [
        "quiz_avg_duration",
        "individual_assignment_avg_duration",
        "group_assignment_avg_duration",
        "forum_avg_duration",
    ]

    df[count_cols] = df[count_cols].fillna(0)

    for col in duration_cols:
        activity_type = col.split("_")[0]
        mean_val = df[df[col].notnull()][col].mean()
        df[col] = df[col].fillna(mean_val)

    gender_encoder = LabelEncoder()
    df["gender_encoded"] = gender_encoder.fit_transform(df["gender"])

    features = [
        "course_id",
        "quiz_count",
        "individual_assignment_count",
        "group_assignment_count",
        "forum_count",
        "quiz_avg_duration",
        "individual_assignment_avg_duration",
        "group_assignment_avg_duration",
        "forum_avg_duration",
        "gender_encoded",
        "age",
    ]
    target = "difficulty"

    X = df[features]
    y = df[target]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(
        gender_encoder, "ds_project/jenyapp/static/jenyapp/data/gender_encoder.pkl"
    )
    joblib.dump(
        label_encoder, "ds_project/jenyapp/static/jenyapp/data/label_encoder.pkl"
    )
    joblib.dump(scaler, "ds_project/jenyapp/static/jenyapp/data/scaler.pkl")

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder.classes_


if __name__ == "__main__":
    preprocess_data("ds_project/jenyapp/static/jenyapp/data/data2.csv")
