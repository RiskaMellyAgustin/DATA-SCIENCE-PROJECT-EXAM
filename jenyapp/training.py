import pandas as pd
import numpy as np
import joblib
import matplotlib
import seaborn as sns
import os
import matplotlib.pyplot as plt

matplotlib.use("Agg")
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)

    print("\n=== Dataset Information ===")
    print(f"Original shape: {data.shape}")
    print("\nFirst 5 rows:")
    print(data.head())

    # Handle missing values
    data = data.dropna()
    print(f"\nShape after dropping missing values: {data.shape}")

    # Remove leakage columns and irrelevant features
    columns_to_drop = [
        "stu_id",
        "course_id",
        "grade",
        "quiz_avg_duration",
        "individual_assignment_avg_duration",
        "group_assignment_avg_duration",
        "forum_avg_duration",
        "gender",
        "age",
    ]

    for col in columns_to_drop:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)
            print(f"Dropped column: {col}")

    print("\n=== Feature Engineering ===")
    # Activity ratios
    data["total_activities"] = (
        data["quiz_count"]
        + data["individual_assignment_count"]
        + data["group_assignment_count"]
    )
    data["quiz_ratio"] = data["quiz_count"] / (data["total_activities"] + 1)
    data["assignment_ratio"] = (
        data["individual_assignment_count"] + data["group_assignment_count"]
    ) / (data["total_activities"] + 1)

    # Engagement metrics
    data["engagement_score"] = (
        data["quiz_count"] * 0.3
        + data["individual_assignment_count"] * 0.4
        + data["group_assignment_count"] * 0.3
    )

    data["forum_activity_ratio"] = data["forum_count"] / (data["total_activities"] + 1)

    print(
        "Added new features: total_activities, quiz_ratio, assignment_ratio, engagement_score, forum_activity_ratio"
    )

    # Encode label
    le = LabelEncoder()
    data["difficulty"] = le.fit_transform(data["difficulty"])

    # Split features and target
    X = data.drop("difficulty", axis=1)
    y = data["difficulty"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("\nFeatures used:", X_train.columns.tolist())
    return X_train, X_test, y_train, y_test, le.classes_


def evaluate_model(model, X_test, y_test, classes, model_name="Model"):
    # Make predictions
    y_pred = model.predict(X_test)

    print(f"\n=== {model_name} Evaluation ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Weighted F1-score:", f1_score(y_test, y_pred, average="weighted"))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=classes, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(
        f"jenyapp/static/jenyapp/images/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    )
    plt.close()


def train_and_save_best_model():
    try:
        X_train, X_test, y_train, y_test, classes = load_and_preprocess_data(
            "jenyapp/static/jenyapp/data/data2.csv"
        )

        # Handle class imbalance with SMOTE
        print("\n=== Handling Class Imbalance with SMOTE ===")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(
            f"Class distribution after SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}"
        )

        # Best performing model / Optimized Random Forest
        print("\n=== Training Optimized Random Forest Model ===")
        rf_params = {
            "n_estimators": 300,
            "max_depth": 9,
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": 42,
            "n_jobs": -1,
        }

        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train_res, y_train_res)

        # Evaluate model
        evaluate_model(rf_model, X_test, y_test, classes, "Optimized Random Forest")

        # Feature Importance Analysis
        print("\n=== Feature Importance ===")
        importance = rf_model.feature_importances_
        feat_importances = pd.Series(importance, index=X_train.columns).sort_values(
            ascending=False
        )
        print("Top 5 most important features:")
        print(feat_importances.head(5))

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        feat_importances.plot(kind="barh")
        plt.title("Feature Importance - Random Forest")
        plt.savefig("jenyapp/static/jenyapp/images/feature_importance_rf.png")
        plt.close()

        # Save the model and related artifacts
        os.makedirs("jenyapp/static/jenyapp/model", exist_ok=True)

        # Save model
        model_path = "jenyapp/static/jenyapp/model/best_model.pkl"
        joblib.dump(rf_model, model_path)

        le = LabelEncoder()
        le.fit(y_train)
        joblib.dump(le, "jenyapp/static/jenyapp/model/label_encoder.pkl")

        joblib.dump(classes, "jenyapp/static/jenyapp/model/class_names.pkl")

        # Save feature names
        joblib.dump(
            X_train.columns.tolist(), "jenyapp/static/jenyapp/model/feature_names.pkl"
        )

        test_data = pd.concat([X_test, pd.Series(y_test, name="difficulty")], axis=1)
        test_data.to_csv("jenyapp/static/jenyapp/model/test_dataset.csv", index=False)
        print("\nTest dataset saved for external validation")

        print(f"\nModel saved successfully at {model_path}")

        with open("jenyapp/static/jenyapp/model/validation_report.txt", "w") as f:
            f.write("=== Model Validation Report ===\n\n")
            f.write("Model Type: Random Forest Classifier\n")
            f.write(f"Model Parameters: {rf_params}\n\n")

            f.write("=== Feature Importance ===\n")
            for feat, imp in zip(feat_importances.index, feat_importances.values):
                f.write(f"{feat}: {imp:.4f}\n")

            f.write("\n=== Model Performance ===\n")
            f.write(
                f"Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}\n"
            )
            f.write(
                f"Weighted F1-score: {f1_score(y_test, rf_model.predict(X_test), average='weighted'):.4f}\n"
            )

            f.write("\n=== Recommendations ===\n")
            f.write("1. Focus on improving data collection for 'Hard' class samples\n")
            f.write(
                "2. Consider adding time-on-task metrics for better engagement measurement\n"
            )
            f.write(
                "3. Track student progress throughout the course, not just final outcomes\n"
            )
            f.write("4. Consider adding prerequisite knowledge assessment data\n")

        print("\n=== Training Completed Successfully ===")

    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    train_and_save_best_model()
