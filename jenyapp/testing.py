import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold


def load_test_data(filepath):
    """Load dan preprocess data testing"""
    data = pd.read_csv(filepath)

    data = data.dropna()

    data["total_activities"] = (
        data["quiz_count"]
        + data["individual_assignment_count"]
        + data["group_assignment_count"]
    )
    data["engagement_score"] = (
        data["quiz_count"] * 0.3
        + data["individual_assignment_count"] * 0.4
        + data["group_assignment_count"] * 0.3
    )

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

    return data


def run_comprehensive_validation():
    try:
        model = joblib.load("jenyapp/static/jenyapp/model/best_model.pkl")
        label_encoder = joblib.load("jenyapp/static/jenyapp/model/label_encoder.pkl")
        feature_names = joblib.load("jenyapp/static/jenyapp/model/feature_names.pkl")
        class_names = joblib.load("jenyapp/static/jenyapp/model/class_names.pkl")

        test_data = pd.read_csv("jenyapp/static/jenyapp/model/test_dataset.csv")
        X_test = test_data[feature_names]
        y_test = test_data["difficulty"]

        print("\n=== Basic Evaluation ===")
        y_pred = model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Weighted F1:", f1_score(y_test, y_pred, average="weighted"))

        print("\n=== Cross Validation ===")
        cv = StratifiedKFold(n_splits=5)
        cv_scores = cross_val_score(model, X_test, y_test, cv=cv, scoring="accuracy")
        print(f"CV Scores: {cv_scores}")
        print(f"Mean CV Accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

        # Classification Report
        print("\n=== Detailed Classification Report ===")
        target_names = [str(cls) for cls in class_names]
        print(
            classification_report(
                y_test, y_pred, target_names=target_names, zero_division=0
            )
        )

        # Confusion Matrix Visualization
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names
        )
        plt.title("Confusion Matrix - Final Validation")
        plt.savefig("jenyapp/static/jenyapp/images/final_confusion_matrix.png")
        plt.close()

        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr")
                print(f"\nROC AUC Score: {roc_auc:.4f}")
            else:
                print("\nModel doesn't support probability predictions")
        except Exception as e:
            print(f"\nROC AUC calculation failed: {str(e)}")

        with open("jenyapp/static/jenyapp/model/validation_results.txt", "w") as f:
            f.write("=== Final Validation Results ===\n\n")
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
            f.write(
                f"Weighted F1: {f1_score(y_test, y_pred, average='weighted'):.4f}\n"
            )
            f.write(f"Mean CV Accuracy: {np.mean(cv_scores):.4f}\n")
            f.write("\n=== Classification Report ===\n")
            f.write(
                classification_report(
                    y_test, y_pred, target_names=target_names, zero_division=0
                )
            )

        print("\n=== Validation Completed ===")
        print("Results saved to validation_results.txt")

    except Exception as e:
        print(f"Error during validation: {str(e)}")
        raise


if __name__ == "__main__":
    run_comprehensive_validation()
