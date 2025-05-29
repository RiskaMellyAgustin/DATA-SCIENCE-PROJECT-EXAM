import joblib
import pandas as pd
import numpy as np


class DifficultyPredictor:
    def __init__(self):
        self.model = joblib.load("jenyapp/static/jenyapp/model/best_model.pkl")
        self.label_encoder = joblib.load(
            "jenyapp/static/jenyapp/model/label_encoder.pkl"
        )
        self.features = joblib.load("jenyapp/static/jenyapp/model/feature_names.pkl")

    def _calculate_features(self, input_data):
        features = {
            "quiz_count": input_data["quiz_count"],
            "individual_assignment_count": input_data["individual_assignment_count"],
            "group_assignment_count": input_data["group_assignment_count"],
            "forum_count": input_data["forum_count"],
            "total_activities": (
                input_data["quiz_count"]
                + input_data["individual_assignment_count"]
                + input_data["group_assignment_count"]
            ),
            "quiz_ratio": input_data["quiz_count"]
            / (
                input_data["quiz_count"]
                + input_data["individual_assignment_count"]
                + input_data["group_assignment_count"]
                + 1e-6
            ),
            "assignment_ratio": (
                input_data["individual_assignment_count"]
                + input_data["group_assignment_count"]
            )
            / (
                input_data["quiz_count"]
                + input_data["individual_assignment_count"]
                + input_data["group_assignment_count"]
                + 1e-6
            ),
            "engagement_score": (
                input_data["quiz_count"] * 0.3
                + input_data["individual_assignment_count"] * 0.4
                + input_data["group_assignment_count"] * 0.3
            ),
            "forum_activity_ratio": input_data["forum_count"]
            / (
                input_data["quiz_count"]
                + input_data["individual_assignment_count"]
                + input_data["group_assignment_count"]
                + 1
            ),
        }
        return features

    def predict(self, input_data):
        try:
            processed_features = self._calculate_features(input_data)

            input_df = pd.DataFrame([processed_features])[self.features]

            prediction = self.model.predict(input_df)

            return self.label_encoder.inverse_transform(prediction)[0]
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return None
