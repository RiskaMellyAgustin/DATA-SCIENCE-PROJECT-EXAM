from django.db import models

class StudentPrediction(models.Model):
    activity_count = models.IntegerField()
    login_days = models.IntegerField()
    total_study_time = models.IntegerField()
    avg_time_per_activity = models.IntegerField()
    avg_active_hour = models.IntegerField()
    grade_mean = models.IntegerField()
    grade_stddev = models.IntegerField()

    dominant_activity_type_forum = models.BooleanField(default=False)
    dominant_activity_type_group_assignment = models.BooleanField(default=False)
    dominant_activity_type_individual_assignment = models.BooleanField(default=False)

    algorithm = models.CharField(max_length=20, choices=[('xgboost', 'XGBoost'), ('kmeans', 'K-Means')])
    predicted_segment = models.CharField(max_length=50)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.predicted_segment}"
