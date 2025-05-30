from django.db import models


class Prediction(models.Model):
    class Meta:
        db_table = 'jenyapp_prediction'
        
    GENDER_CHOICES = (
        ("M", "Male"),
        ("F", "Female"),
    )

    gender = models.CharField(max_length=1, choices=GENDER_CHOICES)
    age = models.IntegerField()
    course_id = models.IntegerField()
    quiz_count = models.IntegerField()
    quiz_avg_duration = models.IntegerField()
    individual_assignment_count = models.IntegerField()
    individual_assignment_avg_duration = models.IntegerField()
    group_assignment_count = models.IntegerField()
    group_assignment_avg_duration = models.IntegerField()
    forum_count = models.IntegerField()
    forum_avg_duration = models.IntegerField()
    predicted_difficulty = models.CharField(max_length=10)
    recommendation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction for Course {self.course_id} ({self.predicted_difficulty})"
