from django.db import models

class DropoutPredictionLog(models.Model):
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    total_activities_done = models.IntegerField()
    total_duration_minutes = models.FloatField()
    prediction = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gender} - {self.age} - {self.prediction}"

class AnomalyDetectionLog(models.Model):
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    total_activities_done = models.IntegerField()
    total_duration_minutes = models.FloatField()
    anomaly_score = models.FloatField()
    prediction = models.CharField(max_length=10)  # Normal / Anomali
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gender} - {self.prediction} - {self.anomaly_score:.2f}"

class DropoutRegressionLog(models.Model):
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    total_activities_done = models.IntegerField()
    total_duration_minutes = models.FloatField()
    dropout_score = models.FloatField()
    prediction = models.CharField(max_length=10)  # Dropout / Aktif
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gender} - {self.dropout_score:.2f} - {self.prediction}"

class ClusterAnomalyLog(models.Model):
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    total_activities_done = models.IntegerField()
    total_duration_minutes = models.FloatField()
    cluster_id = models.IntegerField()
    cluster_label = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gender} - Cluster {self.cluster_id} - {self.cluster_label}"
    
class DropoutExistLog(models.Model):
    gender = models.CharField(max_length=10)
    age = models.IntegerField()
    total_activities_done = models.IntegerField()
    total_duration_minutes = models.FloatField()
    prediction = models.CharField(max_length=10)  # Dropout / Active
    probability = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.gender} - {self.age} - {self.prediction}"


# models.py
from django.db import models
from datetime import date

class Student(models.Model):
    stu_id = models.CharField(max_length=20, primary_key=True)
    name = models.CharField(max_length=100)
    email = models.EmailField()
    gender = models.CharField(max_length=10)
    dob = models.DateField()

    def get_age(self):
        today = date.today()
        return today.year - self.dob.year - ((today.month, today.day) < (self.dob.month, self.dob.day))

# Model Enrollment
class Enrollment(models.Model):
    enroll_id = models.AutoField(primary_key=True)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course_id = models.CharField(max_length=20)  # Bisa juga FK ke Course model kalau ada
    grade = models.FloatField()

    def __str__(self):
        return f"{self.student.student_id} - {self.course_id}"


# Model StudentActivityLog
class StudentActivityLog(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    activity_id = models.CharField(max_length=50)  # Bisa juga FK ke CourseActivity
    activity_start = models.DateTimeField()
    activity_end = models.DateTimeField()

    def __str__(self):
        return f"{self.student.student_id} - {self.activity_id}"


