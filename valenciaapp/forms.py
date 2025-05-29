from django import forms

class DataInputForm(forms.Form):
    activity_count = forms.IntegerField(label="Activity Count", min_value=1, max_value=50)
    login_days = forms.IntegerField(label="Login Days", min_value=1, max_value=30)
    total_study_time = forms.IntegerField(label="Total Study Time (minutes)", min_value=10, max_value=600)
    avg_time_per_activity = forms.IntegerField(label="Average Time per Activity (minutes)", min_value=5, max_value=120)
    avg_active_hour = forms.IntegerField(label="Average Active Hour", min_value=0, max_value=23)
    grade_mean = forms.IntegerField(label="Grade Mean", min_value=0, max_value=100)
    grade_stddev = forms.IntegerField(label="Grade Standard Deviation", min_value=0, max_value=20)

    dominant_activity_type_forum = forms.BooleanField(required=False, label="Forum")
    dominant_activity_type_group_assignment = forms.BooleanField(required=False, label="Group Assignment")
    dominant_activity_type_individual_assignment = forms.BooleanField(required=False, label="Individual Assignment")

    # Tambahan pilihan algoritma clustering
    ALGO_CHOICES = [
        ('xgboost', 'XGBoost'),
        ('kmeans', 'K-Means'),
    ]
    algorithm = forms.ChoiceField(label="Select Algorithm", choices=ALGO_CHOICES, required=True)
