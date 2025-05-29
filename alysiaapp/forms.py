from django import forms

class UserInputForm(forms.Form):
    ALGORITHM_CHOICES = [
        ('classification', 'Classification'),
        ('clustering', 'Clustering')
    ]

    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female')
    ]

    ACTIVITY_CHOICES = [
        ('quiz', 'Quiz'),
        ('individual assignment', 'Individual Assignment'),
        ('group assignment', 'Group Assignment'),
        ('forum', 'Forum')
    ]

    DAY_CHOICES = [
        (0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'),
        (3, 'Thursday'), (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday')
    ]

    algorithm = forms.ChoiceField(choices=ALGORITHM_CHOICES, label="Select Algorithm")
    gender = forms.ChoiceField(choices=GENDER_CHOICES, label="Gender")
    age = forms.IntegerField(label="Age")
    grade = forms.FloatField(label="Target Grade (Maksimum 100)")
    activity_type = forms.ChoiceField(choices=ACTIVITY_CHOICES, label="Activity Type")
    duration_minutes = forms.IntegerField(label="Study Duration Plan (in minutes)")
    day_of_week = forms.ChoiceField(choices=DAY_CHOICES, label="Target Day To do Study")
