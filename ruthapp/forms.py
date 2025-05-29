from django import forms

class EffectivenessForm(forms.Form):
    gender = forms.ChoiceField(choices=[('Male', 'Male'), ('Female', 'Female')])
    age = forms.IntegerField(min_value=15)
    course_name = forms.CharField(max_length=100)
    activity_type = forms.ChoiceField(choices=[
        ('Quiz', 'Quiz'),
        ('Forum', 'Forum'),
        ('Assignment', 'Assignment'),
    ])
    total_activities = forms.IntegerField(min_value=0)
    total_duration_minutes = forms.FloatField(min_value=0)
