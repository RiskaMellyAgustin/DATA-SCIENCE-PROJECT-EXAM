

from django import forms

class DropoutPredictionForm(forms.Form):
    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
    ]

    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        label='Gender',
        widget=forms.Select(attrs={'class': 'w-full border-gray-300 rounded-lg p-2'})
    )

    age = forms.IntegerField(
        label='Age',
        widget=forms.NumberInput(attrs={'class': 'w-full border-gray-300 rounded-lg p-2', 'placeholder': 'e.g., 20'})
    )

    total_activities_done = forms.IntegerField(
        label='Total Activities Done',
        widget=forms.NumberInput(attrs={'class': 'w-full border-gray-300 rounded-lg p-2', 'placeholder': 'e.g., 15'})
    )

    total_duration_minutes = forms.FloatField(
        label='Total Duration (minutes)',
        widget=forms.NumberInput(attrs={'class': 'w-full border-gray-300 rounded-lg p-2', 'placeholder': 'e.g., 120.5'})
    )


# forms.py

from django import forms

class StudentSearchForm(forms.Form):
    student_id = forms.CharField(
        label='Student ID',
        max_length=20,
        widget=forms.TextInput(attrs={
            'class': 'w-full border-gray-300 rounded-lg p-2',
            'placeholder': 'Enter student ID, e.g., 1,2,3'
        })
    )

    


