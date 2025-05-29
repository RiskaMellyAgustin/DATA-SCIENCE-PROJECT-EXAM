from django import forms

COURSE_CHOICES = [
    ("1", "Course 1"),
    ("2", "Course 2"),
    ("3", "Course 3"),
    ("4", "Course 4"),
    ("5", "Course 5"),
]

GENDER_CHOICES = [
    ("M", "Male"),
    ("F", "Female"),
]


class PredictForm(forms.Form):
    gender = forms.ChoiceField(
        choices=[("", "----- Select Gender -----")] + GENDER_CHOICES,
        label="Gender",
        required=True,
        widget=forms.Select(),
    )

    age = forms.IntegerField(
        min_value=15,
        max_value=80,
        label="Age",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 20"}),
    )

    course_id = forms.ChoiceField(
        choices=[("", "----- Select Course -----")] + COURSE_CHOICES,
        label="Course",
        required=True,
        widget=forms.Select(),
    )

    quiz_count = forms.IntegerField(
        min_value=0,
        label="Quiz Count",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 5"}),
    )
    quiz_avg_duration = forms.IntegerField(
        min_value=0,
        label="Quiz Avg Duration (minutes)",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 15"}),
    )

    individual_assignment_count = forms.IntegerField(
        min_value=0,
        label="Individual Assignment Count",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 3"}),
    )
    individual_assignment_avg_duration = forms.IntegerField(
        min_value=0,
        label="Individual Assignment Avg Duration (minutes)",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 60"}),
    )

    group_assignment_count = forms.IntegerField(
        min_value=0,
        label="Group Assignment Count",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 2"}),
    )
    group_assignment_avg_duration = forms.IntegerField(
        min_value=0,
        label="Group Assignment Avg Duration (minutes)",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 120"}),
    )

    forum_count = forms.IntegerField(
        min_value=0,
        label="Forum Count",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 4"}),
    )
    forum_avg_duration = forms.IntegerField(
        min_value=0,
        label="Forum Avg Duration (minutes)",
        widget=forms.NumberInput(attrs={"placeholder": "e.g. 10"}),
    )
