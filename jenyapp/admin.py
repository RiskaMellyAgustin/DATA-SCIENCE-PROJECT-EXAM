from django.contrib import admin
from .models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "course_id",
        "gender",
        "age",
        "predicted_difficulty",
        "created_at",
    )
    list_filter = ("course_id", "gender", "predicted_difficulty")
    search_fields = ("course_id", "recommendation")
    readonly_fields = ("created_at",)

    fieldsets = (
        (
            None,
            {
                "fields": (
                    "gender",
                    "age",
                    "course_id",
                    "quiz_count",
                    "quiz_avg_duration",
                    "individual_assignment_count",
                    "individual_assignment_avg_duration",
                    "group_assignment_count",
                    "group_assignment_avg_duration",
                    "forum_count",
                    "forum_avg_duration",
                    "predicted_difficulty",
                    "recommendation",
                    "created_at",
                )
            },
        ),
    )
