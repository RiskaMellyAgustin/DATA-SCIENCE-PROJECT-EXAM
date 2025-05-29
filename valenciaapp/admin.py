from django.contrib import admin
from .models import StudentPrediction

@admin.register(StudentPrediction)
class StudentPredictionAdmin(admin.ModelAdmin):
    list_display = ('timestamp', 'algorithm', 'predicted_segment', 'activity_count', 'login_days', 'grade_mean')
    list_filter = ('algorithm', 'predicted_segment', 'timestamp')
    search_fields = ('predicted_segment',)
    ordering = ('-timestamp',)
