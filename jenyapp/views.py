import matplotlib.pyplot as plt
import os
import numpy as np
import math
from django.shortcuts import render
from .forms import PredictForm
from .predict import DifficultyPredictor
from matplotlib import ticker
from .models import Prediction


def format_hours(hours):
    total_minutes = int(round(hours * 60))
    hrs = total_minutes // 60
    mins = total_minutes % 60
    if hrs > 0 and mins > 0:
        return f"{hrs} hour{'s' if hrs > 1 else ''} {mins} minutes"
    elif hrs > 0:
        return f"{hrs} hour{'s' if hrs > 1 else ''}"
    else:
        return f"{mins} minutes"


def predict_view(request):
    prediction_result = None
    course_name = None
    recommendation_text = None
    bar_chart_filename = None
    recommendation_chart_filename = None
    time_estimation = None
    if request.method == "POST":
        form = PredictForm(request.POST)
        if form.is_valid():
            input_data = {
                "course_id": form.cleaned_data["course_id"],
                "quiz_count": form.cleaned_data["quiz_count"],
                "quiz_avg_duration": form.cleaned_data["quiz_avg_duration"],
                "individual_assignment_count": form.cleaned_data[
                    "individual_assignment_count"
                ],
                "individual_assignment_avg_duration": form.cleaned_data[
                    "individual_assignment_avg_duration"
                ],
                "group_assignment_count": form.cleaned_data["group_assignment_count"],
                "group_assignment_avg_duration": form.cleaned_data[
                    "group_assignment_avg_duration"
                ],
                "forum_count": form.cleaned_data["forum_count"],
                "forum_avg_duration": form.cleaned_data["forum_avg_duration"],
                "gender": form.cleaned_data["gender"],
                "age": form.cleaned_data["age"],
            }

            predictor = DifficultyPredictor()
            prediction_numeric = predictor.predict(input_data)

            label_map = {0: "Easy", 1: "Medium", 2: "Hard"}
            prediction_result = label_map.get(prediction_numeric, "Unknown")

            course_names = {
                1: "Course 1",
                2: "Course 2",
                3: "Course 3",
                4: "Course 4",
                5: "Course 5",
            }
            course_name = course_names.get(
                input_data["course_id"], f"Course {input_data['course_id']}"
            )

            course_length_weeks = 10

            total_time = (
                (input_data["quiz_count"] * input_data["quiz_avg_duration"])
                + (
                    input_data["individual_assignment_count"]
                    * input_data["individual_assignment_avg_duration"]
                )
                + (
                    input_data["group_assignment_count"]
                    * input_data["group_assignment_avg_duration"]
                )
                + (input_data["forum_count"] * input_data["forum_avg_duration"])
            ) / 60

            total_time_per_week = total_time / course_length_weeks

            difficulty_multiplier = {
                "Easy": 1.2,
                "Medium": 1.5,
                "Hard": 2.0,
            }

            recommended_time = total_time_per_week * difficulty_multiplier.get(
                prediction_result, 1.0
            )
            recommended_time = max(2, round(recommended_time))

            activity_breakdown = {
                "Quiz": (input_data["quiz_count"] * input_data["quiz_avg_duration"])
                / 60
                / course_length_weeks,
                "Individual Assignment": (
                    input_data["individual_assignment_count"]
                    * input_data["individual_assignment_avg_duration"]
                )
                / 60
                / course_length_weeks,
                "Group Assignment": (
                    input_data["group_assignment_count"]
                    * input_data["group_assignment_avg_duration"]
                )
                / 60
                / course_length_weeks,
                "Forum": (input_data["forum_count"] * input_data["forum_avg_duration"])
                / 60
                / course_length_weeks,
            }

            sorted_activities = sorted(
                activity_breakdown.items(), key=lambda x: x[1], reverse=True
            )

            vis_labels = [act[0] for act in sorted_activities]
            vis_values = [act[1] for act in sorted_activities]

            if prediction_result == "Easy":
                recommendation_text = (
                    f"Based on your course activities, you should allocate about {recommended_time} hours per week. "
                    f"Your time will mostly be spent on {sorted_activities[0][0]} {format_hours(sorted_activities[0][1])}. "
                    "Focus on consistent weekly review to maintain understanding."
                )

            elif prediction_result == "Medium":
                recommendation_text = (
                    f"This course has a plan for approximately {recommended_time} hours weekly. "
                    f"Prioritize {sorted_activities[0][0]} {format_hours(sorted_activities[0][1])} and "
                    f"{sorted_activities[1][0]} {format_hours(sorted_activities[1][1])}, but also manage other tasks wisely."
                )

            else:
                recommendation_text = (
                    f"This course requires significant time investment - at least {recommended_time} hours weekly. "
                    f"Focus on {sorted_activities[0][0]} {format_hours(sorted_activities[0][1])} and seek help early. "
                    "Don’t neglect the other components either."
                )

            time_estimation = {
                "total_required": round(total_time_per_week, 1),
                "recommended_with_difficulty": recommended_time,
                "activity_breakdown": {
                    k: format_hours(v) for k, v in activity_breakdown.items()
                },
            }

            # Visualization
            plt.figure(figsize=(7, 4))
            activity_counts = {
                "Quiz": input_data["quiz_count"],
                "Individual Assignment": input_data["individual_assignment_count"],
                "Group Assignment": input_data["group_assignment_count"],
                "Forum": input_data["forum_count"],
            }
            activity_counts_rounded = {
                k: int(round(v)) for k, v in activity_counts.items()
            }

            plt.bar(
                activity_counts_rounded.keys(),
                activity_counts_rounded.values(),
                color="#3498db",
            )
            plt.xlabel("Activity Type")
            plt.ylabel("Count")
            plt.yticks(range(0, max(activity_counts_rounded.values()) + 2))
            plt.tight_layout()

            bar_chart_filename = "jenyapp_bar_chart.png"
            chart_path = os.path.join(
                "jenyapp", "static", "jenyapp", "images", bar_chart_filename
            )
            plt.savefig(chart_path)
            plt.close()

            vis_values = [round(val * 60) for val in vis_values]

            plt.figure(figsize=(7, 4))
            plt.bar(vis_labels, vis_values, color="#3498db")
            plt.xlabel("Activity Type")
            plt.ylabel("Minutes per Week")
            ymax = math.ceil(max(vis_values) / 10) * 10 + 20
            plt.ylim(0, ymax)
            plt.tight_layout()

            recommendation_chart_filename = "jenyapp_recommendation_chart.png"
            recommendation_chart_path = os.path.join(
                "jenyapp", "static", "jenyapp", "images", recommendation_chart_filename
            )
            plt.savefig(recommendation_chart_path)
            plt.close()

            Prediction.objects.create(
                gender=input_data["gender"],
                age=input_data["age"],
                course_id=input_data["course_id"],
                quiz_count=input_data["quiz_count"],
                quiz_avg_duration=input_data["quiz_avg_duration"],
                individual_assignment_count=input_data["individual_assignment_count"],
                individual_assignment_avg_duration=input_data[
                    "individual_assignment_avg_duration"
                ],
                group_assignment_count=input_data["group_assignment_count"],
                group_assignment_avg_duration=input_data[
                    "group_assignment_avg_duration"
                ],
                forum_count=input_data["forum_count"],
                forum_avg_duration=input_data["forum_avg_duration"],
                predicted_difficulty=prediction_result,
                recommendation=recommendation_text,
            )
    else:
        form = PredictForm()

    return render(
        request,
        "jenyapp/predict.html",
        {
            "form": form,
            "prediction": prediction_result,
            "course_name": course_name,
            "recommendation": recommendation_text,
            "bar_chart_img": f"jenyapp/images/{bar_chart_filename}"
            if bar_chart_filename
            else None,
            "recommendation_chart_img": f"jenyapp/images/{recommendation_chart_filename}"
            if recommendation_chart_filename
            else None,
        },
    )
