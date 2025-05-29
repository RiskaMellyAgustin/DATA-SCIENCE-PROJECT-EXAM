# import plotly.graph_objects as go
import plotly.io as pio
from django.db import connection
from django.shortcuts import render
import joblib
import psycopg2
import pandas as pd
import os
from django.conf import settings
import numpy as np
import plotly.graph_objects as go


# Get the base directory
BASE_DIR = settings.BASE_DIR

# Path to your model files
MODEL_DIR = os.path.join(BASE_DIR, 'ruthapp', 'ml', 'ml_model')

try:
    # Load model and encoders with full paths
    model = joblib.load(os.path.join(MODEL_DIR, 'student_performance_model.pkl'))
    le_gender = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_gender.pkl'))
    le_course = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_course.pkl'))
    le_perf = joblib.load(os.path.join(MODEL_DIR, 'label_encoder_performance.pkl'))
except Exception as e:
    raise Exception(f"Error loading ML models from {MODEL_DIR}: {str(e)}")

def predict_performance(gender, course, activities_completed=10, avg_duration=60):
    try:
        # Encode input
        encoded_gender = le_gender.transform([gender])[0]
        encoded_course = le_course.transform([course])[0]

        # Create input data
        X_input = [[encoded_gender, encoded_course, activities_completed, avg_duration]]

        # Predict performance
        pred = model.predict(X_input)[0]
        pred_proba = model.predict_proba(X_input)[0]
        
        # Get class order from label encoder
        classes = le_perf.classes_
        
        # Create probability dictionary according to class order
        proba_dict = {
            "High": round(pred_proba[np.where(classes == 'High')[0][0]] * 100, 2),
            "Medium": round(pred_proba[np.where(classes == 'Medium')[0][0]] * 100, 2),
            "Low": round(pred_proba[np.where(classes == 'Low')[0][0]] * 100, 2)
        }

        # Get prediction label
        pred_label = le_perf.inverse_transform([pred])[0]

        # Format average duration for display
        hours = int(avg_duration // 60)
        minutes = int(avg_duration % 60)
        duration_display = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"

        # Features information
        features = {
            "Gender": gender,
            "Course": course,
            "Total Activities": activities_completed,
            "Average Duration": duration_display,
            "RawDuration": avg_duration  # Keep original for any calculations
        }

        return pred_label, proba_dict, features
    
    except Exception as e:
        raise Exception(f"Prediction error: {str(e)}")

def get_activity_distribution(course_id):
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT 
                    at.type_name,
                    COUNT(sal.activity_id) AS total_activities
                FROM student_activity_log sal
                JOIN course_activity ca ON sal.activity_id = ca.activity_id
                JOIN activity_type at ON ca.type_id = at.type_id
                JOIN enrollment e ON sal.stu_id = e.stu_id AND ca.course_id = e.course_id
                WHERE ca.course_id = %s AND e.grade IS NOT NULL
                GROUP BY at.type_name
            """, [course_id])
            
            activity_data = cursor.fetchall()
            
            if not activity_data:
                return None
                
            # Convert to DataFrame for consistency
            df = pd.DataFrame(activity_data, columns=['activity_type', 'total_activities'])
            
            return df
            
    except Exception as e:
        print(f"Error getting activity distribution: {str(e)}")
        return None

def predict_view(request):
    if request.method == "POST":
        try:
            gender = request.POST["gender"]
            course = int(request.POST["course"])
            activities_completed = int(request.POST.get("activities_completed", 10))
            avg_duration = float(request.POST.get("avg_duration", 60))

            # Predict with model
            prediction, proba, features = predict_performance(
                gender, 
                course,
                activities_completed,
                avg_duration
            )

            # Get activity distribution data
            activity_data = get_activity_distribution(course)

            if activity_data is not None:
                fig = go.Figure(data=[go.Pie(
                    labels=activity_data['activity_type'],
                    values=activity_data['total_activities'],
                    hole=0.4,
                    textinfo='percent+label',
                    insidetextorientation='radial'
                )])
                fig.update_layout(
                    title=f"Activity Type Distribution for Course {course}",
                    showlegend=True
                )
                chart_html = pio.to_html(fig, full_html=False)
            else:
                chart_html = "<p><b>No activity data available for this course.</b></p>"

            return render(request, "predict_result.html", {
                "gender": gender,
                "course": course,
                "prediction": prediction,
                "proba": proba,
                "features": features,
                "chart_html": chart_html,
                "input_data": {  # Explicit input data for clarity
                    "activities_completed": activities_completed,
                    "avg_duration": avg_duration
                }
            })

        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            return render(request, "predict_form.html", {"error": error_msg})

    return render(request, "predict_form.html")