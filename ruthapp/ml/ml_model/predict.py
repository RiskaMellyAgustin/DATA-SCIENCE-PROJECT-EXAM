import pandas as pd
import joblib
import psycopg2
import plotly.express as px
import matplotlib.pyplot as plt

# Load model dan encoders
model = joblib.load("student_performance_model.pkl")
le_gender = joblib.load("label_encoder_gender.pkl")
le_course = joblib.load("label_encoder_course.pkl")
le_perf = joblib.load("label_encoder_performance.pkl")

# === INPUT USER ===
input_gender = "Female"
input_course = 2
input_activities_completed = 10
input_avg_duration = 60

# Encode input
encoded_gender = le_gender.transform([input_gender])[0]
encoded_course = le_course.transform([input_course])[0]

# Buat data input
X_input = [[encoded_gender, encoded_course, input_activities_completed, input_avg_duration]]

# Prediksi performa
pred = model.predict(X_input)[0]
pred_proba = model.predict_proba(X_input)[0]
pred_label = le_perf.inverse_transform([pred])[0]

print("🎯 Predicted Performance:", pred_label)
for i, label in enumerate(le_perf.classes_):
    print(f"• {label}: {pred_proba[i]*100:.2f}%")

# === PIE CHART DISTRIBUSI PERFORMA BERDASARKAN TIPE AKTIVITAS ===

# Koneksi PostgreSQL
conn = psycopg2.connect(
    dbname='advanced_project',
    user='postgres',
    password='admin123',
    host='localhost',
    port='5432'
)

# Ambil data performa per tipe aktivitas untuk course tertentu
query = f"""
SELECT 
    at.activity_type_name,
    COUNT(sal.activity_id) AS total_activities,
    AVG(e.grade) AS avg_grade
FROM student_activity_log sal
JOIN course_activity ca ON sal.activity_id = ca.activity_id
JOIN activitytype at ON ca.activity_type_id = at.activity_type_id
JOIN enrollment e ON sal.stu_id = e.stu_id AND ca.course_id = e.course_id
WHERE ca.course_id = {input_course}
GROUP BY at.activity_type_name
"""

df_activity = pd.read_sql_query(query, conn)
conn.close()

# Hitung performa berdasarkan rata-rata grade (High/Medium/Low)
def grade_to_perf(g):
    if g >= 75:
        return "High"
    elif g >= 60:
        return "Medium"
    return "Low"

df_activity["performance"] = df_activity["avg_grade"].apply(grade_to_perf)

# Hitung proporsi performa berdasarkan tipe aktivitas
pie_data = df_activity.groupby("performance")["total_activities"].sum().reset_index()

# Pie chart
fig = px.pie(
    pie_data, 
    names="performance", 
    values="total_activities", 
    title=f"Performance Distribution by Activity Type (Course {input_course})",
    color="performance",
    color_discrete_map={"High":"green", "Medium":"orange", "Low":"red"}
)
fig.show()
