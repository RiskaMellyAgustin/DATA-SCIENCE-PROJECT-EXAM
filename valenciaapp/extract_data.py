import pandas as pd
from sqlalchemy import create_engine


DB_USER = 'postgres'
DB_PASS = 'admin'
DB_HOST = 'localhost'
DB_PORT = '5432'
DB_NAME = 'advanced_db_project'

# Koneksi ke PostgreSQL
engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

# Query to extrac student learning features
query = """
SELECT
    s.stu_id,
    COUNT(DISTINCT sal.activity_id) AS activity_count,
    COUNT(DISTINCT DATE(sal.activity_start)) AS login_days,
    SUM(EXTRACT(EPOCH FROM (sal.activity_end - sal.activity_start))/60) AS total_study_time,
    AVG(EXTRACT(EPOCH FROM (sal.activity_end - sal.activity_start))/60) AS avg_time_per_activity,
    AVG(EXTRACT(HOUR FROM sal.activity_start)) AS avg_active_hour,
    AVG(e.grade) AS grade_mean,
    STDDEV(e.grade) AS grade_stddev,
    (
        SELECT at.type_name
        FROM course_activity ca
        JOIN activity_type at ON ca.type_id = at.type_id
        JOIN student_activity_log sal2 ON sal2.activity_id = ca.activity_id AND sal2.stu_id = s.stu_id
        GROUP BY at.type_name
        ORDER BY COUNT(*) DESC
        LIMIT 1
    ) AS dominant_activity_type
FROM student s
LEFT JOIN student_activity_log sal ON s.stu_id = sal.stu_id
LEFT JOIN enrollment e ON s.stu_id = e.stu_id
GROUP BY s.stu_id;
"""

# Eksekusi dan simpan ke DataFrame
df = pd.read_sql(query, engine)

# Simpan ke CSV untuk eksplorasi lanjut (opsional)
df.to_csv('student_learning_features.csv', index=False)

# Lihat data
print(df.head())
