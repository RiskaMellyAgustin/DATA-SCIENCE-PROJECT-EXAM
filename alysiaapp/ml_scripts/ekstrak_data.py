# alysiaapp/ml_scripts/ekstrak_data.py

import pandas as pd
import psycopg2
import os

print("🚀 Memulai ekstraksi data...")

# Buat folder output
output_folder = "alysiaapp/ml_scripts"
os.makedirs(output_folder, exist_ok=True)

# Koneksi ke PostgreSQL
try:
    conn = psycopg2.connect(
        dbname='advanced_db_project',
        user='postgres',
        password='kebab123',
        host='localhost',
        port='5432'
    )
    print("✅ Koneksi ke database berhasil.")
except Exception as e:
    print("❌ Gagal koneksi ke database:", e)
    exit()

# Query SQL
query = """
    SELECT
        s.stu_id,
        s.gender,
        DATE_PART('year', AGE(s.dob)) AS age,
        e.course_id,
        e.grade,
        sal.activity_start,
        sal.activity_end,
        at.type_name AS activity_type,
        ca.activity_name AS activity_description
    FROM student s
    JOIN enrollment e ON s.stu_id = e.stu_id
    JOIN student_activity_log sal ON s.stu_id = sal.stu_id
    JOIN course_activity ca ON sal.activity_id = ca.activity_id
    JOIN activity_type at ON ca.type_id = at.type_id;
"""

try:
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.strip()

    output_path = os.path.join(output_folder, "dataset.csv")
    df.to_csv(output_path, index=False)

    print("📄 Kolom:", df.columns.tolist())
    print(f"✅ Data berhasil diekspor ke: {output_path}")
except Exception as e:
    print("❌ Gagal ekstrak data:", e)
