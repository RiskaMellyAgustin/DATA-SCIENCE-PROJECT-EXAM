🎓 Data Science Project - Group 3
Selamat datang di repository Data Science Project Group 3!
Project ini merupakan hasil kolaborasi tim dalam mata kuliah Advanced Database & Machine Learning.
Kami membuat sistem prediksi dan analisis data mahasiswa berbasis Django & Machine Learning, dengan database PostgreSQL.

🚀 Use Case

The platform includes multiple learning analytics use cases, such as:
1. lysia Dapyaraka
Best Study Time Prediction
alysiaapp: Helps students estimate the best study time based on previous grades and activity patterns. By analyzing inputs such as gender, age, target grade, study duration, preferred study time, and target activity, the system leverages clustering and classification algorithms to generate a personalized and more effective study schedule.

2. Jeny Fattahul
Predicting Course Difficulty & Recommendations
jenyapp: predicts course difficulty levels using classification algorithms based on student learning activity data, and recommends appropriate academic interventions or strategies to support both educators and students.

3.Riska Melly
Smart Dropout & Student Anomalies for Effective Learning Recommendations                   
mellyapp: This application focuses on early academic intervention by combining dropout risk prediction and anomaly detection based on student learning behavior.

4.Ruth Marsaulina
The effectiveness of student performance 
ruthapp: based on gender and each subject predicts whether male and/or female students will perform at a low, medium, or high level in each course. It also provides the percentage of activity types in each course, such as quizzes, forums, individual assignments, and group assignments.

5.Valencia Damanik
Student Segmentation & Personalized Support
valenciaapp: predicts student engagement segments and recommends appropriate academic strategies based on behavioral learning data. The system segments learners into Active, Balanced, or Passive categories using clustering and classification algorithms, and provides recommendations to support each segment’s learning needs.




🏗️ Tech Stack
🐍 Python 3.10

🌐 Django 4.x

🧠 scikit-learn, XGBoost, CatBoost

🗄️ PostgreSQL

🎨 TailwindCSS, Chart.js

🌎 HTML, CSS, JavaScript

📦 Cara Install & Jalankan Project
1️⃣ Clone Repository
bash
Copy
Edit
git clone https://github.com/RiskaMellyAgustin/DATA-SCIENCE-PROJECT-EXAM.git
cd DATA-SCIENCE-PROJECT-EXAM
2️⃣ Buat Virtual Environment
bash
Copy
Edit
python -m venv venv
3️⃣ Aktifkan Virtual Environment
Windows:

bash
Copy
Edit
venv\Scripts\activate
Mac/Linux:

bash
Copy
Edit
source venv/bin/activate
4️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
5️⃣ Setup Database (PostgreSQL)
Buat database di PostgreSQL (contoh: datascience_db)

Update settings.py:

python
Copy
Edit
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'datascience_db',
        'USER': 'postgres',
        'PASSWORD': 'password',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
Jalankan migrasi:

bash
Copy
Edit
python manage.py makemigrations
python manage.py migrate
6️⃣ Jalankan Server
bash
Copy
Edit
python manage.py runserver
7️⃣ Akses Aplikasi
Buka browser dan akses:

cpp
Copy
Edit
http://127.0.0.1:8000/
📊 Cara Menggunakan Aplikasi
🔎 Prediksi Dropout:
Masukkan ID Mahasiswa atau data manual untuk melihat hasil prediksi dropout.

📈 Prediksi Grade:
Isi data aktivitas mahasiswa dan dapatkan prediksi nilai akhir.

🧭 Anomaly Detection & Clustering:
Lihat analisis cluster di dashboard untuk mendeteksi potensi anomali.

🔐 Admin Page:
Admin dapat melihat seluruh data prediksi dan hasil analisis.

📂 Struktur Folder Penting
csharp
Copy
Edit
├───about
│   ├───migrations
│   │   └───__pycache__
│   └───__pycache__
├───adminapp
│   ├───migrations
│   │   └───__pycache__
│   └───__pycache__
├───alysiaapp
│   ├───migrations
│   │   └───__pycache__
│   ├───ml_scripts
│   │   ├───alysiaapp
│   │   │   └───ml_scripts
│   │   ├───prediksi
│   │   │   └───models
│   │   └───__pycache__
│   ├───static
│   │   └───alysiaapp
│   ├───templates
│   └───__pycache__
├───ds_project
│   └───__pycache__
├───home
│   ├───migrations
│   │   └───__pycache__
│   ├───static
│   │   └───home
│   ├───templates
│   │   └───home
│   └───__pycache__
├───jenyapp
│   ├───migrations
│   │   └───__pycache__
│   ├───static
│   │   └───jenyapp
│   │       ├───css
│   │       ├───data
│   │       ├───images
│   │       └───model
│   ├───templates
│   │   └───jenyapp
│   └───__pycache__
├───mellyapp
│   ├───migrations
│   │   └───__pycache__
│   ├───ml
│   │   ├───anomaly
│   │   ├───catboost_info
│   │   │   ├───learn
│   │   │   └───tmp
│   │   ├───clustering
│   │   ├───dropout
│   │   ├───Exist_Data
│   │   └───Training
│   ├───static
│   │   └───mellyapp
│   │       └───images
│   ├───templates
│   │   └───mellyapp
│   └───__pycache__
├───ml
│   ├───migrations
│   │   └───__pycache__
│   └───__pycache__
├───pkl_file
├───ruthapp
│   ├───migrations
│   │   └───__pycache__
│   ├───ml
│   │   ├───ml_model
│   │   └───ruthapp
│   │       └───ml
│   ├───templates
│   │   └───ruthapp
│   └───__pycache__
├───templates
│   └───admin
└───valenciaapp
    ├───migrations
    │   └───__pycache__
    ├───static
    │   └───img
    ├───templates
    └───__pycache__


👥 Tim Kami
💡 Riska Melly Agustin

💡 Alysia Dapyaraka

💡 Jeny Fatahul

💡 Ruth Marshaulina

💡 Valencia Damanik

📄 License
MIT License

📝 Catatan Penting
File .pkl model ML dan dataset tidak diupload ke GitHub karena keterbatasan ukuran.
Jika Anda ingin menjalankan model secara lokal, silakan hubungi kami untuk mendapatkan file model.
