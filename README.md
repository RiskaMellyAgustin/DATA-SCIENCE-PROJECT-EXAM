🎓 Data Science Project - Group 3
Selamat datang di repository Data Science Project Group 3!
Project ini merupakan hasil kolaborasi tim dalam mata kuliah Advanced Database & Machine Learning.
Kami membuat sistem prediksi dan analisis data mahasiswa berbasis Django & Machine Learning, dengan database PostgreSQL.

🚀 Fitur Aplikasi
✅ Prediksi Dropout Mahasiswa

Menggunakan model XGBoost & Random Forest

Prediksi berdasarkan aktivitas mahasiswa di LMS

Mode: Input manual & input data existing

✅ Prediksi Nilai Akhir (Grade Prediction)

Model regresi: XGBoost

Input: gender, umur, aktivitas, durasi, dll

✅ Anomaly Detection

Algoritma: Isolation Forest

Analisis mahasiswa dengan potensi anomali

✅ Clustering Mahasiswa (Anomaly Clustering)

Algoritma: KMeans

Kelompokkan mahasiswa dalam cluster "Normal", "Potensi Risiko", "Anomali Tinggi"

✅ Dashboard Statistik & Visualisasi

Total pendapatan, jumlah user aktif, recent subscription

Grafik interaktif (Chart.js)

✅ Manajemen User (Admin & Customer)

Role-based access: Admin & Customer

Admin dapat melihat hasil prediksi semua mahasiswa

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
├── customerapp/
├── adminapp/
├── mellyapp/             # Modul Machine Learning
│   ├── ml/               # Model .pkl disimpan di sini
│   ├── views.py
│   ├── forms.py
│   ├── ...
├── templates/
│   ├── index.html
│   ├── dashboard.html
│   └── ...
├── static/
│   ├── css/
│   ├── js/
├── manage.py
├── requirements.txt
├── README.md
└── ...
👥 Tim Kami
💡 Riska Melly Agustin

💡 Alysia

💡 Melly

💡 Ruth

💡 Jeny

💡 Valencia

📄 License
MIT License

📝 Catatan Penting
File .pkl model ML dan dataset tidak diupload ke GitHub karena keterbatasan ukuran.
Jika Anda ingin menjalankan model secara lokal, silakan hubungi kami untuk mendapatkan file model.