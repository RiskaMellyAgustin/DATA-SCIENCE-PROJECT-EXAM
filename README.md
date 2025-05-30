# 🎓 Data Science Project - Group 3

Selamat datang di repository **Data Science Project Group 3**!  
Project ini merupakan hasil kolaborasi tim dalam mata kuliah *Advanced Database & Machine Learning*.

Kami membangun sistem prediksi dan analisis data mahasiswa berbasis **Django** & **Machine Learning**, dengan database **PostgreSQL**.

---
## Demo

![Demo GIF](https://github.com/RiskaMellyAgustin/DATA-SCIENCE-PROJECT-EXAM/raw/main/ScreenRecorder-FINPRO-ADBmp4-ezgif.com-crop%20(1).gif)

## 🚀 Use Cases

### 1. **Alysia Dapyaraka – Best Study Time Prediction**
📱 `alysiaapp`  
Membantu mahasiswa memperkirakan waktu belajar terbaik berdasarkan nilai sebelumnya dan pola aktivitas.  
Input: gender, umur, target nilai, durasi belajar, waktu belajar favorit, target aktivitas.  
🔍 *Clustering & Classification* digunakan untuk menghasilkan jadwal belajar personal dan efektif.

### 2. **Jeny Fatahul – Predicting Course Difficulty & Recommendations**
📱 `jenyapp`  
Memprediksi tingkat kesulitan mata kuliah dengan algoritma klasifikasi berdasarkan aktivitas belajar mahasiswa.  
Memberikan rekomendasi intervensi akademik untuk mendukung dosen dan mahasiswa.

### 3. **Riska Melly – Smart Dropout & Student Anomalies**
📱 `mellyapp`  
Aplikasi untuk intervensi dini dengan memprediksi risiko dropout dan mendeteksi anomali berdasarkan perilaku belajar mahasiswa.

### 4. **Ruth Marsaulina – Effectiveness of Student Performance**
📱 `ruthapp`  
Memprediksi performa berdasarkan gender di setiap mata kuliah: rendah, sedang, atau tinggi.  
Menampilkan persentase aktivitas seperti kuis, forum, tugas individu, dan tugas kelompok.

### 5. **Valencia Damanik – Student Segmentation & Personalized Support**
📱 `valenciaapp`  
Memprediksi segmentasi keterlibatan mahasiswa (Aktif, Seimbang, atau Pasif).  
Memberikan rekomendasi strategi belajar yang sesuai berdasarkan data perilaku.

---

## 🏗️ Tech Stack

- 🐍 Python 3.10  
- 🌐 Django 4.x  
- 🧠 scikit-learn, XGBoost, CatBoost  
- 🗄️ PostgreSQL  
- 🎨 TailwindCSS, Chart.js  
- 🌎 HTML, CSS, JavaScript  

---

## 📦 Cara Install & Jalankan Project

### 1️⃣ Clone Repository
```bash
git clone https://github.com/RiskaMellyAgustin/DATA-SCIENCE-PROJECT-EXAM.git
cd DATA-SCIENCE-PROJECT-EXAM

### 2️⃣ Buat Virtual Environment
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
Buat database di PostgreSQL, contoh: datascience_db

Update settings.py:
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
Buka browser dan kunjungi:
http://127.0.0.1:8000/

📊 Fitur Aplikasi
🔎 Prediksi Dropout: Masukkan ID mahasiswa atau data manual untuk melihat hasil prediksi dropout.

📈 Prediksi Grade: Masukkan data aktivitas belajar untuk memprediksi nilai akhir.

🧭 Anomaly Detection & Clustering: Analisis data dan visualisasi cluster untuk mendeteksi anomali.

🔐 Admin Page: Melihat semua data dan hasil analisis prediktif.

📂 Struktur Folder Penting
<details> <summary>Klik untuk lihat struktur folder</summary>
swift
Copy
Edit
├── about/
├── adminapp/
├── alysiaapp/
│   ├── ml_scripts/
├── ds_project/
├── home/
├── jenyapp/
│   └── static/jenyapp/{css, data, images, model}
├── mellyapp/
│   └── ml/{anomaly, dropout, clustering, catboost_info, ...}
├── ml/
├── pkl_file/
├── ruthapp/
│   └── ml/ml_model/
├── templates/
│   └── admin/
└── valenciaapp/
</details>
👥 Tim Kami
💡 Riska Melly Agustin

💡 Alysia Dapyaraka

💡 Jeny Fatahul

💡 Ruth Marshaulina

💡 Valencia Damanik

📄 License
MIT License

📝 Catatan Penting
File model .pkl dan dataset tidak diunggah ke GitHub karena keterbatasan ukuran.
Jika Anda ingin menjalankan model secara lokal, silakan hubungi kami untuk mendapatkan file tersebut.


