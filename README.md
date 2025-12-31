# Sistem Temu Balik Informasi (IR) Sederhana

Aplikasi ini adalah implementasi sistem **Information Retrieval** sederhana menggunakan metode **Cosine Similarity** dan **TF-IDF**. Dibangun menggunakan Python dan Streamlit.

## Fitur
- **Indexing Dokumen**: Menggunakan TF-IDF untuk mengubah teks menjadi vektor.
- **Pencarian**: Menghitung kemiripan antara kueri pengguna dan dokumen menggunakan Cosine Similarity.
- **Antarmuka Pengguna**: Web UI interaktif dan responsif menggunakan Streamlit.

## Teknologi
- **Python 3.x**
- **Streamlit**: Untuk antarmuka web.
- **Scikit-Learn**: Untuk algoritma TF-IDF dan Cosine Similarity.
- **Pandas**: Untuk manipulasi data sederhana.

## Cara Menjalankan

### 1. Persiapan Lingkungan
Pastikan Python sudah terinstal. Disarankan menggunakan virtual environment.

### 2. Instalasi Dependensi
Jalankan perintah berikut di terminal:
```bash
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
Jalankan aplikasi dengan perintah:
```bash
streamlit run app.py
```
Aplikasi akan otomatis terbuka di browser default Anda (biasanya di http://localhost:8501).

## Struktur File
- `app.py`: Kode utama aplikasi.
- `requirements.txt`: Daftar pustaka yang dibutuhkan.
- `README.md`: Dokumentasi proyek.
