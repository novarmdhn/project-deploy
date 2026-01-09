# 📚 Dokumentasi Sistem Temu Balik Informasi

> **Mata Kuliah:** Temu Balik Informasi (TBI)  
> **Deskripsi:** Dokumentasi lengkap alur logic dan arsitektur sistem pencarian dokumen menggunakan TF-IDF dan Cosine Similarity.

---

## 📋 Daftar Isi

1. [Gambaran Umum](#-gambaran-umum)
2. [Arsitektur Sistem](#-arsitektur-sistem)
3. [Alur Logic Detail](#-alur-logic-detail)
4. [Komponen Utama](#-komponen-utama)
5. [Diagram Alur](#-diagram-alur)
6. [Penjelasan Algoritma](#-penjelasan-algoritma)

---

## 🎯 Gambaran Umum

Sistem ini merupakan implementasi **Information Retrieval (IR)** sederhana yang memungkinkan pengguna mencari dokumen berdasarkan kata kunci. Sistem menggunakan:

| Komponen | Teknologi |
|----------|-----------|
| **Framework UI** | Streamlit |
| **Algoritma Indexing** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Algoritma Similarity** | Cosine Similarity |
| **Library Utama** | Scikit-Learn, Pandas |

---

## 🏗 Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT WEB APP                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │   INPUT     │───▶│  PROCESSING  │───▶│   OUTPUT     │  │
│   │   (Query)   │    │  (Matching)  │    │  (Results)   │  │
│   └─────────────┘    └──────────────┘    └──────────────┘  │
│                             │                               │
│                             ▼                               │
│                    ┌──────────────┐                        │
│                    │  TF-IDF &    │                        │
│                    │  Cosine Sim  │                        │
│                    └──────────────┘                        │
│                             │                               │
│                             ▼                               │
│                    ┌──────────────┐                        │
│                    │   CORPUS     │                        │
│                    │  (Dokumen)   │                        │
│                    └──────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Alur Logic Detail

### **Fase 1: Inisialisasi Sistem**

```
START
  │
  ▼
┌─────────────────────────────┐
│ 1. Load Library & Module    │
│    - streamlit              │
│    - sklearn                │
│    - pandas                 │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 2. Konfigurasi Streamlit    │
│    - page_title             │
│    - page_icon              │
│    - layout                 │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 3. Load Dataset Dokumen     │
│    - id, title, content     │
│    - Total: 5 dokumen       │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 4. Build TF-IDF Index       │
│    - create_tfidf_index()   │
│    - @st.cache_resource     │
└─────────────────────────────┘
```

### **Fase 2: Proses Pencarian (User Interaction)**

```
┌─────────────────────────────┐
│ User memasukkan query       │
│ melalui text_input          │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ User klik tombol "Cari"     │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ Validasi Query              │
│ - Apakah query kosong?      │
└─────────────────────────────┘
  │
  ├──── Ya ────▶ Tampilkan Error
  │
  ▼ Tidak
┌─────────────────────────────┐
│ Panggil fungsi search()     │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 1. Transform query → vektor │
│    query_vec = vectorizer   │
│              .transform()   │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 2. Hitung Cosine Similarity │
│    cosine_similarity(       │
│      query_vec,             │
│      tfidf_matrix           │
│    )                        │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 3. Urutkan berdasarkan      │
│    similarity score         │
│    (descending)             │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ 4. Filter hasil             │
│    - score > 0              │
│    - ambil top_k (max 3)    │
└─────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│ Tampilkan hasil pencarian   │
│ dalam format card           │
└─────────────────────────────┘
```

---

## 🧩 Komponen Utama

### 1. **Dataset Dokumen (Corpus)**

```python
documents = [
    {
        "id": 1,
        "title": "Pengantar Kecerdasan Buatan",
        "content": "..."
    },
    # ... 5 dokumen total
]
```

**Struktur Data:**
| Field | Tipe | Deskripsi |
|-------|------|-----------|
| `id` | Integer | Identifier unik dokumen |
| `title` | String | Judul dokumen |
| `content` | String | Isi/konten dokumen yang di-index |

---

### 2. **Fungsi Indexing TF-IDF**

```python
@st.cache_resource
def create_tfidf_index(corpus_data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus_data)
    return vectorizer, tfidf_matrix
```

**Penjelasan:**
- `@st.cache_resource`: Menyimpan hasil indexing di cache agar tidak perlu dihitung ulang
- `TfidfVectorizer`: Mengubah teks menjadi representasi vektor numerik
- `stop_words='english'`: Menghilangkan kata-kata umum yang tidak bermakna

---

### 3. **Fungsi Pencarian**

```python
def search(query, vectorizer, tfidf_matrix, documents, top_k=3):
    # Step 1: Transformasi query
    query_vec = vectorizer.transform([query])
    
    # Step 2: Hitung similarity
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Step 3: Urutkan hasil
    related_docs_indices = cosine_similarities.argsort()[::-1]
    
    # Step 4: Filter dan return hasil
    results = []
    for i in related_docs_indices:
        score = cosine_similarities[i]
        if score > 0:
            results.append({...})
    
    return results[:top_k]
```

---

## 📊 Diagram Alur

### Flowchart Utama

```
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Inisialisasi Sistem   │
              │  - Load libraries      │
              │  - Setup Streamlit     │
              │  - Load documents      │
              │  - Build TF-IDF index  │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │   Render UI Streamlit  │
              │   - Input field        │
              │   - Button "Cari"      │
              │   - Sidebar info       │
              └───────────┬────────────┘
                          │
                          ▼
              ┌────────────────────────┐
              │  Menunggu input user   │◀──────────┐
              └───────────┬────────────┘           │
                          │                        │
                          ▼                        │
                   ┌─────────────┐                 │
                   │ User input  │                 │
                   │   query?    │                 │
                   └──────┬──────┘                 │
                          │                        │
              ┌───────────┴───────────┐            │
              │                       │            │
          [TIDAK]                   [YA]           │
              │                       │            │
              ▼                       ▼            │
     ┌─────────────────┐   ┌───────────────────┐   │
     │ Tampilkan Error │   │ Proses Pencarian  │   │
     └────────┬────────┘   └─────────┬─────────┘   │
              │                      │             │
              │                      ▼             │
              │            ┌───────────────────┐   │
              │            │ Transformasi      │   │
              │            │ Query → TF-IDF    │   │
              │            └─────────┬─────────┘   │
              │                      │             │
              │                      ▼             │
              │            ┌───────────────────┐   │
              │            │ Hitung Cosine     │   │
              │            │ Similarity        │   │
              │            └─────────┬─────────┘   │
              │                      │             │
              │                      ▼             │
              │            ┌───────────────────┐   │
              │            │ Ranking &         │   │
              │            │ Filtering         │   │
              │            └─────────┬─────────┘   │
              │                      │             │
              │          ┌───────────┴───────────┐ │
              │          │                       │ │
              │     [ADA HASIL]           [KOSONG] │
              │          │                       │ │
              │          ▼                       ▼ │
              │  ┌─────────────────┐  ┌──────────┐ │
              │  │ Tampilkan Hasil │  │ Warning  │ │
              │  │ dalam Card      │  │ No Match │ │
              │  └────────┬────────┘  └────┬─────┘ │
              │           │                │       │
              └───────────┴────────────────┴───────┘
```

---

## 📐 Penjelasan Algoritma

### TF-IDF (Term Frequency - Inverse Document Frequency)

**Rumus:**

```
TF-IDF(t, d) = TF(t, d) × IDF(t)
```

Dimana:
- **TF(t, d)** = Frekuensi term `t` dalam dokumen `d`
- **IDF(t)** = log(N / df(t))
  - N = Total jumlah dokumen
  - df(t) = Jumlah dokumen yang mengandung term `t`

**Ilustrasi:**

```
Dokumen A: "kucing makan ikan"
Dokumen B: "anjing makan daging"
Dokumen C: "kucing tidur"

Kata "kucing":
- TF di A = 1/3 = 0.33
- TF di B = 0
- TF di C = 1/2 = 0.5

- IDF = log(3/2) = 0.176

TF-IDF("kucing", A) = 0.33 × 0.176 = 0.058
TF-IDF("kucing", C) = 0.5 × 0.176 = 0.088
```

---

### Cosine Similarity

**Rumus:**

```
                    A · B
Cosine(A, B) = ─────────────────
                ||A|| × ||B||
```

Dimana:
- **A · B** = Dot product vektor A dan B
- **||A||** = Magnitude (panjang) vektor A
- **||B||** = Magnitude (panjang) vektor B

**Ilustrasi:**

```
Query Vector:    Q = [0.2, 0.5, 0.0, 0.3]
Document Vector: D = [0.1, 0.4, 0.2, 0.3]

Dot Product = (0.2×0.1) + (0.5×0.4) + (0.0×0.2) + (0.3×0.3)
            = 0.02 + 0.20 + 0.00 + 0.09
            = 0.31

||Q|| = √(0.04 + 0.25 + 0 + 0.09) = √0.38 = 0.616
||D|| = √(0.01 + 0.16 + 0.04 + 0.09) = √0.30 = 0.548

Cosine Similarity = 0.31 / (0.616 × 0.548) = 0.31 / 0.338 = 0.917
```

**Interpretasi Nilai:**
| Range | Interpretasi |
|-------|--------------|
| 1.0 | Identik/Sama persis |
| 0.7 - 0.99 | Sangat mirip |
| 0.4 - 0.69 | Cukup mirip |
| 0.1 - 0.39 | Sedikit mirip |
| 0.0 | Tidak ada kemiripan |

---

## 📁 Struktur File Project

```
project-deploy/
├── app.py              # Kode utama aplikasi
├── requirements.txt    # Dependencies (streamlit, scikit-learn, pandas)
├── README.md          # Dokumentasi singkat
└── DOCUMENTATION.md   # Dokumentasi lengkap (file ini)
```

---

## 🔗 Referensi

- [Scikit-Learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Cosine Similarity - Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

> 📝 **Catatan:** Dokumentasi ini dibuat untuk keperluan tugas mata kuliah Temu Balik Informasi (TBI).
