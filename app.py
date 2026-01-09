import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Sistem Temu Balik Informasi Sederhana",
    page_icon="🔍",
    layout="centered"
)

# Styling dengan CSS kustom (opsional, agar terlihat lebih menarik)
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        color: #333333;
    }
    .result-card h3 {
        color: #000000;
    }
    .result-card p {
        color: #333333;
    }
</style>
""", unsafe_allow_html=True)

# 1. Dataset Dokumen (Corpus) - dari file CSV
# Fungsi untuk load dokumen dari CSV dengan caching
@st.cache_data
def load_documents(csv_path="documents.csv"):
    """
    Membaca dokumen dari file CSV.
    Format CSV harus memiliki kolom: id, title, content
    """
    try:
        df = pd.read_csv(csv_path)
        # Konversi dataframe ke list of dictionaries
        documents = df.to_dict('records')
        return documents
    except FileNotFoundError:
        st.error(f"File '{csv_path}' tidak ditemukan. Pastikan file CSV ada di direktori yang sama dengan app.py")
        return []
    except Exception as e:
        st.error(f"Error membaca file CSV: {e}")
        return []

# Load dokumen dari CSV
documents = load_documents()

# Mengambil teks konten untuk di-index
corpus = [doc['content'] for doc in documents]

# 2. Membangun Vektor TF-IDF (Indexing)
# Kita menggunakan TfidfVectorizer dari Scikit-Learn
@st.cache_resource # Cache agar tidak perlu hitung ulang setiap user berinteraksi
def create_tfidf_index(corpus_data):
    vectorizer = TfidfVectorizer(stop_words='english') # Bisa diganti stop_words bahasa indonesia jika ada library sastrawi
    tfidf_matrix = vectorizer.fit_transform(corpus_data)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_tfidf_index(corpus)

# Fungsi Pencarian
def search(query, vectorizer, tfidf_matrix, documents, top_k=3):
    # 1. Transformasi query menjadi vektor yang sama dengan dokumen
    query_vec = vectorizer.transform([query])
    
    # 2. Hitung Cosine Similarity antara query dan semua dokumen
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # 3. Urutkan hasil dari similarity tertinggi
    related_docs_indices = cosine_similarities.argsort()[::-1]
    
    results = []
    for i in related_docs_indices:
        score = cosine_similarities[i]
        if score > 0: # Hanya ambil yang memiliki kemiripan > 0
            results.append({
                "id": documents[i]['id'],
                "title": documents[i]['title'],
                "content": documents[i]['content'],
                "score": score
            })
            
    return results[:top_k]

# ==========================================
# UI Streamlit
# ==========================================

st.title("🔍 Sistem Pencarian Dokumen (IR)")
st.caption("Implementasi Sederhana Cosine Similarity dengan TF-IDF")

st.write("---")

# Input Pencarian
query = st.text_input("Masukkan kata kunci pencarian...", placeholder="Contoh: kecerdasan buatan, python, informasi")

if st.button("Cari"):
    if query:
        with st.spinner('Mencari dokumen...'):
            search_results = search(query, vectorizer, tfidf_matrix, documents)
        
        st.write(f"Menampilkan hasil untuk: **{query}**")
        
        if search_results:
            for result in search_results:
                # Menampilkan kartu hasil
                st.markdown(f"""
                <div class="result-card">
                    <h3>{result['title']}</h3>
                    <p>{result['content']}</p>
                    <small>Relevansi Score: {result['score']:.4f}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("Tidak ditemukan dokumen yang cocok dengan kata kunci tersebut.")
    else:
        st.error("Silakan masukkan kata kunci terlebih dahulu.")

# Sidebar untuk informasi tambahan
with st.sidebar:
    st.header("Tentang Aplikasi")
    st.info("""
    Aplikasi ini dibuat untuk memenuhi tugas mata kuliah Temu Balik Informasi.
    
    **Metode:**
    - TF-IDF Weighting
    - Cosine Similarity
    
    **Cara Kerja:**
    1. Input kueri
    2. Sistem menghitung kemiripan kueri dengan database dokumen
    3. Menampilkan dokumen paling relevan
    """)
    st.write("---")
    st.write("Koleksi Dokumen:")
    df_docs = pd.DataFrame(documents)
    st.dataframe(df_docs[['title']], hide_index=True)
