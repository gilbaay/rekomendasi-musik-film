# =====================================================
# 🎵 STREAMLIT REKOMENDASI MUSIK & FILM + MIRIP AI (VERSI FIX GAMBAR)
# =====================================================

import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# 🌈 1️⃣ Konfigurasi Tampilan Halaman
# =====================================================
st.set_page_config(page_title="Nextify", page_icon="🎬", layout="centered")

page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #eef6fb, #fefefe);
    color: #111;
    font-family: 'Poppins', sans-serif;
}
[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(255,255,255,0.5);}
h1, h2, h3, h4 {
    text-align: center;
    color: #111 !important;
}
.stRadio > label, .stRadio div[role="radiogroup"] label p {
    color: #111 !important;
    font-weight: 600;
}
.stTextInput > label {
    color: #111 !important;
    font-weight: 600;
    font-size: 18px;
}
input[type="text"] {
    background-color: white !important;
    color: black !important;
    border: 2px solid #ccc !important;
    border-radius: 12px !important;
    padding: 8px 12px !important;
}
button[kind="secondary"], .stButton>button {
    background-color: #ff99c8 !important;
    color: #111 !important;
    font-weight: 600;
    border-radius: 10px;
    border: none;
    transition: all 0.2s ease-in-out;
}
button[kind="secondary"]:hover, .stButton>button:hover {
    background-color: #ff6fa8 !important;
    color: #000 !important;
    transform: scale(1.03);
}
/* Kotak Warning (Lagu/Film tidak ditemukan) */
div[data-testid="stAlert"] {
    background-color: #ffb6c1 !important; /* Pink lembut */
    border: 2px solid #ff6fa8 !important;
    border-radius: 10px !important;
}

/* Ubah warna font di dalam kotak warning */
div[data-testid="stAlert"] p,
div[data-testid="stAlert"] span,
div[data-testid="stAlert"] h4 {
    color: #111 !important; /* Hitam */
    font-weight: 600 !important;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# =====================================================
# 🎶 2️⃣ Data Lagu Bruno Mars & Film Horror
# =====================================================

lagu_data = {
    "Talking to the Moon": {
        "genre": "Pop Ballad",
        "deskripsi": "Lagu melankolis tentang kerinduan di bawah sinar bulan.",
        "cover": "https://i.scdn.co/image/ab67616d0000b273f6b55ca93bd33211227b502b"
    },
    "Grenade": {
        "genre": "Pop Soul",
        "deskripsi": "Tentang pengorbanan cinta yang tidak terbalas.",
        "cover": "https://i.scdn.co/image/ab67616d0000b273f6b55ca93bd33211227b502b"
    },
    "Just the Way You Are": {
        "genre": "Pop",
        "deskripsi": "Lagu cinta yang memuji keindahan alami seseorang.",
        "cover": "https://i.scdn.co/image/ab67616d0000b273f6b55ca93bd33211227b502b"
    },
    "Locked Out of Heaven": {
        "genre": "Funk Rock",
        "deskripsi": "Energi funk dan pop yang terinspirasi dari gaya The Police.",
        "cover": "https://i.scdn.co/image/ab67616d0000b27349055dce3554e72e82082980"
    },
    "Treasure": {
        "genre": "Funk / Disco",
        "deskripsi": "Lagu ceria dengan vibe disko tahun 80-an.",
        "cover": "https://i.scdn.co/image/ab67616d0000b27349055dce3554e72e82082980"
    },
}

film_data = {
    "The Conjuring": {
        "genre": "Horror Supernatural",
        "deskripsi": "Berdasarkan kisah nyata Ed dan Lorraine Warren melawan roh jahat.",
        "poster": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTBuqRC1kIMgtuImFMwAPfcVTlzKsd6pwDcog&s"
    },
    "Insidious": {
        "genre": "Horror / Thriller",
        "deskripsi": "Sebuah keluarga berhadapan dengan dunia roh jahat.",
        "poster": "https://upload.wikimedia.org/wikipedia/id/2/2d/Insidious_poster.jpg"
    },
    "Annabelle": {
        "genre": "Horror Supernatural",
        "deskripsi": "Boneka iblis yang meneror keluarga muda.",
        "poster": "https://upload.wikimedia.org/wikipedia/id/thumb/9/9b/Annabelle-poster.jpg/250px-Annabelle-poster.jpg"
    },
    "The Nun": {
        "genre": "Horror Misteri",
        "deskripsi": "Kisah biarawati iblis Valak yang menakutkan di biara Rumania.",
        "poster": "https://upload.wikimedia.org/wikipedia/id/thumb/b/bc/The_Nun_II_%282023%29.jpg/250px-The_Nun_II_%282023%29.jpg"
    },
    "Smile": {
        "genre": "Psychological Horror",
        "deskripsi": "Kutukan misterius yang memaksa korbannya tersenyum sebelum mati.",
        "poster": "https://upload.wikimedia.org/wikipedia/id/7/7f/Smile_%282022_film%29.jpg"
    },
}

# =====================================================
# 🧠 Fungsi Gambar Aman (URL atau Lokal)
# =====================================================
def safe_image_show(url_or_path, width=250):
    placeholder_url = "https://via.placeholder.com/250x250?text=No+Image"
    try:
        if url_or_path.startswith("http"):
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url_or_path, headers=headers, timeout=10)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                st.image(img, width=width)
            else:
                st.image(placeholder_url, width=width)
        else:
            st.image(url_or_path, width=width)
    except Exception as e:
        st.image(placeholder_url, width=width)
        st.write(f"⚠️ Gagal memuat gambar: {e}")

# =====================================================
# ⚙️ 3️⃣ Fungsi Rekomendasi
# =====================================================
def buat_rekomendasi(data_dict, nama_item, top_n=2):
    df = pd.DataFrame(data_dict).T
    df["text"] = df["genre"] + " " + df["deskripsi"]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["text"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx = df.index.tolist().index(nama_item)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    rekomendasi_idx = [i for i, s in sim_scores[1:top_n+1]]
    return df.iloc[rekomendasi_idx].index.tolist()

# =====================================================
# 🎧 4️⃣ UI
# =====================================================
st.title("🎬 Nextify")
pilihan = st.radio("Pilih Kategori:", ["Lagu Bruno Mars", "Film Horror"])

# =====================================================
# 🎧 5️⃣ Tampilan
# =====================================================
def tampilkan_info(nama, data, tipe="lagu"):
    if tipe == "lagu":
        safe_image_show(data[nama]["cover"])
        st.markdown(f"## 🎵 {nama} — Bruno Mars")
    else:
        safe_image_show(data[nama]["poster"])
        st.markdown(f"## 🎬 {nama}")
    st.markdown(f"**Genre:** {data[nama]['genre']}")
    st.markdown(data[nama]['deskripsi'])

    mirip = buat_rekomendasi(data, nama)
    if tipe == "lagu":
        st.markdown("### 💡 Lagu serupa yang mungkin kamu suka:")
        for m in mirip:
            st.markdown(f"- 🎧 **{m}** ({data[m]['genre']})")
    else:
        st.markdown("### 💡 Film horor mirip yang direkomendasikan:")
        for m in mirip:
            st.markdown(f"- 🎥 **{m}** ({data[m]['genre']})")

# =====================================================
# 🔍 6️⃣ Input User
# =====================================================
if pilihan == "Lagu Bruno Mars":
    query = st.text_input("🔍 Cari lagu Bruno Mars...")
    if query:
        cocok = [n for n in lagu_data if query.lower() in n.lower()]
        if cocok:
            tampilkan_info(cocok[0], lagu_data, "lagu")
        else:
            st.warning("Lagu tidak ditemukan!")
else:
    query = st.text_input("🔍 Cari film horor...")
    if query:
        cocok = [n for n in film_data if query.lower() in n.lower()]
        if cocok:
            tampilkan_info(cocok[0], film_data, "film")
        else:
            st.warning("Film tidak ditemukan!")

# =====================================================
# 👣 7️⃣ Footer
# =====================================================
st.markdown("---")
st.markdown("<p style='text-align:center;'>🎵 Dibuat dengan ❤️ oleh Streamlit + AI Cosine Similarity</p>", unsafe_allow_html=True)

