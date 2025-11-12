import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Tampilan Halaman
st.set_page_config(page_title="Nextify", page_icon="🎬", layout="centered")

page_bg = """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: url("https://blogger.googleusercontent.com/img/a/AVvXsEi90tws6ZDy3M8lxvXzlP7h5A4fO2bDULKsG6Ui_t26ZF6WxLTOHevrFDbXPBaT-ZEq2DAbwrQnZo2MF86sk9xcOpt04SvJr0mfXfKjoLgqR7xrB1TxYWrAEApBeRgGREGZnHPdnZXCi3GmYD8aX26n0vuaaffT5cQRCQLbQtOtAsHRsBrQSNpw8_1s=w1200-h630-p-k-no-nu") no-repeat center center fixed;
    background-size: cover;
    color: #fff;
    font-family: 'Poppins', sans-serif;
}

[data-testid="stHeader"] {background: rgba(0,0,0,0);}
[data-testid="stSidebar"] {background-color: rgba(30,30,30,0.6); color: white;}

/* Judul Nextify */
h1 {
    text-align: center;
    font-size: 64px !important;
    color: #ff0000; /* isi huruf merah solid */
    font-weight: 900 !important;
    letter-spacing: 2px;

    /* Garis Tepi */
    filter: drop-shadow(1px 1px 0 black)
            drop-shadow(-1px 1px 0 black)
            drop-shadow(1px -1px 0 black)
            drop-shadow(-1px -1px 0 black);

    text-shadow: 0px 0px 15px rgba(255, 0, 0, 0.6);
}

/* Logo Nextify */
.stRadio > label, .stRadio div[role="radiogroup"] label p {
    color: #fff !important;
    font-weight: 600;
}
.stTextInput > label {
    color: #fff !important;
    font-weight: 600;
    font-size: 18px;
}

/* Search Bar */
input[type="text"] {
    background-color: #ffffff !important; /* putih solid */
    color: #000000 !important; /* teks hitam */
    border: 2px solid #ff4b2b !important; /* tepi warna merah lembut */
    border-radius: 12px !important;
    padding: 10px 14px !important;
    font-weight: 600;
    transition: all 0.3s ease;
}
input[type="text"]:focus {
    border-color: #ff0000 !important;
    box-shadow: 0 0 10px rgba(255, 0, 0, 0.4);
}
input[type="text"]::placeholder {
    color: rgba(0,0,0,0.5);
}

/* Tombol */
button[kind="secondary"], .stButton>button {
    background: linear-gradient(135deg, #ff4b2b, #ff416c);
    color: white !important;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(255,65,108,0.4);
}
button[kind="secondary"]:hover, .stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(135deg, #ff416c, #ff4b2b);
    box-shadow: 0 4px 20px rgba(255,65,108,0.6);
}

/* Kotak Warning */
div[data-testid="stAlert"] {
    background-color: rgba(255, 182, 193, 0.9) !important;
    border: 2px solid #ff6fa8 !important;
    border-radius: 12px !important;
    color: #000 !important;
    font-weight: 600;
}

/* Gaya footer */
footer, .css-1lsmgbg, .stMarkdown p {
    color: #eee !important;
    text-align: center;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

#Data lagu & Film
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

#fungsi agar gambar tidak error ditampilan website
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

#fungsi rekomendasi 
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

# Tampilan UI TITTLE
st.title("🎬 Nextify")
pilihan = st.radio("Pilih Kategori:", ["Lagu Bruno Mars", "Film Horror"])

# Tampilan GUI
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

# Input User
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

st.markdown("---")
st.markdown("<p style='text-align:center;'>🎵 Dibuat dengan gilbaay❤️</p>", unsafe_allow_html=True)

