import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. Konfigurasi Halaman (Harus di paling atas) ---
st.set_page_config(
    page_title="Estimasi Harga Rumah",
    page_icon="🏠",
    layout="centered"
)

# --- 2. CSS Custom untuk Tampilan Lebih Keren ---
st.markdown("""
    <style>
    .main {
        background-color: #f9f9f9;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #ce3b3b;
        color: white;
    }
    h1 {
        color: #333;
        text-align: center;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)


# --- 3. Load Model & Fitur ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('house_price_model.pkl')
        # Pastikan file ini berisi list nama kolom yang urutannya SAMA dengan saat training
        features = joblib.load('model_features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None


model, feature_names = load_data()

# --- 4. Header Aplikasi ---
st.title("🏠 Smart House Price Estimator")
st.markdown("---")
st.write("Isi spesifikasi rumah di bawah ini untuk mendapatkan estimasi harga pasar yang akurat.")

if model is not None and feature_names is not None:

    # --- 5. Form Input User (Mapping Bahasa & Layout) ---

    # Kita buat dictionary untuk menyimpan input user
    input_data = {}

    # Kelompokkan input agar tampilan rapi menggunakan Columns
    st.subheader("📋 Spesifikasi Utama")
    col1, col2 = st.columns(2)

    with col1:
        # OverallQual -> Kualitas Material
        input_data['OverallQual'] = st.slider(
            "🌟 Kualitas Material & Finishing (1-10)",
            min_value=1, max_value=10, value=5,
            help="1 = Sangat Buruk, 5 = Rata-rata, 10 = Sangat Mewah"
        )

        # GrLivArea -> Luas Bangunan Utama
        input_data['GrLivArea'] = st.number_input(
            "📐 Luas Bangunan Utama (sq ft)",
            min_value=500, value=1500, step=10,
            help="Total luas area hunian di atas tanah (tidak termasuk basement)"
        )

        # YearBuilt -> Tahun Bangun
        input_data['YearBuilt'] = st.number_input(
            "🏗️ Tahun Dibangun",
            min_value=1800, max_value=2025, value=2010
        )

    with col2:
        # GarageCars -> Kapasitas Garasi
        input_data['GarageCars'] = st.selectbox(
            "🚗 Kapasitas Garasi (Mobil)",
            options=[0, 1, 2, 3, 4], index=2
        )

        # TotalBsmtSF -> Luas Basement
        input_data['TotalBsmtSF'] = st.number_input(
            "📦 Luas Basement (sq ft)",
            min_value=0, value=0, step=10
        )

        # FullBath -> Kamar Mandi
        input_data['FullBath'] = st.selectbox(
            "r🚿 Jumlah Kamar Mandi Full",
            options=[1, 2, 3, 4], index=1
        )

    # --- 6. Input Tambahan (Hidden/Advanced Features) ---
    # Fitur-fitur teknis yang diminta user untuk diganti bahasanya

    with st.expander("⚙️ Spesifikasi Detail (Opsional)"):
        col3, col4 = st.columns(2)

        with col3:
            # 1stFlrSF -> Luas Lantai Dasar
            input_data['1stFlrSF'] = st.number_input(
                "Luas Lantai 1 (sq ft)",
                min_value=0, value=1000,
                help="Ukuran luas lantai dasar saja"
            )

            # TotRmsAbvGrd -> Total Ruangan
            input_data['TotRmsAbvGrd'] = st.number_input(
                "Total Jumlah Ruangan",
                min_value=2, max_value=15, value=6,
                help="Total ruangan di atas tanah (tidak termasuk kamar mandi)"
            )

        with col4:
            # YearRemodAdd -> Tahun Renovasi
            input_data['YearRemodAdd'] = st.number_input(
                "Tahun Renovasi Terakhir",
                min_value=1950, max_value=2025, value=2010,
                help="Jika tidak pernah renovasi, isi sama dengan tahun bangun"
            )

            # Fireplaces -> Jumlah Perapian
            input_data['Fireplaces'] = st.selectbox(
                "Jumlah Perapian",
                options=[0, 1, 2, 3]
            )

    # --- Handling Fitur Lain yang Mungkin Ada di Model tapi Tidak di UI ---
    # Ini penting agar model tidak error jika ada fitur 'Id' atau lainnya
    for feature in feature_names:
        if feature not in input_data:
            input_data[feature] = 0  # Isi default 0 atau rata-rata dataset Anda

    # --- 7. Tombol Eksekusi ---
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Hitung Estimasi Harga"):

        # Loading animation
        with st.spinner('Sedang menganalisis data properti...'):

            # Buat DataFrame sesuai urutan kolom saat training
            try:
                # Mengurutkan input_data agar sesuai dengan urutan feature_names dari pickle
                input_df = pd.DataFrame([input_data])

                # Pastikan urutan kolom sama persis dengan model
                input_df = input_df[feature_names]

                # Prediksi
                prediction = model.predict(input_df)[0]

                # Tampilkan Hasil
                st.markdown("---")
                st.subheader("Hasil Prediksi")

                # Format currency USD
                formatted_price = f"${prediction:,.2f}"

                # Tampilan Metric Besar
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin-bottom:0px; color:#555;">Estimasi Nilai Pasar</h3>
                    <h1 style="color:#2ecc71; font-size: 3em; margin-top:5px;">{formatted_price}</h1>
                </div>
                """, unsafe_allow_html=True)

                st.success("Perhitungan selesai! Harga ini adalah estimasi berdasarkan data historis.")

            except Exception as e:
                st.error(f"Terjadi kesalahan dalam pemrosesan data: {e}")
                st.info("Pastikan semua file model (.pkl) cocok dengan kode ini.")

else:
    st.error(
        "File model tidak ditemukan! Pastikan 'house_price_model.pkl' dan 'model_features.pkl' ada di folder yang sama.")