import streamlit as st
import pandas as pd
import joblib

# 1. Load Model dan Fitur
model = joblib.load('house_price_model.pkl')
features = joblib.load('model_features.pkl')

# 2. Judul Aplikasi
st.title("🏠 Aplikasi Prediksi Harga Rumah")
st.write("Masukkan detail rumah di bawah ini untuk mendapatkan estimasi harga.")

# 3. Form Input User
# Kita membuat input dinamis berdasarkan fitur yang dipilih saat training
user_input = {}

st.sidebar.header("Parameter Input")

# Dictionary untuk label yang lebih mudah dibaca (Opsional, sesuaikan dengan fitur Anda)
# Contoh fitur umum di dataset Kaggle: 'OverallQual', 'GrLivArea', 'GarageCars', etc.
for col in features:
    if col == 'OverallQual':
        user_input[col] = st.sidebar.slider("Kualitas Material (1-10)", 1, 10, 5)
    elif col == 'GrLivArea':
        user_input[col] = st.sidebar.number_input("Luas Area (sq ft)", min_value=500, max_value=5000, value=1500)
    elif col == 'GarageCars':
        user_input[col] = st.sidebar.selectbox("Kapasitas Garasi (Mobil)", [0, 1, 2, 3, 4])
    elif col == 'TotalBsmtSF':
        user_input[col] = st.sidebar.number_input("Luas Basement (sq ft)", min_value=0, max_value=3000, value=1000)
    elif col == 'FullBath':
        user_input[col] = st.sidebar.selectbox("Jumlah Kamar Mandi Full", [1, 2, 3, 4])
    elif col == 'YearBuilt':
        user_input[col] = st.sidebar.number_input("Tahun Dibangun", min_value=1900, max_value=2024, value=2000)
    else:
        # Fallback untuk fitur lain
        user_input[col] = st.sidebar.number_input(f"{col}", value=0)

# 4. Tombol Prediksi
if st.button("Prediksi Harga"):
    # Ubah input menjadi DataFrame
    input_df = pd.DataFrame([user_input])

    # Lakukan prediksi
    prediction = model.predict(input_df)[0]

    # Tampilkan hasil
    st.success(f"Estimasi Harga Rumah: ${prediction:,.2f}")

    # Tambahan info
    st.info("Prediksi ini menggunakan model Random Forest berdasarkan data historis.")