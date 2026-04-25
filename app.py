import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Judul Aplikasi
st.set_page_config(page_title="Stock Predictor AI", layout="wide")
st.title("📈 AI Stock Price Predictor")
st.write("Prediksi harga saham berdasarkan data historis, Volume, dan RSI.")

# --- SIDEBAR INPUT ---
st.sidebar.header("Konfigurasi Input")
ticker = st.sidebar.text_input("Masukkan Kode Saham (Ticker)", value="SCMA.JK")
start_date = st.sidebar.date_input("Tanggal Mulai Data", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Tanggal Akhir Data", value=pd.to_datetime("today"))

# --- PROSES DATA ---
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data

if ticker:
    with st.spinner(f"Mengambil data {ticker}..."):
        df = load_data(ticker, start_date, end_date)

    if df.empty:
        st.error("Data tidak ditemukan. Pastikan kode ticker benar (gunakan .JK untuk saham Indonesia).")
    else:
        # Feature Engineering
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['S_5'] = df['Close'].rolling(window=5).mean()
        df['V_5'] = df['Volume'].rolling(window=5).mean()
        df = df.dropna()

        # Fitur & Target
        X = df[['S_5', 'V_5', 'RSI']]
        y = df['Close']

        # Split Data
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Model Training (Random Forest)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())

        # Prediksi
        y_pred = model.predict(X_test)
        
        # --- TAMPILAN DASHBOARD ---
        col1, col2, col3 = st.columns(3)
        
        # Metrik Akurasi
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        last_features = X.tail(1)
        next_price = model.predict(last_features)[0]

        col1.metric("Akurasi (R2 Score)", f"{r2:.2%}")
        col2.metric("Rata-rata Error (MAE)", f"Rp{mae:.2f}")
        col3.metric(f"Prediksi Harga Besok ({ticker})", f"Rp{next_price:.2f}")

        # Grafik Harga
        st.subheader(f"Grafik Prediksi vs Aktual: {ticker}")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label="Harga Asli", color='blue')
        ax.plot(y_test.index, y_pred, label="Prediksi AI", color='red', linestyle='--')
        ax.legend()
        st.pyplot(fig)

        # Tampilkan Data Mentah
        if st.checkbox("Tampilkan Data Mentah"):
            st.write(df.tail(10))