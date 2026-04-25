import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Konfigurasi Halaman
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")

st.title("📈 Dashboard Prediksi Saham AI")
st.markdown("Aplikasi ini menggunakan **Random Forest Machine Learning** untuk memprediksi harga penutupan besok.")

# --- SIDEBAR ---
st.sidebar.header("Pilih Saham")
# Input tanpa perlu .JK, otomatis dikapitalisasi
ticker_symbol = st.sidebar.text_input("Masukkan Kode Saham (Contoh: SCMA, BBCA, BMRI)", value="SCMA").upper()
ticker_full = f"{ticker_symbol}.JK"

start_date = st.sidebar.date_input("Mulai Data", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Sampai Tanggal", value=pd.to_datetime("today"))

# --- LOAD DATA ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

if ticker_symbol:
    df = load_data(ticker_full, start_date, end_date)

    if df.empty:
        st.error(f"Data {ticker_symbol} tidak ditemukan. Pastikan kode benar.")
    else:
        # --- FEATURE ENGINEERING ---
        df['S_5'] = df['Close'].rolling(window=5).mean()
        df['V_5'] = df['Volume'].rolling(window=5).mean()
        
        # Hitung RSI secara manual
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df_clean = df.dropna()

        # --- MACHINE LEARNING (RANDOM FOREST) ---
        X = df_clean[['S_5', 'V_5', 'RSI']]
        y = df_clean['Close']
        
        split = int(len(df_clean) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        
        y_pred = model.predict(X_test)
        next_price = model.predict(X.tail(1))[0]

        # --- METRICS AREA ---
        st.subheader(f"Statistik & Prediksi {ticker_symbol}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Akurasi Model (R2)", f"{r2_score(y_test, y_pred):.2%}")
        m2.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")
        m3.metric("RSI (Momentum)", f"{df_clean['RSI'].iloc[-1]:.2f}")

        # --- CANDLESTICK CHART (PLOTLY) ---
        st.subheader("Grafik Candlestick Interaktif")
        
        fig = go.Figure()

        # Plot Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Harga Market'
        ))

        # Tambahkan Garis Prediksi AI
        fig.add_trace(go.Scatter(
            x=y_test.index, 
            y=y_pred, 
            line=dict(color='orange', width=2), 
            name='AI Prediction'
        ))

        fig.update_layout(
            height=600,
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            title=f"Pergerakan Harga {ticker_symbol}",
            yaxis_title="Harga (IDR)"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Tips:** Gunakan mouse wheel untuk zoom dan klik-tarik pada grafik untuk melihat detail pergerakan harga.")
