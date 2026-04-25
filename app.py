import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- CONFIG ---
st.set_page_config(page_title="AI Stock Predictor Pro", layout="wide")

st.title("📈 Dashboard Prediksi Saham AI")
st.markdown("Aplikasi ini menggunakan **Random Forest Machine Learning** untuk memprediksi harga.")

# --- SIDEBAR ---
st.sidebar.header("Pilih Saham")
ticker_symbol = st.sidebar.text_input("Kode Saham (Contoh: SCMA, BBCA, BMRI)", value="SCMA").upper()
ticker_full = f"{ticker_symbol}.JK"

start_date = st.sidebar.date_input("Mulai Data", value=pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("Sampai Tanggal", value=pd.to_datetime("today"))

# --- DATA ENGINE ---
@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data

if ticker_symbol:
    df = load_data(ticker_full, start_date, end_date)

    if df.empty:
        st.error(f"Data {ticker_symbol} tidak ditemukan. Pastikan kode benar.")
    else:
        # 1. Feature Engineering
        df['S_5'] = df['Close'].rolling(window=5).mean()
        df['V_5'] = df['Volume'].rolling(window=5).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df_clean = df.dropna()

        # 2. Machine Learning
        X = df_clean[['S_5', 'V_5', 'RSI']]
        y = df_clean['Close']
        
        split = int(len(df_clean) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train.values.ravel())
        
        y_pred = model.predict(X_test)
        next_price = model.predict(X.tail(1))[0]

        # 3. Metrics
        st.subheader(f"Statistik & Prediksi {ticker_symbol}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Akurasi Model (R2)", f"{r2_score(y_test, y_pred):.2%}")
        m2.metric("Estimasi Harga Besok", f"Rp{next_price:.2f}")
        m3.metric("RSI Saat Ini", f"{df_clean['RSI'].iloc[-1]:.2f}")

        # 4. GRAFIK CANDLESTICK (BAGIAN KRUSIAL)
        st.subheader("Grafik Candlestick Interaktif")
        
        # Inisialisasi Figure Plotly
        fig = go.Figure(data=[go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Harga Market'
        )])

        # Tambahkan Garis Prediksi AI
        fig.add_trace(go.Scatter(
            x=y_test.index, 
            y=y_pred, 
            line=dict(color='#FFA500', width=2), 
            name='AI Prediction'
        ))

        # Update Layout agar Full-Screen dan Dark Mode
        fig.update_layout(
            height=700,
            template="plotly_dark",
            xaxis_rangeslider_visible=False, # Matikan slider bawah agar rapi
            margin=dict(l=20, r=20, t=50, b=20),
            yaxis_title="Harga (IDR)"
        )

        # Tampilkan di Streamlit
        st.plotly_chart(fig, use_container_width=True)

        st.info("💡 **Gunakan Scroll** pada mouse untuk zoom in/out grafik.")
