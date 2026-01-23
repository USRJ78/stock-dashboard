import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"
)

st.title("📈 Stock Price Comparison Dashboard")

# ----------------- USER INPUT -----------------
tickers_input = st.text_input(
    "Enter NSE tickers (comma separated, without .NS)",
    value="RELIANCE,TCS,INFY"
)

start_date = st.date_input("Start Date", date(2023, 1, 1))
end_date = st.date_input("End Date", date.today())

tickers = [t.strip().upper() + ".NS" for t in tickers_input.split(",") if t.strip()]

# ----------------- DATA LOADER -----------------
@st.cache_data(ttl=3600)
def load_data(tickers, start, end):
    prices = {}

    for ticker in tickers:
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                threads=False
            )

            if not df.empty:
                prices[ticker.replace(".NS", "")] = df["Adj Close"]

        except Exception:
            continue

    if not prices:
        return pd.DataFrame()

    return pd.DataFrame(prices)

# ----------------- FETCH DATA -----------------
prices = load_data(tickers, start_date, end_date)

# ----------------- VALIDATION -----------------
if prices.empty:
    st.error("""
❌ No valid data returned.

**Possible reasons:**
- Yahoo Finance rate-limited NSE stocks
- Invalid ticker symbols
- Too many requests

👉 Try **1–2 tickers only** or retry after **1–2 minutes**
""")
    st.stop()

# ----------------- PRICE CHART -----------------
st.subheader("📊 Adjusted Closing Prices")

fig1, ax1 = plt.subplots()
prices.plot(ax=ax1)
ax1.set_ylabel("Price (₹)")
ax1.grid(True)

st.pyplot(fig1)

# ----------------- PERCENTAGE CHANGE -----------------
st.subheader("📈 Percentage Change Comparison")

pct_change = (prices / prices.iloc[0] - 1) * 100

fig2, ax2 = plt.subplots()
pct_change.plot(ax=ax2)
ax2.set_ylabel("Change (%)")
ax2.grid(True)

st.pyplot(fig2)

# ----------------- DATA TABLE -----------------
st.subheader("📋 Price Data (Last 5 Rows)")
st.dataframe(prices.tail())
