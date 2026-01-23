import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Stock Dashboard", layout="wide")
st.title("📈 Stock Price Dashboard (Yahoo Finance – Stable Mode)")

# ---------------- INPUT ----------------
tickers_input = st.text_input(
    "Enter NSE tickers (comma separated, without .NS)",
    "RELIANCE,TCS"
)

tickers = [t.strip().upper() + ".NS" for t in tickers_input.split(",") if t.strip()]

# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=3600)
def load_data(tickers):
    prices = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)

            df = stock.history(
                period="1y",        # ✅ MUCH safer than start/end
                auto_adjust=True
            )

            if not df.empty and "Close" in df.columns:
                prices[ticker.replace(".NS", "")] = df["Close"]

            time.sleep(1)  # ✅ prevents Yahoo blocking

        except Exception:
            continue

    if len(prices) == 0:
        return pd.DataFrame()

    return pd.DataFrame(prices)

# ---------------- LOAD ----------------
prices = load_data(tickers)

# ---------------- VALIDATION ----------------
if prices.empty:
    st.error("""
❌ No valid data returned.

**Why this happens (important):**
- Yahoo Finance blocks Streamlit Cloud IPs for NSE
- Bulk download is blocked
- Too many refreshes trigger rate limits

✅ **What to do**
- Use **1 ticker at a time**
- Wait **30–60 seconds**
- Do NOT refresh repeatedly
""")
    st.stop()

# ---------------- PRICE PLOT ----------------
st.subheader("📊 Closing Prices")

fig1, ax1 = plt.subplots()
prices.plot(ax=ax1)
ax1.set_ylabel("Price (₹)")
ax1.grid(True)
st.pyplot(fig1)

# ---------------- PERCENT CHANGE ----------------
st.subheader("📈 Percentage Change")

pct_change = (prices / prices.iloc[0] - 1) * 100

fig2, ax2 = plt.subplots()
pct_change.plot(ax=ax2)
ax2.set_ylabel("Change (%)")
ax2.grid(True)
st.pyplot(fig2)

# ---------------- TABLE ----------------
st.subheader("📋 Latest Prices")
st.dataframe(prices.tail())
