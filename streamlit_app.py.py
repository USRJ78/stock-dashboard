# Streamlit Stock Analysis & Portfolio Dashboard
# Yahoo Finance – Stable Version (FIXED)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# ---------------- Page Config ----------------
st.set_page_config(page_title="Stock Analysis Dashboard", layout="wide")

st.title("📈 Stock Analysis & Portfolio Dashboard")
st.caption("Yahoo Finance • Cached • Cloud-safe")

# ---------------- Sidebar ----------------
st.sidebar.header("Inputs")

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    "AAPL,MSFT,RELIANCE.NS"
)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run = st.sidebar.button("Run Analysis")

# ---------------- Data Loader ----------------
@st.cache_data(ttl=3600)
def fetch_ticker(ticker, start, end):
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False
        )
        if df.empty or "Close" not in df:
            return None
        return df["Close"]
    except Exception:
        return None

def load_data(ticker_list, start, end):
    series_list = []

    for t in ticker_list:
        s = fetch_ticker(t, start, end)
        if isinstance(s, pd.Series) and not s.empty:
            s.name = t
            series_list.append(s)

    if not series_list:
        return pd.DataFrame()

    return pd.concat(series_list, axis=1).dropna()

# ---------------- Main Logic ----------------
if run:
    tickers_list = [t.strip().upper() for t in tickers.split(",")]

    with st.spinner("Fetching stock data..."):
        prices = load_data(tickers_list, start_date, end_date)

    if prices.empty:
        st.error(
            "❌ No valid data returned.\n\n"
            "Possible reasons:\n"
            "- Yahoo rate-limited NSE stocks\n"
            "- Invalid ticker symbol\n"
            "- Too many requests\n\n"
            "👉 Try fewer tickers or retry after 1–2 minutes."
        )
        st.stop()

    returns = prices.pct_change().dropna()

    # ---------- % Change ----------
    st.subheader("📊 Percentage Change Comparison")
    pct_change = (prices / prices.iloc[0] - 1) * 100

    fig, ax = plt.subplots()
    pct_change.plot(ax=ax)
    ax.set_ylabel("% Change")
    st.pyplot(fig)
    plt.close()

    # ---------- Histogram ----------
    st.subheader("📈 Returns Histogram")
    returns.hist(bins=40)
    st.pyplot(plt.gcf())
    plt.close()

    # ---------- Correlation ----------
    st.subheader("🔥 Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close()

    # ---------- Portfolio Optimization ----------
    st.subheader("🧮 Portfolio Optimization")

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_portfolios = 3000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(prices.columns))
        weights /= np.sum(weights)
        weights_record.append(weights)

        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = ret / vol

    idx = np.argmax(results[2])

    st.markdown("### ✅ Optimal Portfolio Weights")
    st.dataframe(pd.DataFrame({
        "Ticker": prices.columns,
        "Weight": weights_record[idx]
    }))

    st.markdown(f"**Expected Return:** {results[0, idx]*100:.2f}%")
    st.markdown(f"**Expected Volatility:** {results[1, idx]*100:.2f}%")

    fig, ax = plt.subplots()
    ax.scatter(results[1], results[0], c=results[2], cmap="viridis", s=6)
    ax.scatter(results[1, idx], results[0, idx], color="red", marker="*", s=200)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    st.pyplot(fig)
    plt.close()

else:
    st.info("👈 Enter inputs and click **Run Analysis**")
