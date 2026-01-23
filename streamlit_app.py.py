# Streamlit Stock Analysis & Portfolio Dashboard (API VERSION)
# Run with: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from datetime import date

# ================= CONFIG =================
ALPHA_VANTAGE_API_KEY = "YVPHMGAAFPB3JZHN"

st.set_page_config(
    page_title="Stock Analysis & Portfolio Dashboard",
    layout="wide"
)

st.title("📈 Stock Analysis & Portfolio Dashboard (API Powered)")
st.markdown(
    "Stable cloud-based stock analysis using Alpha Vantage API."
)

# ================= SIDEBAR =================
st.sidebar.header("Inputs")

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    "AAPL,MSFT,RELIANCE.BSE"
)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run = st.sidebar.button("Run Analysis")

# ================= DATA FETCH =================
@st.cache_data(ttl=3600)
def fetch_alpha_vantage(symbol):
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    r = requests.get(url, timeout=10)
    data = r.json()

    if "Time Series (Daily)" not in data:
        return pd.Series(dtype=float)

    df = pd.DataFrame.from_dict(
        data["Time Series (Daily)"], orient="index"
    ).astype(float)

    df.index = pd.to_datetime(df.index)
    return df["5. adjusted close"].sort_index()

def load_data(tickers, start, end):
    prices = {}

    for t in tickers:
        series = fetch_alpha_vantage(t)
        if not series.empty:
            prices[t] = series.loc[start:end]

    if not prices:
        return pd.DataFrame()

    return pd.DataFrame(prices).dropna()

# ================= MAIN =================
if run:
    tickers_list = [t.strip().upper() for t in tickers.split(",")]

    prices = load_data(tickers_list, start_date, end_date)

    if prices.empty:
        st.error(
            "❌ No data fetched.\n\n"
            "Possible reasons:\n"
            "- API rate limit hit (5 calls/min)\n"
            "- Invalid ticker\n"
            "- API key missing/invalid"
        )
        st.stop()

    returns = prices.pct_change().dropna()

    # ---------- % Change ----------
    st.subheader("📊 Percentage Change Comparison")
    pct_change = (prices.divide(prices.iloc[0]) - 1) * 100

    fig, ax = plt.subplots()
    pct_change.plot(ax=ax)
    ax.set_ylabel("% Change")
    st.pyplot(fig)
    plt.clf()

    # ---------- Histogram ----------
    st.subheader("📈 Returns Histogram")
    returns.plot(kind="hist", bins=40, alpha=0.7)
    st.pyplot(plt.gcf())
    plt.clf()

    # ---------- Correlation ----------
    st.subheader("🔥 Correlation Heatmap")
    sns.set_theme()
    fig, ax = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.clf()

    # ---------- Portfolio Optimization ----------
    st.subheader("🧮 Portfolio Optimization")

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(len(tickers_list))
        weights /= np.sum(weights)
        weights_record.append(weights)

        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = ret
        results[1, i] = vol
        results[2, i] = ret / vol

    idx = np.argmax(results[2])
    opt_weights = weights_record[idx]

    st.markdown("### ✅ Optimal Portfolio Weights")
    st.dataframe(
        pd.DataFrame({
            "Ticker": tickers_list,
            "Weight": opt_weights
        })
    )

    st.markdown(f"**Expected Return:** {results[0, idx]*100:.2f}%")
    st.markdown(f"**Volatility:** {results[1, idx]*100:.2f}%")

    fig, ax = plt.subplots()
    ax.scatter(results[1], results[0], c=results[2], cmap="viridis", s=5)
    ax.scatter(results[1, idx], results[0, idx], c="red", marker="*", s=200)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    st.pyplot(fig)
    plt.clf()

else:
    st.info("👈 Enter inputs and click **Run Analysis**")


