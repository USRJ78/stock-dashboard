# Streamlit Stock Analysis & Portfolio Dashboard
# Run with: streamlit run stock_dashboard_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

st.set_page_config(page_title="Stock Analysis & Portfolio Dashboard", layout="wide")

st.title("📈 Stock Analysis & Portfolio Dashboard")
st.markdown("Analyze stocks, compare returns, visualize correlations, and build an optimal portfolio.")

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("Inputs")

tickers = st.sidebar.text_input(
    "Enter stock tickers (comma separated)",
    "AAPL,MSFT,RELIANCE.NS"
)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run = st.sidebar.button("Run Analysis")

# ---------------- Data Functions ----------------
def load_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    return data.dropna()

# ---------------- Main Logic ----------------
if run:
    tickers_list = [t.strip().upper() for t in tickers.split(",")]

    prices = load_data(tickers_list, start_date, end_date)
    returns = prices.pct_change().dropna()

    # ---------- Percentage Change Plot ----------
    st.subheader("📊 Percentage Change Comparison")
    pct_change = (prices / prices.iloc[0] - 1) * 100

    fig, ax = plt.subplots()
    pct_change.plot(ax=ax)
    ax.set_ylabel("% Change")
    st.pyplot(fig)

    # ---------- Histogram ----------
    st.subheader("📈 Returns Histogram")
    fig, ax = plt.subplots()
    returns.hist(ax=ax, bins=40)
    st.pyplot(fig)

    # ---------- Correlation Heatmap ----------
    st.subheader("🔥 Correlation Heatmap")
    corr = returns.corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---------- Portfolio Optimization ----------
    st.subheader("🧮 Portfolio Optimization (Mean-Variance)")

    num_assets = len(tickers_list)
    num_portfolios = 5000

    results = np.zeros((3, num_portfolios))
    weights_record = []

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)

        portfolio_return = np.dot(weights, mean_returns)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std
        results[2, i] = portfolio_return / portfolio_std

    max_sharpe_idx = np.argmax(results[2])
    optimal_weights = weights_record[max_sharpe_idx]

    st.markdown("**Optimal Portfolio Weights:**")
    opt_df = pd.DataFrame({"Ticker": tickers_list, "Weight": optimal_weights})
    st.dataframe(opt_df)

    st.markdown(f"**Expected Annual Return:** {results[0, max_sharpe_idx]*100:.2f}%")
    st.markdown(f"**Expected Volatility:** {results[1, max_sharpe_idx]*100:.2f}%")

    fig, ax = plt.subplots()
    ax.scatter(results[1], results[0], c=results[2], cmap="viridis", s=5)
    ax.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx], color="red", marker="*", s=200)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    st.pyplot(fig)

else:
    st.info("👈 Enter inputs in the sidebar and click **Run Analysis**")