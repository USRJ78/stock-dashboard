import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import date

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Portfolio Simulator", layout="wide")
st.title("📈 Portfolio Simulation & Efficient Frontier (INR)")

# ---------------- INPUTS ----------------
st.sidebar.header("Inputs")

initial_amount = st.sidebar.number_input(
    "Initial Investment Amount (₹)",
    min_value=1000,
    value=100000,
    step=1000
)

tickers_input = st.sidebar.text_input(
    "Stock Tickers (NSE, comma separated)",
    "RELIANCE,TCS,INFY"
)

start_date = st.sidebar.date_input("Start Date", date(2022, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run = st.sidebar.button("Run Analysis")

tickers = [t.strip().upper() + ".NS" for t in tickers_input.split(",") if t.strip()]

# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=3600)
def load_prices(tickers, start, end):
    prices = {}

    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(
                start=start,
                end=end,
                auto_adjust=True
            )

            if not df.empty:
                prices[ticker.replace(".NS", "")] = df["Close"]

            time.sleep(1)

        except Exception:
            continue

    if len(prices) == 0:
        return pd.DataFrame()

    return pd.DataFrame(prices)

# ---------------- MAIN ----------------
if run:
    prices = load_prices(tickers, start_date, end_date)

    if prices.empty:
        st.error("""
❌ No valid data returned.

Possible reasons:
- Yahoo Finance rate-limited NSE
- Invalid ticker
- Too many refreshes

👉 Try fewer tickers or wait 1 minute.
""")
        st.stop()

    returns = prices.pct_change().dropna()

    # ---------------- BASIC PORTFOLIO ----------------
    n = prices.shape[1]
    weights = np.random.random(n)
    weights /= weights.sum()

    allocation_df = pd.DataFrame({
        "Stock": prices.columns,
        "Weight": weights,
        "Allocated Amount (₹)": weights * initial_amount
    })

    shares = (weights * initial_amount) / prices.iloc[0]
    portfolio_value = (prices * shares).sum(axis=1)

    # ---------------- MONTE CARLO PORTFOLIOS ----------------
    num_portfolios = 5000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252

    for i in range(num_portfolios):
        w = np.random.random(n)
        w /= np.sum(w)
        weights_record.append(w)

        port_return = np.dot(w, mean_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sharpe = port_return / port_vol

        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = sharpe

    max_sharpe_idx = np.argmax(results[2])
    max_sharpe_weights = weights_record[max_sharpe_idx]

    # ---------------- DISPLAY ----------------
    st.subheader("🧮 Random Portfolio Allocation")
    st.dataframe(allocation_df.style.format({
        "Weight": "{:.2%}",
        "Allocated Amount (₹)": "₹{:,.0f}"
    }))

    # ---------------- PRICE CHART ----------------
    st.subheader("📊 Stock Prices")
    fig1, ax1 = plt.subplots()
    prices.plot(ax=ax1)
    ax1.set_ylabel("Price (₹)")
    ax1.grid(True)
    st.pyplot(fig1)

    # ---------------- % CHANGE ----------------
    st.subheader("📈 Percentage Change")
    pct_change = (prices / prices.iloc[0] - 1) * 100
    fig2, ax2 = plt.subplots()
    pct_change.plot(ax=ax2)
    ax2.set_ylabel("Change (%)")
    ax2.grid(True)
    st.pyplot(fig2)

    # ---------------- RETURNS ----------------
    st.subheader("📉 Daily Returns")
    fig3, ax3 = plt.subplots()
    returns.plot(ax=ax3)
    ax3.set_ylabel("Daily Return")
    ax3.grid(True)
    st.pyplot(fig3)

    # ---------------- CORRELATION ----------------
    st.subheader("🔥 Correlation Heatmap")
    fig4, ax4 = plt.subplots()
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax4)
    st.pyplot(fig4)

    # ---------------- PORTFOLIO VALUE ----------------
    st.subheader("📈 Portfolio Value Over Time")
    fig5, ax5 = plt.subplots()
    portfolio_value.plot(ax=ax5, color="black")
    ax5.set_ylabel("Portfolio Value (₹)")
    ax5.grid(True)
    st.pyplot(fig5)

    # ---------------- EFFICIENT FRONTIER ----------------
    st.subheader("🎯 Efficient Frontier (Max Sharpe Highlighted)")

    fig6, ax6 = plt.subplots()
    scatter = ax6.scatter(
        results[1],
        results[0],
        c=results[2],
        cmap="viridis",
        s=5
    )

    ax6.scatter(
        results[1, max_sharpe_idx],
        results[0, max_sharpe_idx],
        color="red",
        marker="*",
        s=250,
        label="Max Sharpe Ratio"
    )

    ax6.set_xlabel("Volatility (Risk)")
    ax6.set_ylabel("Expected Return")
    ax6.legend()
    fig6.colorbar(scatter, label="Sharpe Ratio")
    st.pyplot(fig6)

else:
    st.info("👈 Enter inputs and click **Run Analysis**")
