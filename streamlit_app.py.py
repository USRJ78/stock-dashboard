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
st.title("📈 Portfolio Simulation & Optimization (INR)")

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

run_mc = st.sidebar.checkbox("Run Monte Carlo Optimization")
num_simulations = st.sidebar.number_input(
    "Number of Monte Carlo Simulations",
    min_value=500,
    max_value=20000,
    value=5000,
    step=500,
    disabled=not run_mc
)

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
        st.error("❌ No valid data returned. Try fewer tickers or wait a minute.")
        st.stop()

    returns = prices.pct_change().dropna()

    # ---------------- BASE RANDOM PORTFOLIO ----------------
    n = prices.shape[1]
    base_weights = np.random.random(n)
    base_weights /= base_weights.sum()

    allocation_df = pd.DataFrame({
        "Stock": prices.columns,
        "Weight": base_weights,
        "Allocated Amount (₹)": base_weights * initial_amount
    })

    shares = (base_weights * initial_amount) / prices.iloc[0]
    portfolio_value = (prices * shares).sum(axis=1)

    # ---------------- DISPLAY BASE ----------------
    st.subheader("🧮 Random Portfolio Allocation")
    st.dataframe(allocation_df.style.format({
        "Weight": "{:.2%}",
        "Allocated Amount (₹)": "₹{:,.0f}"
    }))

    # ---------------- PRICES ----------------
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

    # ---------------- MONTE CARLO (OPTIONAL) ----------------
    if run_mc:
        st.subheader("🎯 Monte Carlo Portfolio Optimization")

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        results = np.zeros((3, num_simulations))
        weight_store = []

        for i in range(num_simulations):
            w = np.random.random(n)
            w /= np.sum(w)
            weight_store.append(w)

            port_return = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            sharpe = port_return / port_vol

            results[0, i] = port_return
            results[1, i] = port_vol
            results[2, i] = sharpe

        max_idx = np.argmax(results[2])
        best_weights = weight_store[max_idx]

        # ---------------- MC SCATTER ----------------
        fig6, ax6 = plt.subplots()
        scatter = ax6.scatter(
            results[1],
            results[0],
            c=results[2],
            cmap="viridis",
            s=6
        )

        ax6.scatter(
            results[1, max_idx],
            results[0, max_idx],
            color="red",
            marker="*",
            s=300,
            label="Highest Sharpe Ratio"
        )

        ax6.set_xlabel("Volatility (Risk)")
        ax6.set_ylabel("Expected Return")
        ax6.legend()
        fig6.colorbar(scatter, label="Sharpe Ratio")
        st.pyplot(fig6)

        # ---------------- BEST WEIGHTS ----------------
        st.subheader("🏆 Best Monte Carlo Portfolio Weights")

        best_df = pd.DataFrame({
            "Stock": prices.columns,
            "Weight": best_weights,
            "Allocated Amount (₹)": best_weights * initial_amount
        })

        st.dataframe(best_df.style.format({
            "Weight": "{:.2%}",
            "Allocated Amount (₹)": "₹{:,.0f}"
        }))

else:
    st.info("👈 Enter inputs and click **Run Analysis**")
