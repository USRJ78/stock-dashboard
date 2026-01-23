import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Portfolio Simulator", layout="wide")
st.title("📈 Portfolio Simulation & Analysis (INR)")

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

tickers = [t.strip().upper() + ".NS" for t in tickers_input.split(",") if t.strip()]

run = st.sidebar.button("Run Portfolio Simulation")

# ---------------- DATA FETCH ----------------
@st.cache_data(ttl=3600)
def load_prices(tickers):
    prices = {}

    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(
                period="1y",
                auto_adjust=True
            )

            if not df.empty:
                prices[ticker.replace(".NS", "")] = df["Close"]

            time.sleep(1)  # prevent rate limit

        except Exception:
            continue

    if len(prices) == 0:
        return pd.DataFrame()

    return pd.DataFrame(prices)

# ---------------- MAIN ----------------
if run:
    prices = load_prices(tickers)

    if prices.empty:
        st.error("""
❌ No valid data returned.

Possible reasons:
- Yahoo Finance rate-limited NSE
- Invalid ticker
- Too many refreshes

👉 Try 1–2 tickers or wait 1 minute.
""")
        st.stop()

    returns = prices.pct_change().dropna()

    # ---------------- RANDOM WEIGHTS ----------------
    n = prices.shape[1]
    weights = np.random.random(n)
    weights /= weights.sum()

    weight_df = pd.DataFrame({
        "Stock": prices.columns,
        "Weight": weights,
        "Allocated Amount (₹)": weights * initial_amount
    })

    # ---------------- PORTFOLIO VALUE ----------------
    shares = (weights * initial_amount) / prices.iloc[0]
    portfolio_value = (prices * shares).sum(axis=1)

    # ---------------- DISPLAY ----------------
    st.subheader("🧮 Portfolio Allocation")
    st.dataframe(weight_df.style.format({
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

    # ---------------- PERCENT CHANGE ----------------
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

    # ---------------- PORTFOLIO GROWTH ----------------
    st.subheader("📈 Portfolio Value Over Time")
    fig5, ax5 = plt.subplots()
    portfolio_value.plot(ax=ax5, color="black")
    ax5.set_ylabel("Portfolio Value (₹)")
    ax5.grid(True)
    st.pyplot(fig5)

    # ---------------- PIE CHART ----------------
    st.subheader("🎯 Weight Allocation")
    fig6, ax6 = plt.subplots()
    ax6.pie(weights, labels=prices.columns, autopct="%1.1f%%")
    ax6.axis("equal")
    st.pyplot(fig6)

else:
    st.info("👈 Enter inputs in the sidebar and click **Run Portfolio Simulation**")
