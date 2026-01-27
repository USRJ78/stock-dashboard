# (Updated code without rapidfuzz)
# Uses difflib instead of rapidfuzz to avoid extra dependencies

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import plotly.express as px
from difflib import get_close_matches
from datetime import date

st.set_page_config(page_title="Universal Market App", layout="wide")

st.title("📊 Universal Stock & ETF Portfolio App")
st.markdown("Search by **name or ticker**, allocate capital, and run portfolio simulations.")

# ------------------ Helpers ------------------

@st.cache_data(ttl=3600)
def load_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df["SYMBOL"] = df["SYMBOL"].astype(str) + ".NS"
    return dict(zip(df["NAME OF COMPANY"].str.upper(), df["SYMBOL"]))

ETF_MAP = {
    "NIFTY 50 ETF": "NIFTYBEES.NS",
    "BANK NIFTY ETF": "BANKBEES.NS",
    "GOLD ETF": "GOLDBEES.NS",
    "IT ETF": "ITBEES.NS",
}

@st.cache_data(ttl=3600)
def resolve_assets(user_inputs):
    stock_map = load_nse_stock_list()
    resolved = {}

    for item in user_inputs:
        key = item.upper().strip()

        if "." in key:
            resolved[item] = key
            continue

        if key in ETF_MAP:
            resolved[item] = ETF_MAP[key]
            continue

        matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6)
        if matches:
            resolved[item] = stock_map[matches[0]]
        else:
            resolved[item] = None

    return resolved

@st.cache_data(ttl=300)
def load_prices(tickers, start, end):
    tickers = sorted(list(set(tickers)))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data

# ------------------ Sidebar ------------------

st.sidebar.header("Inputs")

@st.cache_data(ttl=3600)
def load_search_options():
    stock_map = load_nse_stock_list()
    etfs = list(ETF_MAP.keys())
    stocks = list(stock_map.keys())
    return sorted(stocks + etfs)

search_options = load_search_options()

selected_assets = st.sidebar.multiselect(
    "🔍 Search & select stocks / ETFs (recommended)",
    options=search_options
)

manual_assets = st.sidebar.text_input(
    "✍️ Or manually type names / tickers (comma separated)",
    ""
)

initial_amount = st.sidebar.number_input("Initial Investment (INR)", value=100000, step=10000)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run_mc = st.sidebar.checkbox("Run Monte Carlo Simulation")
num_sims = st.sidebar.number_input("No. of simulations", 1000, 20000, 5000, step=1000)

run = st.sidebar.button("Run Analysis")

# ------------------ Main ------------------

if run:
    user_assets = []

    if selected_assets:
        user_assets.extend(selected_assets)

    if manual_assets.strip():
        user_assets.extend([x.strip() for x in manual_assets.split(",") if x.strip()])

    if not user_assets:
        st.error("❌ Please select or enter at least one asset")
        st.stop()

    resolved = resolve_assets(user_assets)

    valid = {k: v for k, v in resolved.items() if v}
    invalid = [k for k, v in resolved.items() if not v]

    if invalid:
        st.warning(f"⚠️ Could not resolve: {', '.join(invalid)}")

    if not valid:
        st.error("❌ No valid assets resolved")
        st.stop()

    st.subheader("Resolved Assets")
    st.write(valid)

    prices = load_prices(list(valid.values()), start_date, end_date)

    if prices.empty:
        st.error("❌ No price data fetched")
        st.stop()

    # SAME DATA SOURCE FOR EVERYTHING
    returns = prices.pct_change().dropna()

    # -------- Random Allocation --------
    weights = np.random.random(len(prices.columns))
    weights /= weights.sum()

    allocation = initial_amount * weights
    alloc_df = pd.DataFrame({
        "Asset": prices.columns,
        "Weight": weights,
        "Allocation (INR)": allocation
    })

    st.subheader("💰 Portfolio Allocation")
    st.dataframe(alloc_df)

    # -------- Percentage Change --------
    st.subheader("📊 Percentage Change (%)")
    pct_change = (prices / prices.iloc[0] - 1) * 100
    fig, ax = plt.subplots()
    pct_change.plot(ax=ax)
    ax.set_ylabel("% Change")
    st.pyplot(fig)

    # -------- Price Levels --------
    st.subheader("📈 Price Movement")
    fig, ax = plt.subplots()
    prices.plot(ax=ax)
    ax.set_ylabel("Price")
    st.pyplot(fig)

    # -------- Correlation Heatmap --------
    st.subheader("🔥 Correlation Heatmap")
    corr = returns.corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)

    # -------- Portfolio Value --------
    portfolio_value = (prices / prices.iloc[0]) @ allocation

    st.subheader("💼 Portfolio Value Over Time")
    fig, ax = plt.subplots()
    ax.plot(portfolio_value)
    ax.set_ylabel("Portfolio Value (INR)")
    st.pyplot(fig)

    # -------- Histogram (SAME RETURNS DATA) --------
    st.subheader("📊 Daily % Change Distribution (Histogram)")

    daily_returns_df = returns * 100
    daily_returns_df["Date"] = daily_returns_df.index

    hist_assets = st.multiselect(
        "Select assets for histogram",
        options=[c for c in daily_returns_df.columns if c != "Date"],
        default=[c for c in daily_returns_df.columns if c != "Date"][:1]
    )

    if hist_assets:
        fig = px.histogram(
            daily_returns_df[["Date"] + hist_assets].drop(columns=["Date"])
        )
        fig.update_layout({'plot_bgcolor': "white"})
        st.plotly_chart(fig, use_container_width=True)

    # -------- Monte Carlo --------
    if run_mc:
        st.subheader("🎯 Monte Carlo Simulation")

        mean_returns = returns.mean() * 252
        cov = returns.cov() * 252

        results = np.zeros((3, num_sims))
        weight_list = []

        for i in range(num_sims):
            w = np.random.random(len(prices.columns))
            w /= w.sum()
            weight_list.append(w)

            ret = np.dot(w, mean_returns)
            vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            sharpe = ret / vol

            results[:, i] = [ret, vol, sharpe]

        idx = results[2].argmax()

        fig, ax = plt.subplots()
        ax.scatter(results[1], results[0], c=results[2], cmap="viridis", s=5)
        ax.scatter(results[1, idx], results[0, idx], color="red", s=200, marker="*")
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        st.pyplot(fig)

        best_df = pd.DataFrame({
            "Asset": prices.columns,
            "Weight": weight_list[idx]
        })

        st.markdown("**Best Sharpe Ratio Portfolio Weights**")
        st.dataframe(best_df)

else:
    st.info("👈 Select assets and click Run Analysis")
