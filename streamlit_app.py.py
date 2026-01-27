# Universal Market App (Extended – additive only)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from difflib import get_close_matches
from datetime import date

st.set_page_config(page_title="Universal Market App", layout="wide")

st.title("📊 Universal Stock, ETF & Portfolio App")
st.markdown("Search by **name or ticker**, allocate capital, analyze risk, and simulate portfolios.")

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
        resolved[item] = stock_map[matches[0]] if matches else None

    return resolved

@st.cache_data(ttl=300)
def load_prices(tickers, start, end):
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker"
    )

    if isinstance(data, pd.DataFrame) and "Close" in data:
        data = data["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    return data.dropna()

# ------------------ Sidebar ------------------

st.sidebar.header("Inputs")

@st.cache_data(ttl=3600)
def load_search_options():
    stock_map = load_nse_stock_list()
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

selected_assets = st.sidebar.multiselect(
    "🔍 Search & select stocks / ETFs",
    load_search_options()
)

manual_assets = st.sidebar.text_input(
    "✍️ Or manually type names / tickers (comma separated)"
)

initial_amount = st.sidebar.number_input(
    "Initial Investment (INR)", value=100000, step=10000
)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run_mc = st.sidebar.checkbox("Run Monte Carlo Simulation")
num_sims = st.sidebar.number_input(
    "Monte Carlo simulations", 1000, 20000, 5000, step=1000
)

run = st.sidebar.button("Run Analysis")

# ------------------ Main ------------------

if run:
    user_assets = []
    user_assets += selected_assets
    if manual_assets:
        user_assets += [x.strip() for x in manual_assets.split(",") if x.strip()]

    if not user_assets:
        st.error("❌ No assets selected")
        st.stop()

    resolved = resolve_assets(user_assets)
    valid = {k: v for k, v in resolved.items() if v}

    if not valid:
        st.error("❌ No valid assets resolved")
        st.stop()

    prices = load_prices(list(valid.values()), start_date, end_date)

    if prices.empty:
        st.error("❌ No price data fetched")
        st.stop()

    returns = prices.pct_change().dropna()

    # -------- Allocation --------
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

    # -------- Price Movement --------
    st.subheader("📈 Price Movement")
    st.line_chart(prices)

    # -------- Percentage Change --------
    st.subheader("📊 Percentage Change (%)")
    pct_change = (prices / prices.iloc[0] - 1) * 100
    st.line_chart(pct_change)

    # -------- Correlation Heatmap --------
    st.subheader("🔥 Correlation Heatmap")
    corr = returns.corr()
    fig, ax = plt.subplots()
    im = ax.imshow(corr, cmap="coolwarm")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)

    # -------- Portfolio Value --------
    portfolio_value = (prices / prices.iloc[0]) @ allocation
    st.subheader("💼 Portfolio Value Over Time")
    st.line_chart(portfolio_value)

    # -------- Daily Returns Distribution --------
    st.subheader("📉 Daily Returns Distribution")
    port_returns = returns @ weights

    fig, ax = plt.subplots()
    ax.hist(port_returns, bins=40, density=True, alpha=0.6)
    mu, std = port_returns.mean(), port_returns.std()
    x = np.linspace(mu - 4*std, mu + 4*std, 200)
    ax.plot(x, stats.norm.pdf(x, mu, std))
    ax.set_title("Normal Distribution of Daily Returns")
    st.pyplot(fig)

    # -------- Risk Metrics --------
    st.subheader("⚠️ Risk Metrics")
    vol = port_returns.std() * np.sqrt(252)
    sharpe = (port_returns.mean() * 252) / vol
    max_dd = (portfolio_value / portfolio_value.cummax() - 1).min()

    st.write({
        "Annualized Volatility": f"{vol:.2%}",
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Max Drawdown": f"{max_dd:.2%}"
    })

    # -------- Monte Carlo --------
    if run_mc:
        st.subheader("🎯 Monte Carlo Efficient Frontier")

        mean_ret = returns.mean() * 252
        cov = returns.cov() * 252

        results = np.zeros((3, num_sims))
        weight_list = []

        for i in range(num_sims):
            w = np.random.random(len(prices.columns))
            w /= w.sum()
            weight_list.append(w)

            r = np.dot(w, mean_ret)
            v = np.sqrt(np.dot(w.T, np.dot(cov, w)))
            s = r / v
            results[:, i] = [r, v, s]

        idx = results[2].argmax()

        fig, ax = plt.subplots()
        ax.scatter(results[1], results[0], c=results[2], cmap="viridis", s=5)
        ax.scatter(results[1, idx], results[0, idx], color="red", marker="*", s=250)
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        st.pyplot(fig)

        st.markdown("**Best Sharpe Portfolio Weights**")
        st.dataframe(pd.DataFrame({
            "Asset": prices.columns,
            "Weight": weight_list[idx]
        }))

else:
    st.info("👈 Select assets and click Run Analysis")
