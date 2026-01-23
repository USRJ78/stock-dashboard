# FULL FEATURED STREAMLIT APP
# Universal Stock / ETF / MF Search + Indicators + Portfolio + Monte Carlo

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from rapidfuzz import process
import matplotlib.pyplot as plt

st.set_page_config(page_title="Universal Market App", layout="wide")

# ---------------- CACHE LOADERS ----------------
@st.cache_data
def load_mutual_funds():
    data = requests.get("https://api.mfapi.in/mf").json()
    df = pd.DataFrame(data)
    df["type"] = "mutual_fund"
    return df[["schemeName", "schemeCode", "type"]]

@st.cache_data
def load_nse_symbols():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df["symbol"] = df["SYMBOL"] + ".NS"
    df = df.rename(columns={"NAME OF COMPANY": "name"})
    df["type"] = "stock"
    return df[["name", "symbol", "type"]]

@st.cache_data
def load_etfs():
    etfs = ["NIFTYBEES.NS", "BANKBEES.NS", "GOLDBEES.NS", "SILVERBEES.NS"]
    rows = []
    for t in etfs:
        info = yf.Ticker(t).info
        rows.append({"name": info.get("longName"), "symbol": t, "type": "etf"})
    return pd.DataFrame(rows)

# ---------------- SEARCH RESOLVER ----------------
def resolve_asset(query, stocks, etfs, mfs):
    candidates = []

    for df in [stocks, etfs]:
        match = process.extractOne(query, df["name"], score_cutoff=70)
        if match:
            name, score, idx = match
            row = df.iloc[idx]
            candidates.append({"name": row["name"], "symbol": row["symbol"], "type": row["type"], "score": score})

    mf_match = process.extractOne(query, mfs["schemeName"], score_cutoff=70)
    if mf_match:
        name, score, idx = mf_match
        row = mfs.iloc[idx]
        candidates.append({"name": row["schemeName"], "symbol": row["schemeCode"], "type": "mutual_fund", "score": score})

    return max(candidates, key=lambda x: x["score"]) if candidates else None

# ---------------- PORTFOLIO ----------------
def monte_carlo_simulation(returns, sims):
    results = []
    for _ in range(sims):
        w = np.random.random(len(returns.columns))
        w /= np.sum(w)
        port_ret = np.sum(returns.mean() * w) * 252
        port_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
        sharpe = port_ret / port_vol
        results.append([port_ret, port_vol, sharpe, w])
    return results

# ---------------- UI ----------------
st.title("📈 Universal Market Intelligence Platform")

assets = st.multiselect("Search & Add Assets", options=[], placeholder="Type stock / ETF / MF name")
query = st.text_input("Add asset")

if query:
    stocks = load_nse_symbols()
    etfs = load_etfs()
    mfs = load_mutual_funds()

    result = resolve_asset(query, stocks, etfs, mfs)
    if result:
        st.success(f"Added {result['name']}")
        st.session_state.setdefault("assets", []).append(result)

# ---------------- PORTFOLIO INPUTS ----------------
initial_amount = st.number_input("Initial Investment (INR)", value=100000)
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if "assets" in st.session_state and st.button("Run Analysis"):
    tickers = [a["symbol"] for a in st.session_state["assets"] if a["type"] != "mutual_fund"]

    prices = yf.download(tickers, start=start_date, end=end_date)["Adj Close"].dropna()
    returns = prices.pct_change().dropna()

    # Random allocation
    weights = np.random.random(len(tickers))
    weights /= weights.sum()

    portfolio_value = (prices * weights * initial_amount).sum(axis=1)

    st.subheader("📊 Portfolio Value")
    st.line_chart(portfolio_value)

    # Volatility vs Return
    port_return = returns.mean() * 252
    port_vol = returns.std() * np.sqrt(252)

    fig, ax = plt.subplots()
    ax.scatter(port_vol, port_return)
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Return")
    st.pyplot(fig)

    # ---------------- MONTE CARLO ----------------
    sims = st.number_input("Monte Carlo Simulations", value=500, step=100)
    if st.button("Run Monte Carlo"):
        results = monte_carlo_simulation(returns, sims)
        df = pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe", "Weights"])

        max_sharpe = df.loc[df["Sharpe"].idxmax()]

        fig2, ax2 = plt.subplots()
        ax2.scatter(df["Volatility"], df["Return"], c=df["Sharpe"])
        ax2.scatter(max_sharpe["Volatility"], max_sharpe["Return"], color="red", s=200)
        st.pyplot(fig2)

        st.subheader("🏆 Best Sharpe Portfolio")
        st.write(dict(zip(tickers, max_sharpe["Weights"])))
