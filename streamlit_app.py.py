# (Updated code without rapidfuzz)
# Uses difflib instead of rapidfuzz to avoid extra dependencies

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
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
        resolved[item] = stock_map[matches[0]] if matches else None

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

# -------- Plot function --------
def plot_financial_data(df, title):
    fig = px.line(title=title)
    for col in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[col], name=col)

    fig.update_traces(line_width=3)
    fig.update_layout({'plot_bgcolor': "white"})
    st.plotly_chart(fig, use_container_width=True)

# -------- Scaling function --------
def price_scaling(raw_prices_df):
    scaled_prices_df = raw_prices_df.copy()
    for i in raw_prices_df.columns[1:]:
        scaled_prices_df[i] = raw_prices_df[i] / raw_prices_df[i].iloc[0]
    return scaled_prices_df

# ------------------ Sidebar ------------------

st.sidebar.header("Inputs")

@st.cache_data(ttl=3600)
def load_search_options():
    stock_map = load_nse_stock_list()
    etfs = list(ETF_MAP.keys())
    stocks = list(stock_map.keys())
    return sorted(stocks + etfs)

search_options = load_search_options()

selected_assets = st.sidebar.multiselect("🔍 Search & select stocks / ETFs (recommended)", options=search_options)
manual_assets = st.sidebar.text_input("✍️ Or manually type names / tickers (comma separated)", "")
initial_amount = st.sidebar.number_input("Initial Investment (INR)", value=100000, step=10000)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run_mc = st.sidebar.checkbox("Run Monte Carlo Simulation")
num_sims = st.sidebar.number_input("No. of simulations", 1000, 20000, 5000, step=1000)

run = st.sidebar.button("Run Analysis")

# ------------------ Main ------------------

if run:
    user_assets = selected_assets + [x.strip() for x in manual_assets.split(",") if x.strip()]
    if not user_assets:
        st.error("❌ Please select or enter at least one asset")
        st.stop()

    resolved = resolve_assets(user_assets)
    valid = {k: v for k, v in resolved.items() if v}

    if not valid:
        st.error("❌ No valid assets resolved")
        st.stop()

    prices = load_prices(list(valid.values()), start_date, end_date)
    returns = prices.pct_change().dropna()

    # -------- Allocation --------
    weights = np.random.random(len(prices.columns))
    weights /= weights.sum()
    allocation = initial_amount * weights

    portfolio_positions = (prices / prices.iloc[0]) * allocation
    portfolio_value = portfolio_positions.sum(axis=1)

    portfolio_df = portfolio_positions.copy()
    portfolio_df["Portfolio Value [$]"] = portfolio_value
    portfolio_df["Portfolio Daily Return [%]"] = portfolio_value.pct_change() * 100
    portfolio_df["Date"] = portfolio_df.index
    portfolio_df = portfolio_df[["Date"] + [c for c in portfolio_df.columns if c != "Date"]]

    # -------- Scaled Price Change --------
    st.subheader("📊 Percentage Change (Scaled Prices)")
    scaled_prices_df = prices.copy()
    scaled_prices_df["Date"] = scaled_prices_df.index
    scaled_prices_df = scaled_prices_df[["Date"] + list(prices.columns)]
    scaled_prices_df = price_scaling(scaled_prices_df)
    plot_financial_data(scaled_prices_df, "Scaled Price Change (Base = 1.0)")

    # -------- Price Movement --------
    st.subheader("📈 Price Movement (Actual Prices)")
    raw_prices_df = prices.copy()
    raw_prices_df["Date"] = raw_prices_df.index
    raw_prices_df = raw_prices_df[["Date"] + list(prices.columns)]
    plot_financial_data(raw_prices_df, "Price Movement (Actual Prices)")

    # -------- Portfolio Positions --------
    st.subheader("💼 Portfolio Positions (INR)")
    plot_financial_data(
        portfolio_df.drop(['Portfolio Value [$]', 'Portfolio Daily Return [%]'], axis=1),
        'Portfolio positions [$]'
    )

    # -------- Portfolio Value Over Time --------
    st.subheader("💼 Total Portfolio Value Over Time")
    plot_financial_data(
        portfolio_df[['Date', 'Portfolio Value [$]']],
        'Total Portfolio Value [$]'
    )

    # -------- Daily Returns --------
    st.subheader("📉 Daily Returns (%)")
    daily_returns_df = returns * 100
    daily_returns_df["Date"] = daily_returns_df.index
    daily_returns_df = daily_returns_df[["Date"] + list(returns.columns)]
    plot_financial_data(daily_returns_df, 'Percentage Daily Returns [%]')

    # -------- Correlation Heatmap (Seaborn) --------
    st.subheader("🔥 Correlation Heatmap")

    plt.figure(figsize=(10, 8))
    sns.heatmap(daily_returns_df.drop(columns=['Date']).corr(), annot=True)
    st.pyplot(plt.gcf())
    plt.close()

    # -------- Histogram --------
    st.subheader("📊 Daily % Change Distribution (Histogram)")
    fig = px.histogram(daily_returns_df.drop(columns=["Date"]))
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
