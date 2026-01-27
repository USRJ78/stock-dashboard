# (Updated code without rapidfuzz)
# Uses difflib instead of rapidfuzz to avoid extra dependencies

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from difflib import get_close_matches
from datetime import date

st.set_page_config(page_title="Universal Market App", layout="wide")

st.title("📊 Universal Stock & ETF Portfolio App")
st.markdown("Search by **name or ticker**, allocate capital, and run portfolio simulations.")

# ------------------ Run-state fix ------------------
# This makes the app automatically re-run analysis whenever sidebar inputs change,
# so graphs ALWAYS update with Start/End date changes (no stale charts).
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

def trigger_run():
    st.session_state.run_analysis = True

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
        elif key in ETF_MAP:
            resolved[item] = ETF_MAP[key]
        else:
            matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6)
            resolved[item] = stock_map[matches[0]] if matches else None
    return resolved

@st.cache_data(ttl=300)
def load_prices(tickers, start, end):
    # caching is safe: when start/end change, cache key changes -> new data
    tickers = sorted(list(set(tickers)))
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna()
    data.index = pd.to_datetime(data.index)
    return data

def plot_financial_data(df, title):
    fig = px.line(title=title)
    for col in df.columns[1:]:
        fig.add_scatter(x=df['Date'], y=df[col], name=col)
    fig.update_traces(line_width=3)
    fig.update_layout({'plot_bgcolor': "white"})
    st.plotly_chart(fig, use_container_width=True)

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
    return sorted(list(stock_map.keys()) + list(ETF_MAP.keys()))

search_options = load_search_options()

selected_assets = st.sidebar.multiselect(
    "🔍 Search & select stocks / ETFs (recommended)",
    options=search_options,
    key="selected_assets",
    on_change=trigger_run
)

manual_assets = st.sidebar.text_input(
    "✍️ Or manually type names / tickers (comma separated)",
    "",
    key="manual_assets",
    on_change=trigger_run
)

initial_amount = st.sidebar.number_input(
    "Initial Investment (INR)",
    value=100000,
    step=10000,
    key="initial_amount",
    on_change=trigger_run
)

start_date = st.sidebar.date_input(
    "Start Date",
    date(2021, 1, 1),
    key="start_date",
    on_change=trigger_run
)

end_date = st.sidebar.date_input(
    "End Date",
    date.today(),
    key="end_date",
    on_change=trigger_run
)

run_mc = st.sidebar.checkbox(
    "Run Monte Carlo Simulation",
    key="run_mc",
    on_change=trigger_run
)

num_sims = st.sidebar.number_input(
    "No. of simulations",
    1000,
    20000,
    5000,
    step=1000,
    key="num_sims",
    on_change=trigger_run
)

# Still keep your button, but now date changes ALSO trigger run automatically
if st.sidebar.button("Run Analysis", key="run_button"):
    st.session_state.run_analysis = True

# ------------------ Main ------------------

if st.session_state.run_analysis:

    # basic date validation
    if end_date <= start_date:
        st.error("❌ End Date must be after Start Date")
        st.stop()

    user_assets = list(selected_assets) + [x.strip() for x in manual_assets.split(",") if x.strip()]
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
        st.error("❌ No price data fetched (try different dates / tickers)")
        st.stop()

    returns = prices.pct_change().dropna()

    # -------- Allocation --------
    weights = np.random.random(len(prices.columns))
    weights /= weights.sum()
    allocation = float(initial_amount) * weights

    alloc_df = pd.DataFrame({
        "Asset": prices.columns,
        "Weight": weights,
        "Allocation (INR)": allocation
    })

    st.subheader("💰 Portfolio Allocation")
    st.dataframe(alloc_df)

    # -------- Portfolio calcs --------
    portfolio_positions = (prices / prices.iloc[0]) * allocation
    portfolio_value = portfolio_positions.sum(axis=1)

    portfolio_df = portfolio_positions.copy()
    portfolio_df["Portfolio Value [$]"] = portfolio_value
    portfolio_df["Portfolio Daily Return [%]"] = portfolio_value.pct_change() * 100
    portfolio_df["Date"] = portfolio_df.index
    portfolio_df = portfolio_df[["Date"] + [c for c in portfolio_df.columns if c != "Date"]]

    # -------- Percentage Change (Scaled Prices) --------
    st.subheader("📊 Percentage Change (Scaled Prices)")
    scaled_prices_df = prices.copy()
    scaled_prices_df["Date"] = scaled_prices_df.index
    scaled_prices_df = scaled_prices_df[["Date"] + list(prices.columns)]
    scaled_prices_df = price_scaling(scaled_prices_df)
    plot_financial_data(scaled_prices_df, "Scaled Price Change (Base = 1.0)")

    # -------- Price Movement (Actual Prices) --------
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

    # -------- Heatmap --------
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

    # -------- Monte Carlo (your exact plot + optimal point) --------
    if run_mc:
        st.subheader("🎯 Monte Carlo Simulation")

        mean_returns = returns.mean() * 252
        cov = returns.cov() * 252

        sim_results = []
        weight_list = []

        for _ in range(int(num_sims)):
            w = np.random.random(len(prices.columns))
            w /= w.sum()
            weight_list.append(w)

            port_return = float(np.dot(w, mean_returns))
            port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov, w))))
            sharpe = (port_return / port_vol) if port_vol != 0 else np.nan

            sim_results.append([port_return, port_vol, sharpe])

        sim_out_df = pd.DataFrame(sim_results, columns=["Portfolio_Return", "Volatility", "Sharpe_Ratio"])

        sharpe_series = sim_out_df["Sharpe_Ratio"].replace([np.inf, -np.inf], np.nan)
        optimal_idx = sharpe_series.idxmax()

        optimal_portfolio_return = float(sim_out_df.loc[optimal_idx, "Portfolio_Return"])
        optimal_volatility = float(sim_out_df.loc[optimal_idx, "Volatility"])

        fig = px.scatter(
            sim_out_df,
            x='Volatility',
            y='Portfolio_Return',
            color='Sharpe_Ratio',
            size='Sharpe_Ratio',
            hover_data=['Sharpe_Ratio']
        )

        fig.add_trace(go.Scatter(
            x=[optimal_volatility],
            y=[optimal_portfolio_return],
            mode='markers',
            name='Optimal Point',
            marker=dict(size=[40], color='red')
        ))

        fig.update_layout(coloraxis_colorbar=dict(y=0.7, dtick=5))
        fig.update_layout({'plot_bgcolor': "white"})
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("✅ Optimal Portfolio Weights (Max Sharpe)")
        best_df = pd.DataFrame({
            "Asset": prices.columns,
            "Weight": weight_list[int(optimal_idx)]
        })
        st.dataframe(best_df)

else:
    st.info("👈 Select assets / change dates — graphs will auto-update. (You can also click Run Analysis.)")
