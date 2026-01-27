import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from difflib import get_close_matches
from datetime import date

st.set_page_config(page_title="Notebook Accurate Stock App", layout="wide")

st.title("📈 Stock Price Analysis (Notebook Accurate)")
st.markdown("This app reproduces **exact notebook graphs only**.")

# ------------------ Helpers ------------------

@st.cache_data(ttl=3600)
def load_nse_stock_list():
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    df = pd.read_csv(url)
    df["SYMBOL"] = df["SYMBOL"].astype(str) + ".NS"
    return dict(zip(df["NAME OF COMPANY"].str.upper(), df["SYMBOL"]))

@st.cache_data(ttl=3600)
def resolve_assets(user_inputs):
    stock_map = load_nse_stock_list()
    resolved = {}

    for item in user_inputs:
        key = item.upper().strip()

        if "." in key:
            resolved[item] = key
            continue

        matches = get_close_matches(key, stock_map.keys(), n=1, cutoff=0.6)
        resolved[item] = stock_map[matches[0]] if matches else None

    return resolved

@st.cache_data(ttl=300)
def load_prices(ticker, start, end):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )
    return df.dropna()

# ------------------ Sidebar ------------------

st.sidebar.header("Inputs")

search_options = list(load_nse_stock_list().keys())

selected_assets = st.sidebar.multiselect(
    "🔍 Search & select stock (Notebook style = 1 stock)",
    options=search_options
)

start_date = st.sidebar.date_input("Start Date", date(2021, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

run = st.sidebar.button("Run Analysis")

# ------------------ Main ------------------

if run:
    if not selected_assets:
        st.error("❌ Select at least one stock")
        st.stop()

    resolved = resolve_assets(selected_assets)
    ticker = list(resolved.values())[0]

    df = load_prices(ticker, start_date, end_date)

    if df.empty:
        st.error("❌ No data fetched")
        st.stop()

    # -------- Daily Return (EXACT NOTEBOOK LOGIC) --------
    df["Daily Return"] = df["Close"].pct_change(1) * 100
    df["Daily Return"].replace(np.nan, 0, inplace=True)

    st.subheader("📄 Data Snapshot")
    st.dataframe(df.tail())

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe().round(2))

    # -------- NOTEBOOK GRAPH 1 --------
    st.subheader("📈 Adjusted Close Price (Notebook Graph)")

    fig = px.line(title=f"{ticker} Price")
    fig.add_scatter(
        x=df.index,
        y=df["Close"],
        name="Adj Close"
    )
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    # -------- NOTEBOOK GRAPH 2 (Generic function behavior) --------
    st.subheader("📉 Financial Data Plot (Notebook Function)")

    plot_df = df[["Open", "High", "Low", "Close"]]

    fig2 = px.line(title="Financial Price Lines")
    for col in plot_df.columns:
        fig2.add_scatter(
            x=plot_df.index,
            y=plot_df[col],
            name=col
        )

    fig2.update_traces(line_width=4)
    fig2.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("👈 Select stock and click Run Analysis")
