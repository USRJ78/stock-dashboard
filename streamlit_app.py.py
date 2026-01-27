import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
from datetime import date

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Stock / ETF Dashboard", layout="wide")

st.title("📊 Stock / ETF Analysis Dashboard")

# ---------------- SEARCH BAR ----------------
st.sidebar.header("🔍 Search")

search_query = st.sidebar.text_input(
    "Enter Stock / ETF name or ticker",
    placeholder="Reliance, NIFTY ETF, SPY, AAPL..."
)

start_date = st.sidebar.date_input("Start Date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", date.today())

# ---------------- TICKER RESOLUTION ----------------
def resolve_ticker(query):
    query = query.upper().strip()
    return query  # Yahoo Finance auto-resolves names & ETFs

# ---------------- FETCH DATA ----------------
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    return df

if search_query:
    ticker = resolve_ticker(search_query)

    try:
        data = load_data(ticker, start_date, end_date)

        if data.empty:
            st.error("❌ No data found. Try another stock or ETF.")
            st.stop()

        # ---------------- CALCULATIONS ----------------
        data["Daily Return %"] = data["Close"].pct_change() * 100

        latest_price = round(data["Close"].iloc[-1], 2)
        latest_change = round(data["Daily Return %"].iloc[-1], 2)

        # ---------------- METRICS ----------------
        col1, col2 = st.columns(2)
        col1.metric("Latest Price", f"₹ {latest_price}")
        col2.metric("Daily Change (%)", f"{latest_change} %")

        # ---------------- PRICE CHART ----------------
        st.subheader("📈 Price Chart")

        fig_price = px.line(
            data,
            x=data.index,
            y="Close",
            title=f"{ticker} Price"
        )
        st.plotly_chart(fig_price, use_container_width=True)

        # ---------------- DAILY RETURN CHART ----------------
        st.subheader("📉 Daily Percentage Change")

        fig_return = px.bar(
            data,
            x=data.index,
            y="Daily Return %",
            title="Daily % Change"
        )
        st.plotly_chart(fig_return, use_container_width=True)

        # ---------------- HEATMAP ----------------
        st.subheader("🔥 Monthly Return Heatmap")

        heatmap_df = data["Daily Return %"].resample("M").mean()
        heatmap_df = heatmap_df.to_frame(name="Avg Return %")
        heatmap_df["Month"] = heatmap_df.index.month
        heatmap_df["Year"] = heatmap_df.index.year

        pivot = heatmap_df.pivot("Year", "Month", "Avg Return %")

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot, cmap="RdYlGn", annot=False, ax=ax)
        st.pyplot(fig)

        # ---------------- NORMAL DISTRIBUTION ----------------
        st.subheader("📊 Normal Distribution of Daily Returns")

        returns = data["Daily Return %"].dropna()

        fig, ax = plt.subplots()
        sns.histplot(returns, bins=50, kde=True, ax=ax)

        mu, std = stats.norm.fit(returns)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, std)
        ax.plot(x, p * len(returns) * (xmax - xmin) / 50)

        st.pyplot(fig)

        # ---------------- SAVE TODAY'S DATA ----------------
        st.subheader("💾 Save Today's Data")

        today_df = data.tail(1)
        csv = today_df.to_csv().encode("utf-8")

        st.download_button(
            label="Download Today's Data",
            data=csv,
            file_name=f"{ticker}_today.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("⚠️ Something went wrong.")
        st.exception(e)

else:
    st.info("👈 Enter a stock or ETF name in the sidebar to begin.")
