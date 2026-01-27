import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from datetime import date

st.set_page_config(page_title="Stock Dashboard", layout="wide")

# -------------------------------
# Title
# -------------------------------
st.title("📈 Stock / ETF Analysis Dashboard")

# -------------------------------
# Sidebar – ticker search
# -------------------------------
st.sidebar.header("Search Asset")

ticker = st.sidebar.text_input(
    "Enter Stock / ETF Ticker (Yahoo Finance)",
    value="AAPL"
).upper()

# -------------------------------
# Date selection
# -------------------------------
today = date.today()

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(2023, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=today
)

# -------------------------------
# Load data (CACHED)
# -------------------------------
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, period="max", auto_adjust=True)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

if ticker:
    df = load_data(ticker)
else:
    st.stop()

# -------------------------------
# Filter data by date
# -------------------------------
filtered_df = df[
    (df['Date'] >= pd.to_datetime(start_date)) &
    (df['Date'] <= pd.to_datetime(end_date))
]

if filtered_df.empty:
    st.warning("No data available for selected date range.")
    st.stop()

# -------------------------------
# Daily % Change
# -------------------------------
filtered_df['Daily % Change'] = filtered_df['Close'].pct_change() * 100
returns = filtered_df['Daily % Change'].dropna()

# -------------------------------
# Layout
# -------------------------------
col1, col2 = st.columns(2)

# -------------------------------
# Price chart
# -------------------------------
with col1:
    st.subheader("📊 Closing Price")
    st.line_chart(
        filtered_df.set_index('Date')['Close']
    )

# -------------------------------
# % Change chart
# -------------------------------
with col2:
    st.subheader("📉 Daily % Change")
    st.line_chart(
        filtered_df.set_index('Date')['Daily % Change']
    )

# -------------------------------
# Heatmap of returns
# -------------------------------
st.subheader("🔥 Monthly Return Heatmap")

heatmap_df = filtered_df.copy()
heatmap_df['Year'] = heatmap_df['Date'].dt.year
heatmap_df['Month'] = heatmap_df['Date'].dt.month

monthly_returns = (
    heatmap_df
    .groupby(['Year', 'Month'])['Daily % Change']
    .mean()
    .unstack()
)

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(
    monthly_returns,
    cmap="RdYlGn",
    center=0,
    ax=ax
)

st.pyplot(fig)

# -------------------------------
# Normal Distribution Plot
# -------------------------------
st.subheader("📈 Normal Distribution of Daily % Returns")

if len(returns) >= 10:
    fig2, ax2 = plt.subplots()
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title("Normal Probability Plot")
    st.pyplot(fig2)
else:
    st.warning("Not enough data to plot normal distribution.")

# -------------------------------
# Summary stats
# -------------------------------
st.subheader("📌 Summary Statistics")

stats_df = pd.DataFrame({
    "Metric": [
        "Mean Daily Return (%)",
        "Std Deviation (%)",
        "Max Daily Gain (%)",
        "Max Daily Loss (%)"
    ],
    "Value": [
        round(returns.mean(), 3),
        round(returns.std(), 3),
        round(returns.max(), 3),
        round(returns.min(), 3)
    ]
})

st.table(stats_df)

# -------------------------------
# Save today's data
# -------------------------------
st.subheader("💾 Save Today's Data")

csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{ticker}_{start_date}_to_{end_date}.csv",
    mime="text/csv"
)
