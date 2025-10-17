import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
from io import StringIO

st.set_page_config(page_title="Portfolio Analysis", layout="wide")
st.title("Portfolio Analysis")

# ------------------------
# Alpha Vantage API Key
# ------------------------
ALPHA_KEY = "YOUR_ALPHA_VANTAGE_KEY"  # Replace with your key

# ------------------------
# Load local NSE tickers CSV
# ------------------------
try:
    nse_df = pd.read_csv("nse_tickers.csv")
except FileNotFoundError:
    st.error("Local NSE tickers CSV not found. Upload nse_tickers.csv in the app folder.")
    st.stop()

nse_df = nse_df[['SYMBOL', 'NAME']].dropna()

# ------------------------
# Company Selection
# ------------------------
selected = st.multiselect(
    "Select Companies (up to 10):",
    options=nse_df['SYMBOL'].tolist(),
    format_func=lambda x: f"{x} - {nse_df[nse_df['SYMBOL']==x]['NAME'].values[0]}",
    default=["HDFCBANK", "RELIANCE"]
)

if not selected:
    st.warning("Please select at least one company to proceed.")
    st.stop()

# ------------------------
# Date Selection
# ------------------------
today = datetime.date.today()
start_date = st.date_input("Start Date", today - datetime.timedelta(days=180))
end_date = st.date_input("End Date", today)
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ------------------------
# Investment Inputs
# ------------------------
risk_free_rate = st.number_input("Enter risk-free rate (annual, e.g., 0.05 for 5%)", 0.0, 1.0, 0.05)
total_investment = st.number_input("Enter total investment amount (₹)", 0.0, 10000000.0, 100000.0)

# ------------------------
# Fetch Historical Data via Alpha Vantage
# ------------------------
st.info("Fetching historical data from Alpha Vantage...")

prices = pd.DataFrame()
failed_symbols = []

for sym in selected:
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={sym}.BSE&outputsize=full&apikey={ALPHA_KEY}"
        r = requests.get(url)
        data = r.json()
        ts = data['Time Series (Daily)']
        df = pd.DataFrame(ts).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        prices[sym] = df['5. adjusted close'].astype(float)
    except:
        failed_symbols.append(sym)

if failed_symbols:
    st.warning(f"⚠️ These symbols could not be fetched and will be ignored: {failed_symbols}")

if prices.empty:
    st.error("No valid symbols to process. Select different companies or date range.")
    st.stop()

# ------------------------
# Calculate Returns
# ------------------------
returns = prices.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_stocks = len(returns.columns)

# ------------------------
# Portfolio Optimization (Max Sharpe)
# ------------------------
st.info("Calculating optimum portfolio weights...")
num_portfolios = 50000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)
    weights_record.append(weights)
    port_return = np.sum(weights * mean_returns) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - risk_free_rate) / port_std
    results[0, i] = port_return
    results[1, i] = port_std
    results[2, i] = sharpe

# ------------------------
# Optimal Portfolio
# ------------------------
max_idx = np.argmax(results[2])
opt_weights = weights_record[max_idx]
portfolio_return = results[0, max_idx]
portfolio_volatility = results[1, max_idx]
sharpe_ratio = results[2, max_idx]

# ------------------------
# Display Results
# ------------------------
st.subheader("💹 Optimal Portfolio Weights")
weights_df = pd.DataFrame({
    "Stock": returns.columns,
    "Weight": np.round(opt_weights, 4),
    "Invested Amount (₹)": np.round(opt_weights * total_investment, 2)
})
st.bar_chart(weights_df.set_index("Stock")["Weight"])
st.table(weights_df)

st.subheader("🧮 Portfolio Metrics")
metrics = {
    "Expected Annual Return": round(portfolio_return, 4),
    "Annual Volatility (σ)": round(portfolio_volatility, 4),
    "Sharpe Ratio": round(sharpe_ratio, 4)
}
st.table(pd.DataFrame(metrics, index=["Value"]).T)

# ------------------------
# Monthly & Cumulative Returns
# ------------------------
st.subheader("📅 Monthly Portfolio Returns")
portfolio_daily_returns = returns @ opt_weights
monthly_returns = portfolio_daily_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
st.line_chart(monthly_returns)

st.subheader("📈 Cumulative Portfolio Growth")
cumulative_returns = (1 + portfolio_daily_returns).cumprod()
st.line_chart(cumulative_returns)

# ------------------------
# Download Report
# ------------------------
report = weights_df.copy()
report["Expected Return"] = round(portfolio_return, 4)
report["Volatility"] = round(portfolio_volatility, 4)
report["Sharpe Ratio"] = round(sharpe_ratio, 4)
csv = report.to_csv(index=False).encode()

st.download_button(
    label="📥 Download Portfolio Report as CSV",
    data=csv,
    file_name='portfolio_report.csv',
    mime='text/csv'
)
