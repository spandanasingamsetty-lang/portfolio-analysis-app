import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime

# ------------------------
# Page Setup
# ------------------------
st.set_page_config(page_title="PortfolioAnalysis – Spandana, Visista", layout="wide")
st.title("Portfolio Analysis")
st.markdown("Select NSE companies, set investment parameters, and calculate optimal portfolio weights.")

# ------------------------
# Load NSE Tickers CSV
# ------------------------
# Download the latest NSE CSV from https://www1.nseindia.com/content/equities/EQUITY_L.csv
# and place it in the app folder
nse_df = pd.read_csv('EQUITY_L.csv')  # SYMBOL, NAME OF COMPANY
nse_df = nse_df[['SYMBOL', 'NAME OF COMPANY']].dropna()

# ------------------------
# Company Selection
# ------------------------
selected = st.multiselect(
    "Select Companies (up to 10):",
    options=nse_df['SYMBOL'].tolist(),
    format_func=lambda x: f"{x} - {nse_df[nse_df['SYMBOL']==x]['NAME OF COMPANY'].values[0]}",
    default=["HDFCBANK", "TCS"]
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
# Fetch Historical Data
# ------------------------
st.info("Fetching historical data from Yahoo Finance...")
symbols_yf = [s + ".NS" for s in selected]
data = yf.download(symbols_yf, start=start_date, end=end_date)

# Handle single vs multi ticker
if isinstance(data.columns, pd.MultiIndex):
    if 'Adj Close' in data.columns.get_level_values(0):
        data = data['Adj Close']
else:
    data = data.to_frame(name=selected[0])

if data.empty:
    st.error("Failed to fetch data. Please check tickers or date range.")
    st.stop()

# ------------------------
# Calculate Daily Returns
# ------------------------
returns = data.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_stocks = len(selected)

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
# Find Optimal Portfolio
# ------------------------
max_idx = np.argmax(results[2])
opt_weights = weights_record[max_idx]

# Portfolio Metrics
portfolio_return = results[0, max_idx]
portfolio_volatility = results[1, max_idx]
sharpe_ratio = results[2, max_idx]

# ------------------------
# Display Results
# ------------------------
st.subheader("💹 Optimal Portfolio Weights")
weights_df = pd.DataFrame({
    "Stock": selected,
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
