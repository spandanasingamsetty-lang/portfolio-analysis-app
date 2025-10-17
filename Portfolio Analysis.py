import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import datetime
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="PortfolioAnalysis – Spandana, Visista", layout="wide")
st.title("PortfolioAnalysis – Spandana, Visista")
st.markdown("Dynamic portfolio dashboard with optimum weights, metrics, charts, and investment allocation.")

# -------------------------
# NIFTY 50 Companies
# -------------------------
nifty50_companies = {
    "HDFCBANK": "HDFC Bank",
    "TCS": "Tata Consultancy Services",
    "RELIANCE": "Reliance Industries",
    "HINDUNILVR": "Hindustan Unilever",
    "INFY": "Infosys",
    "ICICIBANK": "ICICI Bank",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "SBIN": "State Bank of India",
    "LT": "Larsen & Toubro",
    "AXISBANK": "Axis Bank",
}

selected = st.multiselect(
    "Select Companies (up to 10):",
    options=list(nifty50_companies.keys()),
    format_func=lambda x: f"{x} - {nifty50_companies[x]}",
    default=["HDFCBANK","TCS","RELIANCE"]
)

if not selected:
    st.warning("Select at least one company to proceed.")
    st.stop()
elif len(selected) > 10:
    st.warning("You can select up to 10 companies only.")
    st.stop()

# -------------------------
# User Inputs
# -------------------------
risk_free_rate = st.number_input("Enter Risk-free rate (annual, e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, value=0.05)
total_investment = st.number_input("Enter Total Investment Amount (₹)", min_value=0.0, value=100000.0)

# -------------------------
# Fetch NSE stock data
# -------------------------
st.info("Fetching latest NSE stock data...")

def fetch_nse_stock(symbol):
    url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }
    session = requests.Session()
    try:
        # get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        response = session.get(url, headers=headers, timeout=5)
        data = response.json()
        last_price = float(data['priceInfo']['lastPrice'])
        day_high = float(data['priceInfo']['intraDayHighLow']['max'])
        day_low = float(data['priceInfo']['intraDayHighLow']['min'])
        return {
            "Symbol": symbol,
            "Name": nifty50_companies[symbol],
            "Last Price": last_price,
            "Day High": day_high,
            "Day Low": day_low
        }
    except:
        return None

results = []
for sym in selected:
    res = fetch_nse_stock(sym)
    if res:
        results.append(res)
    else:
        st.warning(f"⚠️ Could not fetch {sym}")

if not results:
    st.error("❌ No valid symbols to process. NSE may block requests.")
    st.stop()

prices_df = pd.DataFrame(results)
st.success("✅ Live prices fetched successfully!")
st.table(prices_df)

# -------------------------
# Generate sample historical returns (since NSE API doesn’t provide 6mo historical directly)
# In production, replace with proper API (Alpha Vantage / Twelve Data) for historical data
# -------------------------
np.random.seed(42)
num_days = 126  # approx 6 months trading days
historical_returns = pd.DataFrame(
    np.random.normal(0.001, 0.02, size=(num_days, len(results))),
    columns=[res["Symbol"] for res in results]
)

# -------------------------
# Portfolio Optimization
# -------------------------
st.info("Calculating optimum portfolio weights...")

returns = historical_returns
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_stocks = len(results)

num_portfolios = 50000
results_array = np.zeros((3, num_portfolios))
weight_array = []

for i in range(num_portfolios):
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)
    weight_array.append(weights)
    port_return = np.sum(weights * mean_returns) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
    sharpe_ratio = (port_return - risk_free_rate)/port_std
    results_array[0,i] = port_return
    results_array[1,i] = port_std
    results_array[2,i] = sharpe_ratio

max_idx = np.argmax(results_array[2])
optimum_weights = weight_array[max_idx]

# Portfolio metrics
portfolio_daily_returns = returns @ optimum_weights
portfolio_return = np.sum(optimum_weights * mean_returns) * 252
portfolio_volatility = np.sqrt(np.dot(optimum_weights.T, np.dot(cov_matrix*252, optimum_weights)))
sharpe_portfolio = (portfolio_return - risk_free_rate)/portfolio_volatility
m2 = sharpe_portfolio * portfolio_volatility + risk_free_rate

# -------------------------
# Display Results
# -------------------------
st.markdown("### 💹 Optimum Portfolio Weights")
weights_df = pd.DataFrame({
    "Stock": [res["Symbol"] for res in results],
    "Optimum Weight": np.round(optimum_weights,4),
    "Invested Amount (₹)": np.round(optimum_weights * total_investment,2)
})
st.bar_chart(weights_df.set_index("Stock")["Optimum Weight"])
st.table(weights_df.style.set_properties(**{'background-color':'#FFFFFF','color':'#7A1FA2'}))

st.markdown("### 🧮 Portfolio Metrics")
metrics = {
    "📈 Expected Return": round(portfolio_return,4),
    "📉 Volatility (σ)": round(portfolio_volatility,4),
    "⭐ Sharpe Ratio": round(sharpe_portfolio,4),
    "💎 M²": round(m2,4)
}
st.table(pd.DataFrame(metrics, index=["Value"]).T)

st.markdown("### 📅 Simulated Monthly Portfolio Returns")
monthly_returns = portfolio_daily_returns.resample('M').apply(lambda x: (1+x).prod()-1)
st.line_chart(monthly_returns)

st.markdown("### 📈 Cumulative Portfolio Growth")
cumulative_returns = (1 + portfolio_daily_returns).cumprod()
st.line_chart(cumulative_returns)

# -------------------------
# Download Report
# -------------------------
report = weights_df.copy()
report["Expected Return"] = round(portfolio_return,4)
report["Volatility"] = round(portfolio_volatility,4)
report["Sharpe Ratio"] = round(sharpe_portfolio,4)
csv = report.to_csv(index=False).encode()
st.download_button(
    label="📥 Download Portfolio Report as CSV",
    data=csv,
    file_name='portfolio_report.csv',
    mime='text/csv'
)
