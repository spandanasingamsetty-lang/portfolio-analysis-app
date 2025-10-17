import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(page_title="PortfolioAnalysis – Spandana, Visista", layout="wide")

st.title("PortfolioAnalysis – Spandana, Visista")
st.markdown("Dynamic portfolio dashboard with optimum weights, metrics, charts, and investment allocation.")

# -------------------------
# NIFTY 50 Companies
# -------------------------
# key: Yahoo ticker without .NS, value: company name
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

st.markdown("### 📌 Select Companies (up to 10)")
selected = st.multiselect(
    "Choose NIFTY 50 stocks:",
    options=list(nifty50_companies.keys()),
    format_func=lambda x: f"{x} - {nifty50_companies[x]}",
    default=["HDFCBANK","TCS","RELIANCE"]
)

if not selected:
    st.warning("Please select at least one company to proceed.")
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
# Fetch stock data
# -------------------------
st.info("🔄 Fetching 6-month historical prices...")

prices = pd.DataFrame()
valid_symbols = []
invalid_symbols = []

for sym in selected:
    yahoo_sym = sym + ".NS"
    try:
        data = yf.download(yahoo_sym, period="6mo", interval="1d", auto_adjust=True)['Adj Close']
        if data.empty:
            invalid_symbols.append(sym)
        else:
            prices[sym] = data
            valid_symbols.append(sym)
    except:
        invalid_symbols.append(sym)

if invalid_symbols:
    st.warning(f"⚠️ These symbols could not be fetched and will be ignored: {invalid_symbols}")
if not valid_symbols:
    st.error("❌ No valid symbols to process. Select different companies.")
    st.stop()

st.success(f"✅ Successfully fetched data for: {valid_symbols}")

# -------------------------
# Fetch Benchmark
# -------------------------
try:
    benchmark = yf.download("^NSEI", period="6mo", interval="1d", auto_adjust=True)['Adj Close']
except Exception as e:
    st.error(f"❌ Failed to fetch benchmark: {e}")
    st.stop()

# -------------------------
# Portfolio Optimization
# -------------------------
st.info("⚡ Calculating optimum portfolio weights...")

returns = prices.pct_change().dropna()
benchmark_returns = benchmark.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
num_stocks = len(valid_symbols)

num_portfolios = 50000
results = np.zeros((3, num_portfolios))
weight_array = []

for i in range(num_portfolios):
    weights = np.random.random(num_stocks)
    weights /= np.sum(weights)
    weight_array.append(weights)
    port_return = np.sum(weights * mean_returns) * 252
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix*252, weights)))
    sharpe_ratio = (port_return - risk_free_rate)/port_std
    results[0,i] = port_return
    results[1,i] = port_std
    results[2,i] = sharpe_ratio

max_idx = np.argmax(results[2])
optimum_weights = weight_array[max_idx]

# Portfolio metrics
portfolio_daily_returns = returns @ optimum_weights
portfolio_return = np.sum(optimum_weights * mean_returns) * 252
portfolio_volatility = np.sqrt(np.dot(optimum_weights.T, np.dot(cov_matrix*252, optimum_weights)))
cov_with_bench = np.cov(portfolio_daily_returns, benchmark_returns.loc[portfolio_daily_returns.index])[0][1]
benchmark_var = np.var(benchmark_returns.loc[portfolio_daily_returns.index])
beta = cov_with_bench / benchmark_var
treynor = (portfolio_return - risk_free_rate) / beta if beta != 0 else np.nan
benchmark_return_annual = benchmark_returns.mean() * 252
jensen_alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return_annual - risk_free_rate))
sharpe_portfolio = (portfolio_return - risk_free_rate)/portfolio_volatility
benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)
m2 = sharpe_portfolio * benchmark_vol + risk_free_rate

portfolio_daily_returns.index = pd.to_datetime(portfolio_daily_returns.index)
monthly_returns = portfolio_daily_returns.resample('M').apply(lambda x: (1+x).prod()-1)

# -------------------------
# Display results
# -------------------------
st.markdown("### 💹 Optimum Portfolio Weights")
weights_df = pd.DataFrame({
    "Stock": valid_symbols,
    "Optimum Weight": np.round(optimum_weights,4),
    "Invested Amount (₹)": np.round(optimum_weights * total_investment,2)
})
st.bar_chart(weights_df.set_index("Stock")["Optimum Weight"])
st.table(weights_df.style.set_properties(**{'background-color':'#FFFFFF','color':'#7A1FA2'}))

st.markdown("### 🧮 Portfolio Metrics")
metrics = {
    "📈 Expected Return": round(portfolio_return,4),
    "📉 Volatility (σ)": round(portfolio_volatility,4),
    "⭐ Sharpe Ratio": round(results[2,max_idx],4),
    "🧾 Treynor Ratio": round(treynor,4),
    "🧠 Jensen's Alpha": round(jensen_alpha,4),
    "🔗 Beta": round(beta,4),
    "💎 M²": round(m2,4)
}
st.table(pd.DataFrame(metrics, index=["Value"]).T)

st.markdown("### 📅 Monthly Portfolio Returns")
st.line_chart(monthly_returns)

st.markdown("### 📈 Cumulative Portfolio vs NIFTY 50 Benchmark")
cumulative_df = pd.DataFrame({
    "Portfolio": (1+portfolio_daily_returns).cumprod(),
    "NIFTY50": (1+benchmark_returns.loc[portfolio_daily_returns.index]).cumprod()
})
st.line_chart(cumulative_df)

# -------------------------
# Download Report
# -------------------------
report = weights_df.copy()
report["Expected Return"] = round(portfolio_return,4)
report["Volatility"] = round(portfolio_volatility,4)
report["Sharpe Ratio"] = round(results[2,max_idx],4)
report["Treynor Ratio"] = round(treynor,4)
report["Jensen's Alpha"] = round(jensen_alpha,4)
report["Beta"] = round(beta,4)
report["M²"] = round(m2,4)

csv = report.to_csv(index=False).encode()
st.download_button(
    label="📥 Download Portfolio Report as CSV",
    data=csv,
    file_name='portfolio_report.csv',
    mime='text/csv'
)
