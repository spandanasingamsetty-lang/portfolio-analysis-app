import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="🌸 PortfolioAnalysis – Visista, Spandana", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #F3F3F3;}
    h1, h2, h3 {color: #6A0DAD;}
    .stButton>button {background-color: #B57EDC; color:white;}
    .stDownloadButton>button {background-color: #B57EDC; color:white;}
    </style>
""", unsafe_allow_html=True)

st.markdown("## 🌸 PortfolioAnalysis – Visista, Spandana")
st.markdown("Dynamic portfolio dashboard with optimum weights, metrics, charts, and investment allocation.")

# -------------------------
# 1️⃣ CSV Upload
# -------------------------
st.markdown("### 📂 Upload CSV")
uploaded_file = st.file_uploader("Upload CSV with StockSymbol,Weight", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.markdown("**📊 Uploaded Stocks**")
    st.dataframe(df.style.set_properties(**{'background-color': '#FFFFFF', 'color': '#6A0DAD'}))
    
    symbols = df['StockSymbol'].tolist()
    
    # -------------------------
    # 2️⃣ User Inputs
    # -------------------------
    st.markdown("### 🏦 User Inputs")
    risk_free_rate = st.number_input("Enter Risk-free rate (annual, e.g., 0.05 for 5%)", min_value=0.0, max_value=1.0, value=0.05)
    total_investment = st.number_input("Enter Total Investment Amount (₹)", min_value=0.0, value=100000.0)
    
    # -------------------------
    # 3️⃣ Fetch Historical Data
    # -------------------------
    st.info("🔄 Fetching 6-month historical data from Yahoo Finance...")
    prices = pd.DataFrame()
    for sym in symbols:
        data = yf.download(sym+".NS", period="6mo", interval="1d")['Adj Close']
        prices[sym] = data
    st.success("✅ Stock data fetched successfully!")
    
    benchmark = yf.download("^NSEI", period="6mo", interval="1d")['Adj Close']
    
    # -------------------------
    # 4️⃣ Portfolio Optimization
    # -------------------------
    st.info("⚡ Calculating optimum portfolio weights...")
    
    returns = prices.pct_change().dropna()
    benchmark_returns = benchmark.pct_change().dropna()
    
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    num_stocks = len(symbols)
    
    # Monte Carlo simulation
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
    
    # Beta
    covariance_with_benchmark = np.cov(portfolio_daily_returns, benchmark_returns.loc[portfolio_daily_returns.index])[0][1]
    benchmark_var = np.var(benchmark_returns.loc[portfolio_daily_returns.index])
    beta = covariance_with_benchmark / benchmark_var
    
    # Treynor
    treynor = (portfolio_return - risk_free_rate) / beta if beta != 0 else np.nan
    
    # Jensen's Alpha
    benchmark_return_annual = benchmark_returns.mean() * 252
    jensen_alpha = portfolio_return - (risk_free_rate + beta * (benchmark_return_annual - risk_free_rate))
    
    # M2
    sharpe_portfolio = (portfolio_return - risk_free_rate)/portfolio_volatility
    benchmark_vol = np.std(benchmark_returns) * np.sqrt(252)
    m2 = sharpe_portfolio * benchmark_vol + risk_free_rate
    
    # Monthly returns
    portfolio_daily_returns.index = pd.to_datetime(portfolio_daily_returns.index)
    monthly_returns = portfolio_daily_returns.resample('M').apply(lambda x: (1+x).prod()-1)
    
    # -------------------------
    # 5️⃣ Display Results
    # -------------------------
    st.markdown("### 💹 Optimum Portfolio Weights")
    weights_df = pd.DataFrame({
        "Stock": symbols,
        "Optimum Weight": np.round(optimum_weights, 4),
        "Invested Amount (₹)": np.round(optimum_weights * total_investment, 2)
    })
    st.bar_chart(weights_df.set_index("Stock")["Optimum Weight"])
    st.table(weights_df.style.set_properties(**{'background-color': '#FFFFFF', 'color': '#6A0DAD'}))
    
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
    # 6️⃣ Download Report
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
