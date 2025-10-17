import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import yfinance as yf

# Function to fetch the list of all NSE-listed companies
def fetch_nse_tickers():
    url = "https://www.nseindia.com/api/equity-stockIndices"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "en-US,en;q=0.9"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers, timeout=5)  # Get cookies
    response = session.get(url, headers=headers, timeout=5)
    data = response.json()
    tickers = [item['symbol'] for item in data['data']]
    return tickers

# Fetch the list of tickers
tickers = fetch_nse_tickers()

# Streamlit UI
st.title("NSE Portfolio Optimizer")
st.markdown("Select companies to analyze:")

# Multi-select dropdown for company selection
selected_tickers = st.multiselect(
    "Choose companies:",
    options=tickers,
    default=["RELIANCE", "TCS", "HDFCBANK"]
)

if not selected_tickers:
    st.warning("Please select at least one company.")
else:
    st.write(f"Selected companies: {', '.join(selected_tickers)}")

    # Fetch historical data for the selected companies
    data = yf.download(selected_tickers, start="2020-01-01", end=datetime.datetime.today().strftime('%Y-%m-%d'))['Adj Close']

    # Calculate daily returns
    daily_returns = data.pct_change().dropna()

    # Calculate mean returns and covariance matrix
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()

    # Number of portfolios to simulate
    num_portfolios = 50000
    results = np.zeros((3, num_portfolios))
    weights_record = []

    # Simulate portfolios
    for i in range(num_portfolios):
        weights = np.random.random(len(selected_tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_return = np.sum(weights * mean_returns) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        portfolio_sharpe = portfolio_return / portfolio_volatility
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = portfolio_sharpe

    # Convert results to DataFrame
    portfolio_df = pd.DataFrame(results.T, columns=['Returns', 'Volatility', 'Sharpe Ratio'])
    portfolio_df['Weights'] = weights_record

    # Locate the portfolio with the highest Sharpe ratio
    max_sharpe_idx = portfolio_df['Sharpe Ratio'].idxmax()
    optimal_weights = portfolio_df.iloc[max_sharpe_idx]['Weights']

    # Display optimal weights
    st.subheader("Optimal Portfolio Weights")
    optimal_weights_df = pd.DataFrame(optimal_weights, index=selected_tickers, columns=['Weight'])
    st.write(optimal_weights_df)

    # Display portfolio performance metrics
    st.subheader("Portfolio Performance Metrics")
    st.write(f"Expected Annual Return: {portfolio_df['Returns'].iloc[max_sharpe_idx]:.2%}")
    st.write(f"Annual Volatility: {portfolio_df['Volatility'].iloc[max_sharpe_idx]:.2%}")
    st.write(f"Sharpe Ratio: {portfolio_df['Sharpe Ratio'].iloc[max_sharpe_idx]:.2f}")

    # Plot efficient frontier
    st.subheader("Efficient Frontier")
    st.line_chart(portfolio_df[['Returns', 'Volatility']])

    # Download portfolio report
    csv = portfolio_df.to_csv().encode()
    st.download_button(
        label="Download Portfolio Report",
        data=csv,
        file_name='portfolio_report.csv',
        mime='text/csv'
    )
