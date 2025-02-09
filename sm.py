import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Stock Market Analysis", layout="wide")
st.title("Stock Market Analysis Dashboard")

# Sidebar inputs
st.sidebar.header("Settings")
days = st.sidebar.slider("Number of days", 5, 30, 10)
symbols = st.sidebar.multiselect(
    "Select stocks",
    ["TSLA", "NVDA", "AVAX-USD", "^GSPC"],
    default=["TSLA", "NVDA", "^GSPC"]
)

if symbols:
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create a DataFrame to store all percentage changes
    all_data = pd.DataFrame()
    
    # Fetch data for each symbol
    for symbol in symbols:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=start_date, end=end_date)
        if not hist.empty:
            pct_change = ((hist['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
            all_data[symbol] = pct_change
    
    # Display price changes
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Price Performance")
        # Create price performance plot
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        for column in all_data.columns:
            sns.lineplot(data=all_data, x=all_data.index, y=column, label=column, marker='o')
        
        plt.title(f'{days}-Day Price Performance (% Change)', fontsize=12)
        plt.xlabel('Date')
        plt.ylabel('Price Change (%)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Performance Summary")
        final_changes = all_data.iloc[-1].sort_values(ascending=False)
        
        # Create a summary DataFrame
        summary_data = []
        for symbol in final_changes.index:
            data = all_data[symbol].dropna()
            if len(data) > 0:
                # Calculate statistics
                total_change = final_changes[symbol]
                daily_std = data.std()
                max_gain = data.max()
                max_loss = data.min()
                
                summary_data.append({
                    'Symbol': symbol,
                    'Total Change (%)': f"{total_change:.1f}",
                    'Volatility (σ)': f"{daily_std:.1f}",
                    'Max Gain (%)': f"{max_gain:.1f}",
                    'Max Loss (%)': f"{max_loss:.1f}"
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, hide_index=True)
    
    # Beta Analysis
    if '^GSPC' in symbols and len(symbols) > 1:
        st.subheader("Beta Analysis (vs S&P 500)")
        
        # Create beta analysis plots
        non_sp500 = [s for s in symbols if s != '^GSPC']
        cols = st.columns(len(non_sp500))
        
        for i, symbol in enumerate(non_sp500):
            with cols[i]:
                # Get common dates where both asset and SP500 have data
                mask = all_data[[symbol, '^GSPC']].notna().all(axis=1)
                asset_data = all_data[symbol][mask]
                sp500_data = all_data['^GSPC'][mask]
                
                if len(asset_data) > 0:
                    # Prepare data for regression
                    X = sp500_data.values.reshape(-1, 1)
                    y = asset_data.values.reshape(-1, 1)
                    
                    # Fit regression line
                    reg = LinearRegression().fit(X, y)
                    y_pred = reg.predict(X)
                    
                    # Calculate R-squared and beta
                    r2 = r2_score(y, y_pred)
                    beta = reg.coef_[0][0]
                    
                    # Create scatter plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plt.scatter(sp500_data, asset_data, alpha=0.5)
                    plt.plot(sp500_data, y_pred, '--r', alpha=0.8)
                    
                    plt.title(f'{symbol} vs S&P 500\nβ = {beta:.2f}, R² = {r2:.2f}')
                    plt.xlabel('S&P 500 Return (%)')
                    plt.ylabel(f'{symbol} Return (%)')
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    
                    # Display beta interpretation
                    if abs(beta) > 1.5:
                        volatility = "much more volatile than"
                    elif abs(beta) > 1.1:
                        volatility = "more volatile than"
                    elif abs(beta) < 0.9:
                        volatility = "less volatile than"
                    else:
                        volatility = "similarly volatile to"
                    
                    st.write(f"**Beta Interpretation:**")
                    st.write(f"- {symbol} is {volatility} the market")
                    st.write(f"- {r2*100:.1f}% of price movements can be explained by market movements")