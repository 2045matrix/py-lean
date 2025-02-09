import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Performance Analysis", page_icon="📈")

st.title("Performance Analysis")
st.markdown("Compare performance across different assets")

# Date range selector
col1, col2 = st.columns(2)
with col1:
    days = st.slider("Number of days", 5, 30, 10)
with col2:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

# Symbols to analyze
symbols = ['TSLA', 'NVDA', 'AVAX-USD', '^GSPC']
selected_symbols = st.multiselect(
    "Select assets to compare",
    symbols,
    default=['TSLA', 'NVDA']
)

if selected_symbols:
    # Create DataFrame to store all percentage changes
    @st.cache_data(ttl=300)
    def get_performance_data(symbols, start_date, end_date):
        all_data = pd.DataFrame()
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start_date, end=end_date)
            if not hist.empty:
                pct_change = ((hist['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                all_data[symbol] = pct_change
        return all_data

    performance_data = get_performance_data(selected_symbols, start_date, end_date)
    
    # Create interactive plot using plotly
    fig = px.line(
        performance_data,
        title=f"Relative Performance (Past {days} Days)",
        labels={"value": "% Change", "variable": "Asset"}
    )
    fig.update_layout(
        showlegend=True,
        hovermode='x unified',
        plot_bgcolor='white',
        yaxis_title="Percentage Change (%)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Display summary statistics
    st.subheader("Performance Summary")
    summary = pd.DataFrame({
        'Total Change (%)': performance_data.iloc[-1],
        'Max Change (%)': performance_data.max(),
        'Min Change (%)': performance_data.min(),
        'Volatility (%)': performance_data.std()
    }).round(2)
    
    st.dataframe(summary)
