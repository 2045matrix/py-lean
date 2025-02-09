import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

st.set_page_config(page_title="Price Monitor", page_icon="📊")

st.title("Price Monitor")
st.markdown("Real-time price monitoring for stocks and cryptocurrencies")

# Initialize session state
if 'show_percentage' not in st.session_state:
    st.session_state.show_percentage = False

def toggle_percentage_view():
    st.session_state.show_percentage = not st.session_state.show_percentage

# Calculate date range (last 10 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=10)

# Define tickers
tickers_list = ["TSLA", "NVDA", "ETH-USD", "AVAX-USD"]

# Fetch data
@st.cache_data(ttl=300)  # Cache data for 5 minutes
def get_ticker_data():
    tickers = yf.Tickers(" ".join(tickers_list))
    data = tickers.history(start=start_date, end=end_date)
    # Convert MultiIndex columns to single index for easier access
    data.columns = [f"{col[1]}_{col[0]}" for col in data.columns]
    return data

# Add refresh button in sidebar
if st.sidebar.button("🔄 Refresh Data", type="primary"):
    st.cache_data.clear()
    st.rerun()

# Toggle between price and percentage view
st.sidebar.button("🔀 Toggle % Change View", on_click=toggle_percentage_view)

data = get_ticker_data()

# Create tabs for different views
tab1, tab2 = st.tabs(["Current Prices", "Historical Data"])

with tab1:
    # Display current prices in a clean grid
    st.subheader("Latest Prices")
    col1, col2 = st.columns(2)
    
    latest_data = data.iloc[-1]
    
    for i, ticker in enumerate(tickers_list):
        with col1 if i < len(tickers_list)/2 else col2:
            price = latest_data[f"{ticker}_Close"]
            prev_price = data[f"{ticker}_Close"].iloc[-2]
            change = ((price - prev_price) / prev_price) * 100
            
            if st.session_state.show_percentage:
                value = f"{change:.2f}%"
                delta = f"${price:.2f}"
            else:
                value = f"${price:.2f}"
                delta = f"{change:.2f}%"
            
            st.metric(
                label=ticker,
                value=value,
                delta=delta
            )

with tab2:
    st.subheader("Historical Price Data")
    # Create a selection for the ticker
    selected_ticker = st.selectbox("Select Asset", tickers_list)
    
    # Display historical data
    hist_data = data[f"{selected_ticker}_Close"]
    st.line_chart(hist_data)
    
    # Add download button
    if st.button("📥 Download Historical Data"):
        # Create a DataFrame with datetime index
        download_df = pd.DataFrame({
            'Date': data.index,
            'Close': data[f"{selected_ticker}_Close"],
            'Open': data[f"{selected_ticker}_Open"],
            'High': data[f"{selected_ticker}_High"],
            'Low': data[f"{selected_ticker}_Low"],
            'Volume': data[f"{selected_ticker}_Volume"]
        }).set_index('Date')
        
        csv = download_df.to_csv()
        st.download_button(
            label="Click to Download CSV",
            data=csv,
            file_name=f"{selected_ticker}_historical_data.csv",
            mime="text/csv",
        )
