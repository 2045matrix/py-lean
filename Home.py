import streamlit as st

st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Financial Market Analysis Dashboard")
st.sidebar.success("Select a page above.")

st.markdown("""
Welcome to the Financial Market Analysis Dashboard! This application provides various tools and visualizations for analyzing financial market data.

### Features:
- **Price Monitor**: Real-time monitoring of stock and crypto prices
- **Performance Analysis**: Compare performance across different assets
- **Trend Analysis**: View trend lines and regression analysis
- **Historical Data**: Access historical price data for multiple assets

Choose a page from the sidebar to get started!
""")
