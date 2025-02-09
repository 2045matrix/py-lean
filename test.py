import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Define color codes for terminal output
COLORS = {
    'TSLA': '\033[92m',  # Green
    'NVDA': '\033[94m',  # Blue
    'ETH-USD': '\033[95m',  # Magenta
    'AVAX-USD': '\033[95m',  # Magenta
    'RESET': '\033[0m'
}

# Calculate date range (last 10 days)
end_date = datetime.now()
start_date = end_date - timedelta(days=10)

# Fetch historical data for all tickers
tickers = yf.Tickers("TSLA NVDA ETH-USD AVAX-USD")
data = tickers.history(start=start_date, end=end_date)

# Display header with colored ticker names
print(f"\n{COLORS['TSLA']}TSLA{COLORS['RESET']} | {COLORS['NVDA']}NVDA{COLORS['RESET']} | {COLORS['ETH-USD']}ETH/USD{COLORS['RESET']} | {COLORS['AVAX-USD']}AVAX/USD{COLORS['RESET']}")
print("----------------------------------------")

# Display closing prices for the last 10 trading days
for date in data.index[-10:]:  # Ensures we only show up to 10 entries
    tsla_close = data.loc[date, ('Close', 'TSLA')]
    nvda_close = data.loc[date, ('Close', 'NVDA')]
    eth_close = data.loc[date, ('Close', 'ETH-USD')]
    avax_close = data.loc[date, ('Close', 'AVAX-USD')]
    
    print(f"{date.strftime('%Y-%m-%d')}: "
          f"{COLORS['TSLA']}{tsla_close:.2f}{COLORS['RESET']} | "
          f"{COLORS['NVDA']}{nvda_close:.2f}{COLORS['RESET']} | "
          f"{COLORS['ETH-USD']}{eth_close:.2f}{COLORS['RESET']} | "
          f"{COLORS['AVAX-USD']}{avax_close:.2f}{COLORS['RESET']}")

# Symbols to fetch
symbols = ['TSLA', 'NVDA', 'AVAX-USD', '^GSPC']

# Fetch data for each symbol
for symbol in symbols:
    print(f"\n{symbol} Historical Data:")
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date)
    print(hist[['Close']].round(2))

# Set the Seaborn style and color palette
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Create a DataFrame to store all percentage changes
all_data = pd.DataFrame()

# Fetch data for each symbol
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date)
    
    # Calculate percentage change from first day
    pct_change = ((hist['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
    all_data[symbol] = pct_change

# Create the plot
plt.figure()
sns.set_context("talk")  # Larger context for better readability

# Plot each line with Seaborn and add regression lines
for column in all_data.columns:
    data = all_data[column].dropna()
    if len(data) > 0:
        # Plot actual data
        sns.lineplot(data=all_data, x=all_data.index, y=column, label=column, marker='o', markersize=8)
        
        # Prepare data for regression
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values.reshape(-1, 1)
        
        # Fit regression line
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)
        
        # Calculate R-squared
        r2 = r2_score(y, y_pred)
        
        # Plot regression line
        dates = data.index
        plt.plot(dates, y_pred, '--', alpha=0.8, 
                label=f'{column} trend (R² = {r2:.2f})')

# Customize the plot
plt.title('10-Day Price Performance with Trend Lines', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price Change (%)', fontsize=12)

# Add horizontal line at 0%
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Format y-axis with percentage
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}%'.format(y)))

# Format x-axis dates
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Add a legend with a better position
plt.legend(title='Assets & Trends', title_fontsize=12, fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot with higher DPI for better quality
plt.savefig('price_comparison_regression.png', dpi=300, bbox_inches='tight')
plt.close()

# Print the final percentage changes and trend analysis
print("\nPerformance Analysis:")
final_changes = all_data.iloc[-1].sort_values(ascending=False)
for symbol in final_changes.index:
    data = all_data[symbol].dropna()
    if len(data) > 0:
        # Calculate daily change rate from regression
        X = np.arange(len(data)).reshape(-1, 1)
        y = data.values.reshape(-1, 1)
        reg = LinearRegression().fit(X, y)
        daily_change = reg.coef_[0][0]
        r2 = r2_score(y, reg.predict(X))
        
        print(f"\n{symbol}:")
        print(f"Total Change: {final_changes[symbol]:.1f}%")
        print(f"Average Daily Change: {daily_change:.2f}%")
        print(f"Trend Reliability (R²): {r2:.2f}")

# Set the Seaborn style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Calculate date range
end_date = datetime.now()
start_date = end_date - timedelta(days=10)

# Symbols to fetch
symbols = ['TSLA', 'NVDA', 'AVAX-USD', '^GSPC']

# Create a DataFrame to store all percentage changes
all_data = pd.DataFrame()

# Fetch data for each symbol
for symbol in symbols:
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date, end=end_date)
    
    # Calculate percentage change from first day
    pct_change = ((hist['Close'] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
    all_data[symbol] = pct_change

# Create subplots for each asset vs S&P500
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Asset Performance vs S&P 500 (Beta Analysis)', fontsize=16, y=1.05)

# Colors for each asset
colors = {'TSLA': 'blue', 'NVDA': 'green', 'AVAX-USD': 'purple'}

# Analyze each asset against S&P500
results = []
for i, symbol in enumerate([s for s in symbols if s != '^GSPC']):
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
        
        # Plot scatter and regression line
        ax = axes[i]
        ax.scatter(sp500_data, asset_data, color=colors[symbol], alpha=0.6)
        ax.plot(sp500_data, y_pred, '--', color='red', alpha=0.8)
        
        # Add labels and title
        ax.set_xlabel('S&P 500 Return (%)')
        ax.set_ylabel(f'{symbol} Return (%)')
        ax.set_title(f'{symbol} vs S&P 500\nβ = {beta:.2f}, R² = {r2:.2f}')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Store results
        results.append({
            'Symbol': symbol,
            'Beta': beta,
            'R_squared': r2,
            'Correlation': np.corrcoef(sp500_data, asset_data)[0,1]
        })

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('beta_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Print analysis results
print("\nMarket (S&P 500) Correlation Analysis:")
print("-" * 50)
for result in results:
    print(f"\n{result['Symbol']} vs S&P 500:")
    print(f"Beta (β): {result['Beta']:.2f}")
    print(f"R-squared (R²): {result['R_squared']:.2f}")
    print(f"Correlation: {result['Correlation']:.2f}")
    print(f"Interpretation:")
    
    # Beta interpretation
    if abs(result['Beta']) > 1.5:
        volatility = "much more volatile than"
    elif abs(result['Beta']) > 1.1:
        volatility = "more volatile than"
    elif abs(result['Beta']) < 0.9:
        volatility = "less volatile than"
    else:
        volatility = "similarly volatile to"
    
    # Direction interpretation
    if result['Beta'] > 0:
        direction = "same"
    else:
        direction = "opposite"
    
    print(f"- Moves in the {direction} direction and is {volatility} the market")
    print(f"- {result['R_squared']*100:.1f}% of price movements can be explained by market movements")