import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import os
import xlsxwriter

# Function to analyze stock data
def analyze_stock(ticker):
    # Step 1: Download Stock Data from Yahoo Finance for the Last 5 Years
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)  # Last 5 years
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Step 2: Calculate Percentage Changes
    stock_data['Day_Change'] = stock_data['Close'].pct_change() * 100
    stock_data['Month_Change'] = stock_data['Close'].pct_change(periods=22) * 100  # Approx. 22 trading days in a month
    stock_data['Year_Change'] = stock_data['Close'].pct_change(periods=252) * 100  # Approx. 252 trading days in a year
    stock_data['Week_Change'] = stock_data['Close'].pct_change(periods=5) * 100  # Approx. 5 weeks

    # Add a 'Year' column
    stock_data['Year'] = stock_data.index.year

    # Calculate average changes for each year
    annual_summary = stock_data.groupby('Year').agg({
        'Day_Change': 'mean',
        'Month_Change': 'mean',
        'Week_Change': 'mean',
        'Year_Change': 'mean'
    }).reset_index()

    # Save the annual summary to CSV
    annual_summary.to_csv(f'{ticker}_annual_summary.csv', index=False)
    print(f"Annual summary for {ticker} saved to CSV file.")

# Function to calculate weekly investment return
def calculate_weekly_investment_return(ticker, weekly_investment, duration_years=5):
    # Define the duration
    end_date = datetime.today()
    start_date = end_date - timedelta(days=duration_years * 365)
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1wk')

    total_invested = 0
    total_shares = 0

    for index, row in stock_data.iterrows():
        weekly_price = row['Close']
        shares_bought = weekly_investment / weekly_price
        total_shares += shares_bought
        total_invested += weekly_investment

    current_value = total_shares * stock_data['Close'][-1]
    total_return = current_value - total_invested
    percentage_return = (total_return / total_invested) * 100

    # Create a summary DataFrame
    summary_data = {
        "Metric": ["Total Amount Invested", "Current Value of Investment", "Total Return", "Percentage Return"],
        "Amount": [f"₹{total_invested:.2f}", f"₹{current_value:.2f}", f"₹{total_return:.2f}", f"{percentage_return:.2f}%"]
    }
    investment_summary = pd.DataFrame(summary_data)

    # Save the investment summary to CSV
    investment_summary.to_csv(f'{ticker}_investment_summary.csv', index=False)
    print(f"Investment summary for {ticker} saved to CSV file.")

# Get ticker symbol from user input
ticker = input("Enter the stock ticker symbol: ")
weekly_investment = float(input("Enter the weekly investment amount: "))

# Analyze the stock and calculate the investment return
analyze_stock(ticker)
calculate_weekly_investment_return(ticker, weekly_investment, duration_years=5)

# Function to fetch stock price data and calculate percentage changes
def fetch_stock_price_data(ticker):
    try:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)

        # Weekly data
        weekly_data = data.resample('W').last()
        weekly_closing_prices = weekly_data['Close']

        # Monthly data
        monthly_data = data.resample('M').last()
        monthly_closing_prices = monthly_data['Close']
        monthly_percent_change = monthly_closing_prices.pct_change() * 100

        return weekly_closing_prices, monthly_closing_prices, monthly_percent_change
    except Exception as e:
        print(f"Error fetching data for '{ticker}': {str(e)}")
        return None, None, None

# Function to save stock analysis to CSV
def save_stock_analysis_to_csv(ticker):
    # Fetch stock price data
    weekly_closing_prices, monthly_closing_prices, monthly_percent_change = fetch_stock_price_data(ticker)

    # Create a DataFrame for the weekly and monthly data
    if weekly_closing_prices is not None and monthly_closing_prices is not None:
        stock_data = pd.DataFrame({
            'Date (Weekly)': weekly_closing_prices.index,
            'Weekly Closing Price': weekly_closing_prices.values,
        })

        monthly_data = pd.DataFrame({
            'Date (Monthly)': monthly_closing_prices.index,
            'Monthly Closing Price': monthly_closing_prices.values,
            'Monthly % Change': monthly_percent_change.values
        })

        # Save to CSV
        stock_data.to_csv(f'{ticker}_weekly_closing_prices.csv', index=False)
        monthly_data.to_csv(f'{ticker}_monthly_data.csv', index=False)

        print(f"Stock analysis for {ticker} saved to CSV files.")
    else:
        print(f"Could not fetch data for {ticker}.")

if __name__ == "__main__":
    ticker = ticker

    # Save stock analysis to CSV
    save_stock_analysis_to_csv(ticker)

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Get ticker input from the user
ticker = ticker

# Download stock data for the last 6 months (weekly frequency)
data = yf.download(ticker, period='6mo', interval='1wk')

# Calculate weekly returns
data['Weekly Return'] = data['Adj Close'].pct_change()

# Calculate buying and selling pressure
data['Buying Pressure'] = np.where(data['Weekly Return'] > 0, data['Volume'], 0)
data['Selling Pressure'] = np.where(data['Weekly Return'] < 0, data['Volume'], 0)

# Behavioral analysis: Calculate rolling statistics
data['Rolling Mean'] = data['Weekly Return'].rolling(window=4).mean()  # 4 weeks moving average
data['Rolling Std'] = data['Weekly Return'].rolling(window=4).std()    # 4 weeks moving standard deviation

# Human behavior proxy (volatility)
data['Behavior Indicator'] = np.where(data['Rolling Std'] > data['Rolling Std'].mean(), 'High Volatility', 'Low Volatility')

# Forecasting the next week's price using a simple model (Holt-Winters Exponential Smoothing)
model = ExponentialSmoothing(data['Adj Close'].dropna(), trend="add", seasonal=None)
fit = model.fit()
forecast = fit.forecast(1)  # Forecasting the next week

# Add forecast data to the DataFrame
next_date = data.index[-1] + pd.Timedelta(days=7)  # Assuming weekly data
forecast_df = pd.DataFrame({'Date': [next_date], 'Forecasted Price': forecast})
forecast_df.set_index('Date', inplace=True)

# Concatenate forecast with the original data
data_with_forecast = pd.concat([data, forecast_df])

# Save the DataFrame to a CSV file
csv_filename = f"{ticker}_weekly_analysis.csv"
data_with_forecast.to_csv(csv_filename)

print(f"Data saved to {csv_filename}")
print(f"Forecasted Next Week Price: {forecast.values[0]:.2f}")

import yfinance as yf
import pandas as pd

def fetch_value_investing_metrics(ticker):
    try:
        stock = yf.Ticker(ticker)
        pe_ratio = stock.info.get('trailingPE', None)
        pb_ratio = stock.info.get('priceToBook', None)
        dividend_yield = stock.info.get('dividendYield', None)

        return pe_ratio, pb_ratio, dividend_yield
    except Exception as e:
        print(f"Error fetching value investing metrics for '{ticker}': {str(e)}")
        return None, None, None

def fetch_stock_price_data(ticker):
    try:
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=3)).strftime('%Y-%m-%d')
        data = yf.download(ticker, start=start_date, end=end_date)
        monthly_data = data.resample('M').last()
        monthly_percent_change = monthly_data['Close'].pct_change() * 100
        return monthly_data['Close'], monthly_percent_change
    except Exception as e:
        print(f"Error fetching data for '{ticker}': {str(e)}")
        return None, None

def save_financial_metrics_to_csv(ticker):
    try:
        tickerData = yf.Ticker(ticker)
        quarterly_financials = tickerData.quarterly_balance_sheet.T

        # Specify the financial metrics you want to save
        metrics = [
            'Net Tangible Assets',
            'Total Debt',
            'Retained Earnings',
            'Common Stock Equity',
            'Total Liabilities Net Minority Interest',
            'Total Assets',
            'Current Assets',
            'Current Liabilities',
            'Cash And Cash Equivalents',
            'Working Capital'
        ]

        available_metrics = [metric for metric in metrics if metric in quarterly_financials.columns]

        # Fetch value investing metrics
        pe_ratio, pb_ratio, dividend_yield = fetch_value_investing_metrics(ticker)

        # Create a DataFrame for the value investing metrics
        value_investing_data = {
            'Metric': ['P/E Ratio', 'P/B Ratio', 'Dividend Yield'],
            'Value': [
                pe_ratio if pe_ratio is not None else 'N/A',
                pb_ratio if pb_ratio is not None else 'N/A',
                dividend_yield * 100 if dividend_yield is not None else 'N/A'  # Convert to percentage
            ]
        }
        value_investing_df = pd.DataFrame(value_investing_data)

        # Add quarterly financial metrics to a DataFrame
        if not quarterly_financials.empty:
            quarterly_financials_df = quarterly_financials[available_metrics]

            # Save both value investing metrics and quarterly financial metrics to a single CSV file
            combined_df = pd.concat([value_investing_df, quarterly_financials_df.reset_index()], axis=1)

            csv_filename = f"{ticker}_financial_metrics.csv"
            combined_df.to_csv(csv_filename, index=False)

            print(f"Financial metrics saved to {csv_filename}")

        else:
            print(f"No quarterly financial data available for {ticker}")

    except Exception as e:
        print(f"Error saving financial metrics for '{ticker}': {str(e)}")

if __name__ == "__main__":
    ticker = ticker

    # Fetch value investing metrics
    pe_ratio, pb_ratio, dividend_yield = fetch_value_investing_metrics(ticker)

    # Fetch stock price data (for other potential uses)
    monthly_closing_prices, monthly_percent_change = fetch_stock_price_data(ticker)

    # Save financial metrics to CSV
    save_financial_metrics_to_csv(ticker)

    # Print value investing metrics
    print(f"Value Investing Metrics for {ticker}:")

    if pe_ratio is not None:
        print(f"P/E Ratio for {ticker}: {pe_ratio:.2f}")

    if pb_ratio is not None:
        print(f"P/B Ratio for {ticker}: {pb_ratio:.2f}")

    if dividend_yield is not None:
        print(f"Dividend Yield for {ticker}: {dividend_yield * 100:.2f}%")


# Function to download data
def download_data(ticker, period="5y"):
    stock_data = yf.download(ticker, period=period)
    return stock_data

# Function to calculate SMA
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

# Function to calculate EMA
def calculate_ema(data, window):
    return data['Close'].ewm(span=window, adjust=False).mean()

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    sma = calculate_sma(data, window)
    std_dev = data['Close'].rolling(window=window).std()
    upper_band = sma + (std_dev * num_std_dev)
    lower_band = sma - (std_dev * num_std_dev)
    return upper_band, lower_band

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    short_ema = calculate_ema(data, short_window)
    long_ema = calculate_ema(data, long_window)
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Function to calculate ADX (Average Directional Index)
def calculate_adx(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']

    plus_dm = high.diff()
    minus_dm = low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))

    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)

    atr = tr.rolling(window).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/window).mean() / atr))

    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/window).mean()

    return adx

# Function to calculate Stochastic Oscillator
def calculate_stochastic(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    stochastic = 100 * (data['Close'] - low_min) / (high_max - low_min)
    return stochastic

# Function to calculate ATR (Average True Range)
def calculate_atr(data, window=14):
    high = data['High']
    low = data['Low']
    close = data['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr

# Function to calculate CCI (Commodity Channel Index)
def calculate_cci(data, window=20):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    cci = (tp - tp.rolling(window).mean()) / (0.015 * tp.rolling(window).std())
    return cci

# Function to calculate Williams %R
def calculate_williams_r(data, window=14):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    williams_r = (high_max - data['Close']) / (high_max - low_min) * -100
    return williams_r

# Function to calculate support and resistance levels for different trading styles
def calculate_support_resistance(data, window=20):
    support_level = data['Low'].rolling(window=window).min().iloc[-1]
    resistance_level = data['High'].rolling(window=window).max().iloc[-1]
    return support_level, resistance_level

def calculate_support_resistance_by_style(data):
    styles = {
        'Swing': 50,
        'Intraday': 10,
        'Long Term': 200,
        'Momentum': 30,
        'Scalping': 5
    }

    support_resistance = {}

    for style, window in styles.items():
        support, resistance = calculate_support_resistance(data, window)
        support_resistance[f'Support {style}'] = support
        support_resistance[f'Resistance {style}'] = resistance

    return support_resistance

# Function to determine buy/sell indications
def determine_indications(data):
    indications = {}

    # SMA
    if data['Close'].iloc[-1] > data['SMA_20'].iloc[-1]:
        indications['SMA'] = 'Buy'
    else:
        indications['SMA'] = 'Sell'

    # EMA
    if data['Close'].iloc[-1] > data['EMA_20'].iloc[-1]:
        indications['EMA'] = 'Buy'
    else:
        indications['EMA'] = 'Sell'

    # RSI
    if data['RSI_14'].iloc[-1] < 30:
        indications['RSI'] = 'Buy'
    elif data['RSI_14'].iloc[-1] > 70:
        indications['RSI'] = 'Sell'
    else:
        indications['RSI'] = 'Hold'

    # Bollinger Bands
    if data['Close'].iloc[-1] <= data['Lower_Band'].iloc[-1]:
        indications['Bollinger Bands'] = 'Buy'
    elif data['Close'].iloc[-1] >= data['Upper_Band'].iloc[-1]:
        indications['Bollinger Bands'] = 'Sell'
    else:
        indications['Bollinger Bands'] = 'Hold'

    # MACD
    if data['MACD'].iloc[-1] > data['Signal_Line'].iloc[-1]:
        indications['MACD'] = 'Buy'
    else:
        indications['MACD'] = 'Sell'

    # Stochastic Oscillator
    if data['Stochastic_14'].iloc[-1] < 20:
        indications['Stochastic'] = 'Buy'
    elif data['Stochastic_14'].iloc[-1] > 80:
        indications['Stochastic'] = 'Sell'
    else:
        indications['Stochastic'] = 'Hold'

    # CCI
    if data['CCI_20'].iloc[-1] < -100:
        indications['CCI'] = 'Buy'
    elif data['CCI_20'].iloc[-1] > 100:
        indications['CCI'] = 'Sell'
    else:
        indications['CCI'] = 'Hold'

    # Williams %R
    if data['Williams_%R_14'].iloc[-1] < -80:
        indications['Williams %R'] = 'Buy'
    elif data['Williams_%R_14'].iloc[-1] > -20:
        indications['Williams %R'] = 'Sell'
    else:
        indications['Williams %R'] = 'Hold'

    return indications

# Main function to perform analysis and save results to CSV
def perform_analysis(ticker, save_csv=True):
    # Download data for the past 5 years
    data = download_data(ticker, period="5y")

    # Calculate technical indicators
    data['SMA_20'] = calculate_sma(data, 20)
    data['EMA_20'] = calculate_ema(data, 20)
    data['RSI_14'] = calculate_rsi(data, 14)
    data['Upper_Band'], data['Lower_Band'] = calculate_bollinger_bands(data, 20)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['Stochastic_14'] = calculate_stochastic(data, 14)
    data['CCI_20'] = calculate_cci(data, 20)
    data['Williams_%R_14'] = calculate_williams_r(data, 14)
    data['ADX_14'] = calculate_adx(data, 14)

    # Calculate support and resistance levels for different styles
    support_resistance = calculate_support_resistance_by_style(data)

    # Determine indications
    indications = determine_indications(data)

    # Prepare data for CSV export
    result = {
        'Indicator': list(indications.keys()) + list(support_resistance.keys()),
        'Indication': list(indications.values()) + [f'{support_resistance[key]:.2f}' for key in support_resistance]
    }

    result_df = pd.DataFrame(result)

    # Save results to CSV
    if save_csv:
        result_df.to_csv(f'{ticker}_indications.csv', index=False)

    return result_df

# Example usage
ticker = ticker
analysis_results = perform_analysis(ticker)

# List of CSV filenames with the {ticker} placeholder replaced
ticker = ticker
filenames = [
    f'{ticker}_annual_summary.csv',
    f'{ticker}_financial_metrics.csv',
    f'{ticker}_indications.csv',
    f'{ticker}_investment_summary.csv',
    f'{ticker}_monthly_data.csv',
    f'{ticker}_weekly_analysis.csv',
    f'{ticker}_weekly_closing_prices.csv'
]

# Create a new Excel writer object
output_excel_file = f'/content/{ticker}_compiled_data.xlsx'
with pd.ExcelWriter(output_excel_file, engine='xlsxwriter') as writer:
    # Loop through each CSV file and save it as a new sheet in the Excel file
    for i, filename in enumerate(filenames, start=1):
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            sheet_name = f'Sheet{i}'  # Name each sheet as Sheet1, Sheet2, etc.
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            os.remove(filename)  # Delete the CSV file after adding it to the Excel sheet
        else:
            print(f"File {filename} not found.")

print(f'All files have been compiled into {output_excel_file} SAVED.')
