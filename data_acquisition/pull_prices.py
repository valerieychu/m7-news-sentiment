import yfinance as yf
import pandas as pd

tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']

# Start a few days before Jan 1 2025 so lag features work for earliest dates
START = '2024-12-25'
END   = '2026-04-14'

print("Downloading price data...")
frames = []

for ticker in tickers:
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    df = df.reset_index()
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
    df['ticker'] = ticker
    keep = ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    df = df[[c for c in keep if c in df.columns]]
    frames.append(df)
    print(f"  {ticker}: {len(df)} rows")

prices = pd.concat(frames, ignore_index=True)
prices['date'] = pd.to_datetime(prices['date']).dt.date
prices.to_csv('stock_prices.csv', index=False)
print(f"\nDone. Saved stock_prices.csv — {len(prices)} total rows")