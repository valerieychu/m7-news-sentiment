import pandas as pd
import numpy as np

# ── Load raw files ────────────────────────────────────────────────────────────
prices   = pd.read_csv('stock_prices.csv',   parse_dates=['date'])
sentiment = pd.read_csv('gdelt_sentiment.csv', parse_dates=['article_date'])

prices['date']            = pd.to_datetime(prices['date']).dt.normalize()
sentiment['article_date'] = pd.to_datetime(sentiment['article_date']).dt.normalize()
sentiment = sentiment.rename(columns={'article_date': 'date'})

print(f"Prices:    {len(prices)} rows")
print(f"Sentiment: {len(sentiment)} rows")

# ── Join on (ticker, date) ────────────────────────────────────────────────────
# Use prices as the spine — we only care about trading days.
# GDELT has data on weekends; those rows have no matching price and are dropped.
df = prices.merge(sentiment, on=['ticker', 'date'], how='left')

print(f"After join: {len(df)} rows")
print(f"Dates missing sentiment: {df['article_count'].isna().sum()} rows")

# Fill missing sentiment rows with 0 (no news = neutral signal)
# Only fill numeric sentiment columns, not identifiers
sentiment_cols = [
    'article_count', 'unique_source_count', 'total_word_count',
    'tone_weighted', 'tone_positive_weighted', 'tone_negative_weighted',
    'tone_polarity_weighted', 'tone_avg', 'tone_stddev',
    'tone_min', 'tone_max',
    'positive_article_count', 'negative_article_count', 'neutral_article_count',
    'avg_activity_density'
]
df[sentiment_cols] = df[sentiment_cols].fillna(0)

# ── Sort for lag computation ──────────────────────────────────────────────────
df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

# ── Compute next-day direction label (the target variable) ───────────────────
df['close_next'] = df.groupby('ticker')['close'].shift(-1)
df['return_t1']  = (df['close_next'] - df['close']) / df['close']
df['direction_t1'] = (df['return_t1'] > 0).astype(float)

# Secondary targets
df['close_t2'] = df.groupby('ticker')['close'].shift(-2)
df['close_t3'] = df.groupby('ticker')['close'].shift(-3)
df['direction_t2'] = ((df['close_t2'] - df['close']) / df['close'] > 0).astype(float)
df['direction_t3'] = ((df['close_t3'] - df['close']) / df['close'] > 0).astype(float)

# ── Sentiment lag features ────────────────────────────────────────────────────
for lag in [1, 2, 3]:
    df[f'tone_lag{lag}']     = df.groupby('ticker')['tone_weighted'].shift(lag)
    df[f'articles_lag{lag}'] = df.groupby('ticker')['article_count'].shift(lag)
    df[f'polarity_lag{lag}'] = df.groupby('ticker')['tone_polarity_weighted'].shift(lag)

# 5-day rolling average of tone (within each ticker group)
df['tone_rolling5'] = df.groupby('ticker')['tone_weighted'].transform(
    lambda x: x.rolling(5, min_periods=1).mean()
)

# Tone momentum: how much did sentiment change since yesterday?
df['tone_momentum'] = df['tone_weighted'] - df['tone_lag1']

# Sentiment volatility: rolling std of tone
df['tone_vol5'] = df.groupby('ticker')['tone_weighted'].transform(
    lambda x: x.rolling(5, min_periods=2).std()
)

# ── Price features ────────────────────────────────────────────────────────────
# Realized volatility: 5-day rolling std of daily returns
df['daily_return'] = df.groupby('ticker')['close'].pct_change()
df['realized_vol5'] = df.groupby('ticker')['daily_return'].transform(
    lambda x: x.rolling(5, min_periods=2).std()
)

# ── Earnings week flag ────────────────────────────────────────────────────────
# Known approximate M7 earnings months: Jan, Apr, Jul, Oct
# Flag the week of the 15th-30th in those months as "earnings week"
# Refine these dates manually if you want precision
df['month'] = df['date'].dt.month
df['day']   = df['date'].dt.day
df['earnings_week'] = (
    df['month'].isin([1, 4, 7, 10]) & df['day'].between(15, 30)
).astype(int)

# ── Drop rows without a valid label ──────────────────────────────────────────
# The last 3 rows per ticker have no t+1/t+2/t+3 close — remove them
df = df.dropna(subset=['direction_t1'])

# ── Final column selection ────────────────────────────────────────────────────
final_cols = [
    # Identifiers
    'date', 'ticker',

    # Price data (for reference, not model features)
    'open', 'high', 'low', 'close', 'volume',
    'daily_return', 'realized_vol5',

    # Raw sentiment
    'article_count', 'unique_source_count',
    'tone_weighted', 'tone_positive_weighted', 'tone_negative_weighted',
    'tone_polarity_weighted', 'tone_avg', 'tone_stddev',
    'positive_article_count', 'negative_article_count', 'neutral_article_count',
    'avg_activity_density',

    # Engineered sentiment features
    'tone_lag1', 'tone_lag2', 'tone_lag3',
    'articles_lag1', 'articles_lag2', 'articles_lag3',
    'polarity_lag1', 'polarity_lag2', 'polarity_lag3',
    'tone_rolling5', 'tone_momentum', 'tone_vol5',

    # Calendar features
    'earnings_week',

    # Target variables
    'direction_t1', 'direction_t2', 'direction_t3',
    'return_t1',    # continuous version of t+1 target
]

df = df[[c for c in final_cols if c in df.columns]]

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv('dataset.csv', index=False)

print(f"\n✓ Saved dataset.csv")
print(f"  Shape: {df.shape}")
print(f"  Date range: {df['date'].min()} → {df['date'].max()}")
print(f"  Tickers: {sorted(df['ticker'].unique())}")
print(f"\nClass balance (direction_t1):")
print(df['direction_t1'].value_counts(normalize=True).round(3))
print(f"\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])