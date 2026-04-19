# Data Sanity Check

import pandas as pd

df = pd.read_csv('dataset.csv', parse_dates=['date'])

# 1. Shape — expect ~2,500–3,000 rows, ~35 columns
print(df.shape)

# 2. Date range
print(df['date'].min(), "→", df['date'].max())

# 3. Row count per ticker — should be roughly equal (~350 each)
print(df.groupby('ticker').size())

# 4. Class balance — expect ~53–56% "up" days
print(df['direction_t1'].value_counts(normalize=True))

# 5. Missing values — lag columns will have NaNs for earliest dates; that's expected
print(df.isnull().sum()[df.isnull().sum() > 0])

# 6. Sanity check: NVDA should have negative/mixed sentiment around DeepSeek event
nvda = df[df['ticker'] == 'NVDA'].set_index('date').sort_index()
print(nvda.loc['2025-01-25':'2025-02-05', ['tone_weighted', 'direction_t1']])
