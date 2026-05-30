import pandas as pd
df = pd.read_csv("dataset.csv", parse_dates=["date"])
print("Total rows:", len(df))
print("gdelt_missing == 1:", df["gdelt_missing"].sum())
print("By month:")
print(df[df["gdelt_missing"]==1].groupby(df["date"].dt.to_period("M")).size())

# Should see ~112 missing rows clustered in 2024-12, 2025-06, 2025-07, 2026-04 — 
    # same dates as before, but now flagged. 