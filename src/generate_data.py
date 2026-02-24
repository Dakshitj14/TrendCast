"""
TrendCast - Data Generator
Simulates the Kaggle "S&P 500 Stock Data" dataset (by Cam Nugent)
Dataset: https://www.kaggle.com/datasets/camnugent/sandp500
Contains historical OHLCV data for S&P 500 companies (2013-2018)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

np.random.seed(42)

TICKERS = {
    "AAPL":  {"start": 150, "vol": 0.018, "drift": 0.0003},
    "MSFT":  {"start":  60, "vol": 0.016, "drift": 0.0004},
    "GOOGL": {"start": 550, "vol": 0.014, "drift": 0.0003},
    "AMZN":  {"start": 300, "vol": 0.020, "drift": 0.0005},
    "TSLA":  {"start": 200, "vol": 0.038, "drift": 0.0004},
    "JPM":   {"start":  55, "vol": 0.016, "drift": 0.0002},
    "JNJ":   {"start":  95, "vol": 0.010, "drift": 0.0002},
    "NVDA":  {"start":  20, "vol": 0.028, "drift": 0.0006},
    "META":  {"start":  55, "vol": 0.022, "drift": 0.0004},
    "XOM":   {"start":  90, "vol": 0.014, "drift": 0.0001},
}

def generate_ohlcv(ticker, params, dates):
    """Generate realistic OHLCV data using Geometric Brownian Motion."""
    n = len(dates)
    returns = np.random.normal(params["drift"], params["vol"], n)
    
    # Add some regime changes and momentum
    for i in range(1, n):
        if np.random.rand() < 0.002:  # 0.2% chance of crash/rally
            returns[i] += np.random.choice([-0.05, 0.05])
    
    prices = [params["start"]]
    for r in returns[1:]:
        prices.append(prices[-1] * (1 + r))
    
    prices = np.array(prices)
    
    rows = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        daily_vol = close * params["vol"]
        high = close + abs(np.random.normal(0, daily_vol * 0.6))
        low  = close - abs(np.random.normal(0, daily_vol * 0.6))
        open_ = prices[i-1] if i > 0 else close
        open_ *= (1 + np.random.normal(0, params["vol"] * 0.3))
        volume = int(np.random.lognormal(15, 0.5))
        
        rows.append({
            "date": date.strftime("%Y-%m-%d"),
            "open": round(max(open_, 1), 2),
            "high": round(max(high, close, open_), 2),
            "low":  round(min(low, close, open_), 2),
            "close": round(close, 2),
            "volume": volume,
            "Name": ticker
        })
    return rows

def generate_dataset():
    start_date = datetime(2013, 2, 8)
    end_date   = datetime(2018, 2, 7)
    all_dates  = pd.bdate_range(start=start_date, end=end_date).tolist()
    
    all_rows = []
    for ticker, params in TICKERS.items():
        rows = generate_ohlcv(ticker, params, all_dates)
        all_rows.extend(rows)
    
    df = pd.DataFrame(all_rows)
    df = df.sort_values(["Name", "date"]).reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/all_stocks_5yr.csv", index=False)
    print(f"✅ Dataset generated : {len(df):,} rows, {df['Name'].nunique()} tickers")
    print(f"   Date range: {df['date'].min()} → {df['date'].max()}")
    return df

if __name__ == "__main__":
    generate_dataset()
