"""
TrendCast - Feature Engineering
Computes technical indicators from raw OHLCV data.
Indicators: RSI, MACD, Bollinger Bands, SMA, EMA, ATR, OBV, Stochastic, Williams %R
"""

import pandas as pd
import numpy as np

# ─── Technical Indicator Functions ──────────────────────────────────────────

def sma(series, window):
    return series.rolling(window=window).mean()

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    fast_ema   = ema(series, fast)
    slow_ema   = ema(series, slow)
    macd_line  = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, window=20, num_std=2):
    mid  = sma(series, window)
    std  = series.rolling(window=window).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    width = (upper - lower) / (mid + 1e-9)
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return upper, mid, lower, width, pct_b

def atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def obv(close, volume):
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (direction * volume).cumsum()

def stochastic(high, low, close, k_period=14, d_period=3):
    lowest_low   = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-9)
    d = k.rolling(window=d_period).mean()
    return k, d

def williams_r(high, low, close, period=14):
    highest_high = high.rolling(window=period).max()
    lowest_low   = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)

def momentum(series, period=10):
    return series / series.shift(period) - 1

# ─── Main Feature Builder ────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [date, open, high, low, close, volume, Name]
    Output: DataFrame with all technical features + binary target label
    """
    all_frames = []

    for ticker, grp in df.groupby("Name"):
        g = grp.copy().sort_values("date").reset_index(drop=True)
        c = g["close"]
        h = g["high"]
        l = g["low"]
        v = g["volume"]

        # ── Price & Volume Features ──
        g["return_1d"]  = c.pct_change(1)
        g["return_5d"]  = c.pct_change(5)
        g["return_10d"] = c.pct_change(10)
        g["return_20d"] = c.pct_change(20)
        g["log_return"] = np.log(c / c.shift(1))

        # ── Moving Averages ──
        for w in [5, 10, 20, 50]:
            g[f"sma_{w}"] = sma(c, w)
            g[f"ema_{w}"] = ema(c, w)
            g[f"close_to_sma{w}"] = c / (g[f"sma_{w}"] + 1e-9) - 1

        # ── RSI ──
        g["rsi_14"] = rsi(c, 14)
        g["rsi_7"]  = rsi(c, 7)

        # ── MACD ──
        g["macd"], g["macd_signal"], g["macd_hist"] = macd(c)
        g["macd_cross"] = (g["macd"] > g["macd_signal"]).astype(int)

        # ── Bollinger Bands ──
        g["bb_upper"], g["bb_mid"], g["bb_lower"], g["bb_width"], g["bb_pct"] = bollinger_bands(c)

        # ── ATR (Volatility) ──
        g["atr_14"]  = atr(h, l, c, 14)
        g["atr_pct"] = g["atr_14"] / (c + 1e-9)

        # ── OBV ──
        g["obv"]        = obv(c, v)
        g["obv_sma20"]  = sma(g["obv"], 20)
        g["obv_signal"] = (g["obv"] > g["obv_sma20"]).astype(int)

        # ── Stochastic ──
        g["stoch_k"], g["stoch_d"] = stochastic(h, l, c)
        g["stoch_cross"] = (g["stoch_k"] > g["stoch_d"]).astype(int)

        # ── Williams %R ──
        g["williams_r"] = williams_r(h, l, c)

        # ── Momentum ──
        g["mom_10"] = momentum(c, 10)
        g["mom_20"] = momentum(c, 20)

        # ── Volume Features ──
        g["vol_sma20"]  = sma(v, 20)
        g["vol_ratio"]  = v / (g["vol_sma20"] + 1e-9)
        g["vwap_proxy"] = (h + l + c) / 3

        # ── Rolling Volatility ──
        g["volatility_10"] = g["log_return"].rolling(10).std()
        g["volatility_20"] = g["log_return"].rolling(20).std()

        # ── Target: 1 if price goes up in next 5 days, else 0 ──
        g["target"]         = (c.shift(-5) > c).astype(int)
        g["future_return5d"] = c.shift(-5) / c - 1

        all_frames.append(g)

    result = pd.concat(all_frames, ignore_index=True)
    result = result.dropna().reset_index(drop=True)
    return result


if __name__ == "__main__":
    import os
    os.chdir("/home/claude/TrendCast")
    from src.generate_data import generate_dataset
    raw = generate_dataset()
    feat = build_features(raw)
    feat.to_csv("data/features.csv", index=False)
    print(f"✅ Features built: {feat.shape[0]:,} rows × {feat.shape[1]} columns")
    print(f"   Target distribution: {feat['target'].value_counts().to_dict()}")
