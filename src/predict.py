"""
TrendCast - Prediction Module
Load trained model and predict trend for a given stock ticker.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# ─── Fix paths so it works when run directly or as module ────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.features import build_features

# ─── Feature list (copied here to avoid circular import) ─────────────────────
FEATURE_COLS = [
    "return_1d", "return_5d", "return_10d", "return_20d", "log_return",
    "close_to_sma5", "close_to_sma10", "close_to_sma20", "close_to_sma50",
    "ema_5", "ema_10", "ema_20", "ema_50",
    "rsi_14", "rsi_7",
    "macd", "macd_signal", "macd_hist", "macd_cross",
    "bb_width", "bb_pct",
    "atr_pct",
    "obv_signal",
    "stoch_k", "stoch_d", "stoch_cross",
    "williams_r",
    "mom_10", "mom_20",
    "vol_ratio",
    "volatility_10", "volatility_20",
]

def predict_trend(ticker: str, raw_df: pd.DataFrame):
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
    model  = joblib.load(os.path.join(BASE_DIR, "models", "best_model.pkl"))

    ticker_data = raw_df[raw_df["Name"] == ticker].copy()
    if len(ticker_data) < 60:
        raise ValueError(f"Not enough data for {ticker}. Need at least 60 rows.")

    feat_df = build_features(ticker_data).dropna()

    latest = feat_df.tail(1).copy()
    for f in FEATURE_COLS:
        if f not in latest.columns:
            latest[f] = 0.0
    latest = latest[FEATURE_COLS]

    X_scaled   = scaler.transform(latest)
    prob_up    = model.predict_proba(X_scaled)[0][1]
    pred       = "UP ↑" if prob_up >= 0.5 else "DOWN ↓"
    confidence = prob_up if prob_up >= 0.5 else 1 - prob_up

    last_row = feat_df.iloc[-1]
    signals  = {
        "RSI(14)":        round(float(last_row.get("rsi_14", np.nan)), 2),
        "MACD Histogram": round(float(last_row.get("macd_hist", np.nan)), 4),
        "Bollinger %B":   round(float(last_row.get("bb_pct", np.nan)), 4),
        "Stochastic K":   round(float(last_row.get("stoch_k", np.nan)), 2),
        "Williams %R":    round(float(last_row.get("williams_r", np.nan)), 2),
        "OBV Signal":     int(last_row.get("obv_signal", 0)),
        "Mom(20)":        round(float(last_row.get("mom_20", np.nan)), 4),
        "Volatility(20)": round(float(last_row.get("volatility_20", np.nan)), 4),
    }

    return {
        "ticker":         ticker,
        "prediction":     pred,
        "probability_up": round(float(prob_up), 4),
        "confidence":     round(float(confidence), 4),
        "signals":        signals,
    }


if __name__ == "__main__":
    from src.generate_data import generate_dataset

    print("\n🔮 TrendCast — Generating predictions...\n")
    raw = generate_dataset()

    for ticker in ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]:
        result = predict_trend(ticker, raw)
        print(f"{'─'*45}")
        print(f"  📈 TrendCast Signal : {result['ticker']}")
        print(f"  Prediction         : {result['prediction']}")
        print(f"  P(Up)              : {result['probability_up']:.2%}")
        print(f"  Confidence         : {result['confidence']:.2%}")
        print(f"  Indicators         :")
        for k, v in result["signals"].items():
            print(f"    {k:20s} = {v}")
    print(f"{'─'*45}")