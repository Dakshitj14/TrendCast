"""
TrendCast - Web UI (Flask)
Run: python app.py
Then open: http://localhost:5000
"""

import os, sys, json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from flask import Flask, render_template, jsonify, request
from src.generate_data import generate_dataset
from src.features import build_features
from src.predict import predict_trend, FEATURE_COLS

import joblib

app = Flask(__name__)

# ── Load data once at startup ──────────────────────────────────────────────
print("⏳ Loading dataset...")
RAW_DF = generate_dataset()
FEAT_DF = build_features(RAW_DF)
print("✅ Ready!")

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "JPM", "JNJ", "NVDA", "META", "XOM"]

TICKER_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.", "TSLA": "Tesla Inc.", "JPM": "JPMorgan Chase",
    "JNJ": "Johnson & Johnson", "NVDA": "NVIDIA Corp.", "META": "Meta Platforms",
    "XOM": "Exxon Mobil Corp."
}

TICKER_SECTORS = {
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Consumer", "TSLA": "Automotive", "JPM": "Finance",
    "JNJ": "Healthcare", "NVDA": "Semiconductors", "META": "Technology",
    "XOM": "Energy"
}

@app.route("/")
def index():
    return render_template("index.html", tickers=TICKERS, ticker_names=TICKER_NAMES)

@app.route("/api/predict/<ticker>")
def api_predict(ticker):
    if ticker not in TICKERS:
        return jsonify({"error": "Unknown ticker"}), 400
    try:
        result = predict_trend(ticker, RAW_DF)
        # Add price history for chart
        grp = FEAT_DF[FEAT_DF["Name"] == ticker].sort_values("date").tail(60)
        result["price_history"] = {
            "dates":  grp["date"].astype(str).tolist(),
            "close":  grp["close"].round(2).tolist(),
            "sma20":  grp["sma_20"].round(2).tolist(),
            "sma50":  grp["sma_50"].round(2).tolist(),
            "bb_upper": grp["bb_upper"].round(2).tolist(),
            "bb_lower": grp["bb_lower"].round(2).tolist(),
        }
        result["rsi_history"] = grp["rsi_14"].round(2).tolist()
        result["macd_history"] = {
            "macd":   grp["macd"].round(4).tolist(),
            "signal": grp["macd_signal"].round(4).tolist(),
            "hist":   grp["macd_hist"].round(4).tolist(),
        }
        result["name"]   = TICKER_NAMES.get(ticker, ticker)
        result["sector"] = TICKER_SECTORS.get(ticker, "")
        result["current_price"] = round(float(grp["close"].iloc[-1]), 2)
        result["price_change"]  = round(float(grp["return_1d"].iloc[-1]) * 100, 2)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/overview")
def api_overview():
    results = []
    for ticker in TICKERS:
        try:
            r = predict_trend(ticker, RAW_DF)
            grp = FEAT_DF[FEAT_DF["Name"] == ticker].sort_values("date").tail(2)
            results.append({
                "ticker":      ticker,
                "name":        TICKER_NAMES[ticker],
                "sector":      TICKER_SECTORS[ticker],
                "prediction":  r["prediction"],
                "prob_up":     r["probability_up"],
                "confidence":  r["confidence"],
                "price":       round(float(grp["close"].iloc[-1]), 2),
                "change_pct":  round(float(grp["return_1d"].iloc[-1]) * 100, 2),
                "rsi":         round(float(grp["rsi_14"].iloc[-1]), 1),
                "volatility":  round(float(grp["volatility_20"].iloc[-1]) * 100, 2),
            })
        except:
            pass
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=False, port=5000)
