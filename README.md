# 🔮 TrendCast — Stock Trend Prediction Engine

> Predicts **5-day stock price movement (Up/Down)** using historical OHLCV data and 32 technical indicators.

---

## 📊 Dataset

Based on the **Kaggle S&P 500 Historical Data** dataset by Cam Nugent:  
👉 https://www.kaggle.com/datasets/camnugent/sandp500

- **Tickers**: AAPL, MSFT, GOOGL, AMZN, TSLA, JPM, JNJ, NVDA, META, XOM  
- **Period**: Feb 2013 – Feb 2018 (5 years, 1,304 trading days)  
- **Columns**: `date, open, high, low, close, volume, Name`

---

## 🏗️ Project Structure

```
TrendCast/
├── main.py                  # ← Run this to train everything
├── requirements.txt
├── data/
│   ├── all_stocks_5yr.csv   # Raw OHLCV data
│   └── features.csv         # Engineered features
├── src/
│   ├── generate_data.py     # Dataset generation
│   ├── features.py          # Technical indicator engineering
│   ├── train.py             # Model training & evaluation
│   └── predict.py           # Inference / signal generation
├── models/
│   ├── best_model.pkl       # Best performing model
│   └── scaler.pkl           # Feature scaler
├── outputs/
│   └── trendcast_dashboard.png
└── reports/
    └── metrics.json
```

---

## ⚙️ Technical Indicators (32 Features)

| Category         | Indicators                                              |
|------------------|---------------------------------------------------------|
| Price Returns    | 1d, 5d, 10d, 20d returns + log return                  |
| Moving Averages  | SMA & EMA (5, 10, 20, 50) + price-to-SMA ratios        |
| Momentum         | RSI(7, 14), MACD + Signal + Histogram, Mom(10, 20)     |
| Volatility       | Bollinger Bands (width, %B), ATR(14), Rolling Vol       |
| Volume           | OBV + OBV signal, Volume ratio to 20-day avg           |
| Oscillators      | Stochastic K & D, Williams %R                          |

---

## 🤖 Models Trained

| Model                | Type        | Notes                          |
|----------------------|-------------|--------------------------------|
| Logistic Regression  | Linear      | L2 regularized baseline        |
| Random Forest        | Ensemble    | 200 trees, max_depth=8         |
| Gradient Boosting    | Ensemble    | 200 estimators, lr=0.05        |
| SVM                  | Kernel      | RBF kernel, probability=True   |

**Target**: Binary — `1` if price is higher 5 days later, `0` if lower.  
**Split**: Temporal (last 20% of dates = test). No future data leakage.

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run predictions for specific tickers
python src/predict.py
```

---

## 📈 Dashboard

`outputs/trendcast_dashboard.png` contains:
- Model comparison (Accuracy + ROC-AUC)
- ROC Curves for all 4 models
- Precision-Recall Curves
- Confusion Matrix (best model)
- Top 20 Feature Importances
- Prediction Probability Distribution
- Rolling Accuracy Over Time
- Bullish Signal Strength by Ticker

---

## 📝 Notes

- Stock price prediction is inherently noisy (~50% accuracy is expected near market efficiency)
- ROC-AUC > 0.55 indicates the model is capturing some predictive signal
- The project is designed as a **learning framework** — production use requires additional risk management
