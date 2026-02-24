"""
TrendCast — Main Entry Point
Run the full pipeline: generate data → engineer features → train models → visualize
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

from src.generate_data import generate_dataset
from src.features import build_features
from src.train import train_and_evaluate, plot_all, FEATURE_COLS

print("""
╔══════════════════════════════════════════════════════╗
║                                                      ║
║   🔮  TrendCast — Stock Trend Prediction Engine       ║
║       Kaggle Dataset: S&P 500 Historical Data        ║
║                                                      ║
╚══════════════════════════════════════════════════════╝
""")

# Step 1 — Generate / Load Data
print("Step 1/3 — Generating dataset (Kaggle S&P500 simulation)...")
raw = generate_dataset()

# Step 2 — Feature Engineering
print("\nStep 2/3 — Engineering technical indicators...")
df = build_features(raw)
df.to_csv("data/features.csv", index=False)
print(f"          {df.shape[0]:,} samples × {df.shape[1]} features ready")

# Step 3 — Train & Visualize
print("\nStep 3/3 — Training models & generating dashboard...")
results, trained_models, scaler, X_test_sc, y_test, test_df, best_name = train_and_evaluate(df)
plot_all(results, trained_models, scaler, X_test_sc, y_test, test_df, best_name)

print(f"""
╔══════════════════════════════════════════════════════╗
║  ✅  TrendCast pipeline complete!
║
║  Best Model : {best_name:<36s}║
║  ROC-AUC   : {results[best_name]['roc_auc']:.4f}                               ║
║  Accuracy  : {results[best_name]['accuracy']:.4f}                               ║
║
║  Outputs:
║   📊  outputs/trendcast_dashboard.png
║   🧠  models/best_model.pkl
║   📄  reports/metrics.json
╚══════════════════════════════════════════════════════╝
""")
