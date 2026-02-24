"""
TrendCast - Model Training
Trains and evaluates multiple classifiers for stock trend prediction.
Models: Logistic Regression, Random Forest, Gradient Boosting, SVM
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance

# ─── Configuration ───────────────────────────────────────────────────────────

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

TARGET_COL = "target"

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   min_samples_leaf=20, random_state=42, n_jobs=-1),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
    "SVM":                 SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}

# ─── Training & Evaluation ───────────────────────────────────────────────────

def load_features(path="data/features.csv"):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def time_split(df, test_ratio=0.2):
    """Temporal train/test split — never leak future data."""
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
    return train, test

def train_and_evaluate(df):
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # ── Data prep ──
    df_clean = df[FEATURE_COLS + [TARGET_COL, "date", "Name"]].dropna()
    train_df, test_df = time_split(df_clean)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]
    X_test  = test_df[FEATURE_COLS]
    y_test  = test_df[TARGET_COL]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, "models/scaler.pkl")

    print(f"\n{'='*60}")
    print(f"  TrendCast — Model Training")
    print(f"{'='*60}")
    print(f"  Train samples : {len(X_train):,}")
    print(f"  Test  samples : {len(X_test):,}")
    print(f"  Features      : {len(FEATURE_COLS)}")
    print(f"  Target balance: {y_train.value_counts().to_dict()}")
    print(f"{'='*60}\n")

    results = {}
    trained_models = {}

    for name, model in MODELS.items():
        print(f"🔧 Training: {name} ...")
        model.fit(X_train_sc, y_train)

        y_pred      = model.predict(X_test_sc)
        y_proba     = model.predict_proba(X_test_sc)[:, 1]
        acc         = accuracy_score(y_test, y_pred)
        roc_auc     = roc_auc_score(y_test, y_proba)
        avg_prec    = average_precision_score(y_test, y_proba)

        results[name] = {
            "accuracy":   round(acc, 4),
            "roc_auc":    round(roc_auc, 4),
            "avg_precision": round(avg_prec, 4),
            "y_pred":     y_pred,
            "y_proba":    y_proba,
            "report":     classification_report(y_test, y_pred, output_dict=True),
        }
        trained_models[name] = model

        # Save model
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, f"models/{safe_name}.pkl")
        print(f"   Accuracy={acc:.4f}  ROC-AUC={roc_auc:.4f}  AvgPrecision={avg_prec:.4f}")

    # ── Pick best model ──
    best_name = max(results, key=lambda n: results[n]["roc_auc"])
    print(f"\n🏆 Best model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.4f})")
    joblib.dump(trained_models[best_name], "models/best_model.pkl")

    # ── Save metrics ──
    metrics_out = {k: {m: v for m, v in v.items() if isinstance(v, (int, float))}
                   for k, v in results.items()}
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    return results, trained_models, scaler, X_test_sc, y_test, test_df, best_name

# ─── Plotting ────────────────────────────────────────────────────────────────

PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#55A868",
    "Gradient Boosting":   "#C44E52",
    "SVM":                 "#8172B2",
}

def plot_all(results, trained_models, scaler, X_test_sc, y_test, test_df, best_name):
    fig = plt.figure(figsize=(22, 28), facecolor="#0d1117")
    fig.suptitle("TrendCast — ML Stock Trend Prediction Dashboard",
                 fontsize=22, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.04, left=0.06, right=0.97)

    dark_ax_kw = dict(facecolor="#161b22", frameon=True)

    def style_ax(ax, title="", xlabel="", ylabel=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        if title:  ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        if xlabel: ax.set_xlabel(xlabel, fontsize=9)
        if ylabel: ax.set_ylabel(ylabel, fontsize=9)

    # ── 1. Model Comparison Bar ──
    ax1 = fig.add_subplot(gs[0, 0])
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    aucs   = [results[n]["roc_auc"]  for n in names]
    x      = np.arange(len(names))
    bars1  = ax1.bar(x - 0.2, accs, 0.35, label="Accuracy",  color=[PALETTE[n] for n in names], alpha=0.85)
    bars2  = ax1.bar(x + 0.2, aucs, 0.35, label="ROC-AUC",   color=[PALETTE[n] for n in names], alpha=0.5, hatch="//")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=8)
    ax1.set_ylim(0.4, 1.0)
    ax1.legend(facecolor="#1c2128", labelcolor="white", fontsize=8)
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
                 ha="center", va="bottom", color="white", fontsize=7)
    style_ax(ax1, "Model Comparison", "", "Score")

    # ── 2. ROC Curves ──
    ax2 = fig.add_subplot(gs[0, 1])
    for name in names:
        fpr, tpr, _ = roc_curve(y_test, results[name]["y_proba"])
        ax2.plot(fpr, tpr, label=f"{name} ({results[name]['roc_auc']:.3f})",
                 color=PALETTE[name], linewidth=2)
    ax2.plot([0,1],[0,1], "w--", linewidth=1, alpha=0.4)
    ax2.legend(facecolor="#1c2128", labelcolor="white", fontsize=7)
    style_ax(ax2, "ROC Curves", "False Positive Rate", "True Positive Rate")

    # ── 3. Precision-Recall Curves ──
    ax3 = fig.add_subplot(gs[0, 2])
    for name in names:
        prec, rec, _ = precision_recall_curve(y_test, results[name]["y_proba"])
        ax3.plot(rec, prec, label=f"{name} (AP={results[name]['avg_precision']:.3f})",
                 color=PALETTE[name], linewidth=2)
    ax3.legend(facecolor="#1c2128", labelcolor="white", fontsize=7)
    style_ax(ax3, "Precision-Recall Curves", "Recall", "Precision")

    # ── 4. Confusion Matrix (best model) ──
    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
                xticklabels=["↓ Down", "↑ Up"], yticklabels=["↓ Down", "↑ Up"],
                linewidths=0.5, linecolor="#30363d",
                annot_kws={"color":"white","size":12})
    ax4.set_xlabel("Predicted", color="white")
    ax4.set_ylabel("Actual", color="white")
    style_ax(ax4, f"Confusion Matrix — {best_name}")

    # ── 5. Feature Importance ──
    ax5 = fig.add_subplot(gs[1, 1:])
    best_model = trained_models[best_name]
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    else:
        perm = permutation_importance(best_model, X_test_sc, y_test, n_repeats=5, random_state=42)
        importances = perm.importances_mean

    fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=True).tail(20)
    bars = ax5.barh(fi.index, fi.values, color="#55A868", alpha=0.85)
    for bar, val in zip(bars, fi.values):
        ax5.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", color="white", fontsize=7)
    style_ax(ax5, f"Top 20 Feature Importances ({best_name})", "Importance", "")
    ax5.tick_params(axis="y", labelsize=8, colors="white")

    # ── 6. Predicted Probability Distribution ──
    ax6 = fig.add_subplot(gs[2, 0])
    proba = results[best_name]["y_proba"]
    ax6.hist(proba[y_test == 0], bins=40, alpha=0.7, color="#C44E52", label="Down (0)", density=True)
    ax6.hist(proba[y_test == 1], bins=40, alpha=0.7, color="#55A868", label="Up (1)",   density=True)
    ax6.axvline(0.5, color="white", linestyle="--", linewidth=1)
    ax6.legend(facecolor="#1c2128", labelcolor="white", fontsize=8)
    style_ax(ax6, "Prediction Probability Distribution", "P(Up)", "Density")

    # ── 7. Rolling Accuracy Over Time ──
    ax7 = fig.add_subplot(gs[2, 1:])
    test_df2 = test_df.copy().reset_index(drop=True)
    test_df2["correct"]      = (results[best_name]["y_pred"] == y_test.values)
    test_df2["rolling_acc"]  = test_df2["correct"].rolling(50).mean()
    test_df2_grp = test_df2.groupby("date")["rolling_acc"].mean().dropna()
    ax7.plot(test_df2_grp.index, test_df2_grp.values, color="#4C72B0", linewidth=1.5)
    ax7.axhline(0.5, color="white", linestyle="--", linewidth=1, alpha=0.5, label="50% baseline")
    ax7.fill_between(test_df2_grp.index, 0.5, test_df2_grp.values,
                     where=(test_df2_grp.values > 0.5), alpha=0.2, color="#55A868")
    ax7.fill_between(test_df2_grp.index, 0.5, test_df2_grp.values,
                     where=(test_df2_grp.values < 0.5), alpha=0.2, color="#C44E52")
    ax7.legend(facecolor="#1c2128", labelcolor="white", fontsize=8)
    style_ax(ax7, f"Rolling Accuracy Over Time ({best_name})", "Date", "Accuracy (50-sample rolling)")
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)

    # ── 8. Signal Distribution by Ticker ──
    ax8 = fig.add_subplot(gs[3, :])
    test_df2["signal"]  = results[best_name]["y_proba"]
    ticker_signal = test_df2.groupby("Name")["signal"].mean().sort_values()
    colors = ["#C44E52" if v < 0.5 else "#55A868" for v in ticker_signal.values]
    bars8 = ax8.bar(ticker_signal.index, ticker_signal.values, color=colors, alpha=0.85, edgecolor="#30363d")
    ax8.axhline(0.5, color="white", linestyle="--", linewidth=1.2)
    for bar, val in zip(bars8, ticker_signal.values):
        ax8.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", color="white", fontsize=9)
    style_ax(ax8, "Average Bull Signal Strength by Ticker (>0.5 = Bullish Bias)", "Ticker", "Mean P(Up)")
    ax8.set_ylim(0.3, 0.75)

    plt.savefig("outputs/trendcast_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor="#0d1117")
    print("📊 Dashboard saved → outputs/trendcast_dashboard.png")

# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir("/home/claude/TrendCast")
    from src.features import build_features
    from src.generate_data import generate_dataset

    print("📦 Generating dataset...")
    raw = generate_dataset()
    print("⚙️  Building features...")
    df  = build_features(raw)
    df.to_csv("data/features.csv", index=False)

    results, trained_models, scaler, X_test_sc, y_test, test_df, best_name = train_and_evaluate(df)
    plot_all(results, trained_models, scaler, X_test_sc, y_test, test_df, best_name)

    print("\n✅ Training complete! Files saved:")
    print("   models/   → trained model pickles")
    print("   outputs/  → dashboard PNG")
    print("   reports/  → metrics JSON")
