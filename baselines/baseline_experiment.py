import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def generate_market_data(n_days: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generates realistic synthetic daily OHLCV return series with regime shifts."""
    rng = np.random.default_rng(seed)
    regimes = rng.choice([0, 1], size=n_days, p=[0.7, 0.3])
    mu = np.where(regimes == 0, 0.0005, -0.0008)
    sigma = np.where(regimes == 0, 0.01, 0.024)

    returns = rng.normal(mu, sigma)
    price = 100 * np.exp(np.cumsum(returns))

    df = pd.DataFrame({"close": price, "ret_1d": returns})
    df["ret_5d"] = df["close"].pct_change(5)
    df["ret_20d"] = df["close"].pct_change(20)
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["vol_30d"] = df["ret_1d"].rolling(30).std()
    df["sma_ratio"] = df["close"] / df["close"].rolling(20).mean() - 1.0

    # Target: Next-day positive return direction (1 = Up, 0 = Down)
    df["target"] = (df["ret_1d"].shift(-1) > 0).astype(int)
    return df.dropna().reset_index(drop=True)


def run_experiment():
    df = generate_market_data()
    feature_cols = ["ret_1d", "ret_5d", "ret_20d", "vol_10d", "vol_30d", "sma_ratio"]
    X = df[feature_cols]
    y = df["target"]

    # Strict chronological train/validation split (no lookahead leakage)
    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.bar(["Directional Acc", "F1 Score"], [acc, f1], color=["#1f77b4", "#ff7f0e"])
    plt.ylim(0, 1.0)
    plt.title("Out-of-Sample Market Direction Metrics")
    plt.savefig("outputs/metrics_plot.png", bbox_inches="tight")
    plt.close()

    results = {"val_accuracy": round(acc, 4), "val_f1": round(f1, 4)}
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f)

    print(f"METRIC_VAL_ACCURACY: {acc:.4f}")


if __name__ == "__main__":
    run_experiment()
