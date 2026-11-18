import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def compute_rolling_shannon_entropy(
    series: pd.Series, window: int = 20, bins: int = 8
) -> pd.Series:
    """Computes rolling Shannon Entropy (bits) over binned log returns."""

    def _entropy(arr: np.ndarray) -> float:
        counts, _ = np.histogram(arr, bins=bins)
        probs = counts[counts > 0] / len(arr)
        return float(-np.sum(probs * np.log2(probs)))

    return series.rolling(window).apply(_entropy, raw=True)


def load_market_data() -> pd.DataFrame:
    """Loads market data and creates lagged features."""
    csv_path = "data/market_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0004, 0.011, size=1500)
        close = 100 * np.exp(np.cumsum(returns))
        df = pd.DataFrame(
            {
                "Open": close * (1 + rng.normal(0, 0.003, size=1500)),
                "High": close * (1 + np.abs(rng.normal(0, 0.006, size=1500))),
                "Low": close * (1 - np.abs(rng.normal(0, 0.006, size=1500))),
                "Close": close,
                "Volume": rng.integers(100000, 500000, size=1500),
            }
        )

    df["ret_1d"] = df["Close"].pct_change(1)
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["ret_5d"] = df["Close"].pct_change(5)
    df["ret_20d"] = df["Close"].pct_change(20)
    df["vol_10d"] = df["ret_1d"].rolling(10).std()
    df["sma_ratio"] = df["Close"] / df["Close"].rolling(20).mean() - 1.0
    df["hl_spread"] = (df["High"] - df["Low"]) / df["Close"]
    df["shannon_entropy_20d"] = compute_rolling_shannon_entropy(
        df["log_ret"], window=20, bins=8
    )
    df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

    base_features = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_10d",
        "sma_ratio",
        "hl_spread",
        "shannon_entropy_20d",
    ]

    # create lagged versions (lags 1 and 2)
    for col in base_features:
        for lag in (1, 2):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def run_experiment():
    df = load_market_data()

    base_features = [
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "vol_10d",
        "sma_ratio",
        "hl_spread",
        "shannon_entropy_20d",
    ]
    lagged_features = [
        f"{col}_lag{lag}" for col in base_features for lag in (1, 2)
    ]
    feature_cols = base_features + lagged_features

    X = df[feature_cols]
    y = df["target"]

    split_idx = int(len(df) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(4, 4))
    plt.bar(["Val Accuracy"], [acc], color="#1f77b4")
    plt.ylim(0, 1.0)
    plt.title("Out‑of‑Sample Accuracy")
    plt.savefig("outputs/metrics_plot.png", bbox_inches="tight")
    plt.close()

    results = {"val_accuracy": round(acc, 4)}
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f)

    print(f"METRIC_VAL_ACCURACY: {acc:.4f}")


if __name__ == "__main__":
    run_experiment()