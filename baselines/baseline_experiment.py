import json
import os
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def run_experiment():
    # Generate synthetic financial/tabular dataset
    X, y = make_classification(
        n_samples=2000,
        n_features=20,
        n_informative=12,
        n_redundant=4,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline Model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)

    os.makedirs("outputs", exist_ok=True)

    # Save diagnostic plot
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "F1 Score"], [acc, f1], color=["#1f77b4", "#ff7f0e"])
    plt.ylim(0, 1.0)
    plt.title("Validation Metrics")
    plt.savefig("outputs/metrics_plot.png")
    plt.close()

    # Save structured results
    results = {"val_accuracy": round(acc, 4), "val_f1": round(f1, 4)}
    with open("outputs/metrics.json", "w") as f:
        json.dump(results, f)

    print(f"METRIC_VAL_ACCURACY: {acc:.4f}")


if __name__ == "__main__":
    run_experiment()
