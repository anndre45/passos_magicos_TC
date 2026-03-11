import pandas as pd
from pathlib import Path
from scipy.stats import ks_2samp

TRAIN_DATA = Path("data/processed/train_data.csv")
PROD_DATA = Path("logs/predictions.csv")


def detect_drift():

    if not TRAIN_DATA.exists() or not PROD_DATA.exists():
        print("Dados insuficientes para análise de drift")
        return

    train_df = pd.read_csv(TRAIN_DATA)
    prod_df = pd.read_csv(PROD_DATA)

    numeric_features = [
        "IAA",
        "IEG",
        "IPS",
        "IDA",
        "IPV",
        "IAN"
    ]

    drift_report = {}

    for feature in numeric_features:

        if feature not in train_df.columns:
            continue

        stat, p_value = ks_2samp(
            train_df[feature],
            prod_df[feature]
        )

        drift_report[feature] = {
            "p_value": float(p_value),
            "drift_detected": p_value < 0.05
        }

    return drift_report


if __name__ == "__main__":

    report = detect_drift()

    print("\nDrift Report")
    print("------------")

    for feature, result in report.items():

        status = "DRIFT DETECTED" if result["drift_detected"] else "OK"

        print(
            f"{feature}: {status} "
            f"(p-value={result['p_value']:.4f})"
        )