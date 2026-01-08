"""
Task 3: Data pre-processing + imbalance handling experiments.

"""

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    seconds_in_day = 24 * 60 * 60
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"].astype("float32"))
    tod = (df["Time"] % seconds_in_day).astype("float32")
    df["TimeOfDay_sin"] = np.sin(2 * np.pi * tod / seconds_in_day).astype("float32")
    df["TimeOfDay_cos"] = np.cos(2 * np.pi * tod / seconds_in_day).astype("float32")
    df["TimeHours"] = (df["Time"] / 3600.0).astype("float32")
    return df


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    data_csv = os.path.join(base_dir, "creditcard (1).csv")

    cols = pd.read_csv(data_csv, nrows=0).columns.tolist()
    dtype_map = {c: "float32" for c in cols if c != "Class"}
    dtype_map["Class"] = "int8"
    df = pd.read_csv(data_csv, dtype=dtype_map, low_memory=False).drop_duplicates().reset_index(drop=True)
    df = engineer_features(df)

    fraud = df[df["Class"] == 1]
    legit = df[df["Class"] == 0].sample(n=8000, random_state=42)
    exp = pd.concat([fraud, legit]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    X = exp.drop(columns=["Class"])
    y = exp["Class"].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    pipelines = {
        "LR_cost_sensitive": ImbPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="liblinear", class_weight="balanced", C=0.1, max_iter=2000)),
        ]),
        "LR_SMOTE": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42, k_neighbors=5)),
            ("clf", LogisticRegression(solver="liblinear", C=0.1, max_iter=2000)),
        ]),
        "LR_under_sample": ImbPipeline([
            ("scaler", StandardScaler()),
            ("under", RandomUnderSampler(random_state=42)),
            ("clf", LogisticRegression(solver="liblinear", C=0.1, max_iter=2000)),
        ]),
    }

    rows = []
    for name, pipe in pipelines.items():
        scores = []
        for tr, va in cv.split(X, y):
            pipe.fit(X.iloc[tr], y[tr])
            proba = pipe.predict_proba(X.iloc[va])[:, 1]
            scores.append(average_precision_score(y[va], proba))
        rows.append({
            "method": name,
            "mean_pr_auc": float(np.mean(scores)),
            "std_pr_auc": float(np.std(scores)),
        })

    res = pd.DataFrame(rows).sort_values("mean_pr_auc", ascending=False)
    res.to_csv(os.path.join(out_dir, "task3_imbalance_strategy_comparison.csv"), index=False)

    with open(os.path.join(out_dir, "task3_imbalance_strategy_notes.txt"), "w", encoding="utf-8") as f:
        f.write("Imbalance strategy comparison (3-fold CV on subset: all fraud + 8,000 legitimate)\n")
        f.write("Metric: Average Precision (PR-AUC)\n\n")
        f.write(res.to_string(index=False))
        f.write("\n")

    print("Saved Task 3 outputs to:", out_dir)


if __name__ == "__main__":
    main()
