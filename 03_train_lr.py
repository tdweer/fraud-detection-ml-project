"""
Task 4-5: Logistic Regression model selection, training, and evaluation.

- Hyperparameter tuning (C, penalty) using 3-fold stratified CV on a tuning subset.
- Final training uses class_weight='balanced' (cost-sensitive learning).
- Outputs include confusion matrix, ROC curve, PR curve, and coefficient importance.

Note: this dataset is numeric (V1..V28, Time, Amount) so we standardise features and do not encode categories.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import joblib


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

    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build tuning subset: all fraud + 5,000 legit
    train_df = X_train.copy()
    train_df["Class"] = y_train
    fraud_df = train_df[train_df["Class"] == 1]
    legit_df = train_df[train_df["Class"] == 0].sample(n=min(5000, (y_train == 0).sum()), random_state=42)

    tune_df = pd.concat([fraud_df, legit_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_tune = tune_df.drop(columns=["Class"])
    y_tune = tune_df["Class"].astype(int).values

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rows = []
    for C in [0.01, 0.1, 1.0]:
        for pen in ["l1", "l2"]:
            scores = []
            for tr, va in cv.split(X_tune, y_tune):
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(
                        solver="liblinear",
                        class_weight="balanced",
                        C=C,
                        penalty=pen,
                        max_iter=2000
                    ))
                ])
                model.fit(X_tune.iloc[tr], y_tune[tr])
                p = model.predict_proba(X_tune.iloc[va])[:, 1]
                scores.append(average_precision_score(y_tune[va], p))
            rows.append({"C": C, "penalty": pen, "mean_ap": float(np.mean(scores)), "std_ap": float(np.std(scores))})

    lr_cv = pd.DataFrame(rows).sort_values("mean_ap", ascending=False)
    lr_cv.to_csv(os.path.join(out_dir, "task4_lr_cv_results.csv"), index=False)
    best = lr_cv.iloc[0].to_dict()

    final_lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            C=float(best["C"]),
            penalty=str(best["penalty"]),
            max_iter=2000
        ))
    ])

    # Final fit subset for speed: all fraud + 20k legit
    legit_fit = train_df[train_df["Class"] == 0].sample(n=min(20000, (y_train == 0).sum()), random_state=42)
    fit_df = pd.concat([fraud_df, legit_fit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    final_lr.fit(fit_df.drop(columns=["Class"]), fit_df["Class"].astype(int).values)

    joblib.dump(final_lr, os.path.join(out_dir, "model_logreg.joblib"))

    # Evaluate on full test
    proba = final_lr.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model": "Logistic Regression (balanced)",
        "precision@0.5": float(precision_score(y_test, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_test, pred, zero_division=0)),
        "f1@0.5": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, "task5_lr_results.csv"), index=False)

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix - LR (thr=0.5)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legit(0)", "Fraud(1)"])
    plt.yticks([0, 1], ["Legit(0)", "Fraud(1)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_cm_lr.png"), dpi=240)
    plt.close()

    # ROC / PR
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve - LR (test)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_roc_lr.png"), dpi=240)
    plt.close()

    p, r, _ = precision_recall_curve(y_test, proba)
    plt.figure()
    plt.plot(r, p)
    plt.title("PR curve - LR (test)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_pr_lr.png"), dpi=240)
    plt.close()

    # Coefficients
    lr = final_lr.named_steps["lr"]
    coef = pd.Series(lr.coef_.ravel(), index=fit_df.drop(columns=["Class"]).columns)
    coef_df = pd.DataFrame({"feature": coef.index, "coef": coef.values})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df.sort_values("abs_coef", ascending=False).to_csv(os.path.join(out_dir, "task5_lr_coefficients.csv"), index=False)

    top = coef_df.sort_values("abs_coef", ascending=False).head(15).sort_values("abs_coef")
    plt.figure(figsize=(8, 6))
    plt.barh(top["feature"], top["coef"])
    plt.title("Top 15 LR coefficients (by |coef|)")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_lr_top_coefficients.png"), dpi=260)
    plt.close()

    with open(os.path.join(out_dir, "meta_lr.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved LR outputs to:", out_dir)


if __name__ == "__main__":
    main()
