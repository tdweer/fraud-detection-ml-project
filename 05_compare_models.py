"""
Task 5: Compare models on the same test set and generate combined ROC/PR plots.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

import numpy as np


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

    lr = joblib.load(os.path.join(out_dir, "model_logreg.joblib"))
    xgb = joblib.load(os.path.join(out_dir, "model_xgboost.joblib"))

    p_lr = lr.predict_proba(X_test)[:, 1]
    p_x = xgb.predict_proba(X_test)[:, 1]

    # ROC
    fpr_lr, tpr_lr, _ = roc_curve(y_test, p_lr)
    fpr_x, tpr_x, _ = roc_curve(y_test, p_x)

    auc_lr = roc_auc_score(y_test, p_lr)
    auc_x = roc_auc_score(y_test, p_x)

    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f"LR AUC={auc_lr:.4f}")
    plt.plot(fpr_x, tpr_x, label=f"XGB AUC={auc_x:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC curve comparison (test set)")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_roc_compare.png"), dpi=240)
    plt.close()

    # PR
    prec_lr, rec_lr, _ = precision_recall_curve(y_test, p_lr)
    prec_x, rec_x, _ = precision_recall_curve(y_test, p_x)
    ap_lr = average_precision_score(y_test, p_lr)
    ap_x = average_precision_score(y_test, p_x)

    plt.figure()
    plt.plot(rec_lr, prec_lr, label=f"LR AP={ap_lr:.4f}")
    plt.plot(rec_x, prec_x, label=f"XGB AP={ap_x:.4f}")
    plt.title("Precision-Recall curve comparison (test set)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_pr_compare.png"), dpi=240)
    plt.close()

    pd.DataFrame([
        {"model": "LR", "roc_auc": float(auc_lr), "pr_auc": float(ap_lr)},
        {"model": "XGB", "roc_auc": float(auc_x), "pr_auc": float(ap_x)},
    ]).to_csv(os.path.join(out_dir, "task5_compare_auc.csv"), index=False)

    print("Saved comparison plots to:", out_dir)


if __name__ == "__main__":
    main()
