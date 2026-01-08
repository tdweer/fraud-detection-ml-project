"""
Task 4-5: XGBoost model selection, training, and evaluation.

"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
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

    # Tuning subset: all fraud + 5,000 legit
    train_df = X_train.copy()
    train_df["Class"] = y_train
    fraud_df = train_df[train_df["Class"] == 1]
    legit_df = train_df[train_df["Class"] == 0].sample(n=min(5000, (y_train == 0).sum()), random_state=42)

    tune_df = pd.concat([fraud_df, legit_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_tune = tune_df.drop(columns=["Class"])
    y_tune = tune_df["Class"].astype(int).values

    neg = int((y_tune == 0).sum())
    pos = int((y_tune == 1).sum())
    spw = float(neg) / float(pos)

    cv2 = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

    def cv_ap(params):
        scores = []
        for tr, va in cv2.split(X_tune, y_tune):
            m = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=2,
                random_state=42,
                scale_pos_weight=spw,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                reg_lambda=1.0,
                gamma=0.0,
                **params
            )
            m.fit(X_tune.iloc[tr], y_tune[tr])
            p = m.predict_proba(X_tune.iloc[va])[:, 1]
            scores.append(average_precision_score(y_tune[va], p))
        return float(np.mean(scores)), float(np.std(scores))

    grid = [
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05},
    ]

    rows = []
    for params in grid:
        mean_ap, std_ap = cv_ap(params)
        rows.append({**params, "mean_ap": mean_ap, "std_ap": std_ap})

    cv_df = pd.DataFrame(rows).sort_values("mean_ap", ascending=False)
    cv_df.to_csv(os.path.join(out_dir, "task4_xgb_cv_results.csv"), index=False)
    best = cv_df.iloc[0].to_dict()

    # Fit subset: all fraud + 60k legit
    legit_fit = train_df[train_df["Class"] == 0].sample(n=min(60000, (y_train == 0).sum()), random_state=42)
    fit_df = pd.concat([fraud_df, legit_fit]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    X_fit = fit_df.drop(columns=["Class"])
    y_fit = fit_df["Class"].astype(int).values

    neg_fit = int((y_fit == 0).sum())
    pos_fit = int((y_fit == 1).sum())
    spw_fit = float(neg_fit) / float(pos_fit)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=2,
        random_state=42,
        scale_pos_weight=spw_fit,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        gamma=0.0,
        n_estimators=int(best["n_estimators"]),
        max_depth=int(best["max_depth"]),
        learning_rate=float(best["learning_rate"]),
    )
    model.fit(X_fit, y_fit)

    joblib.dump(model, os.path.join(out_dir, "model_xgboost.joblib"))

    # Evaluate on full test
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "model": "XGBoost (scale_pos_weight)",
        "precision@0.5": float(precision_score(y_test, pred, zero_division=0)),
        "recall@0.5": float(recall_score(y_test, pred, zero_division=0)),
        "f1@0.5": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure()
    plt.imshow(cm, aspect="auto")
    plt.title("Confusion Matrix - XGB (thr=0.5)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legit(0)", "Fraud(1)"])
    plt.yticks([0, 1], ["Legit(0)", "Fraud(1)"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_cm_xgb.png"), dpi=240)
    plt.close()

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=X_fit.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(out_dir, "task5_xgb_feature_importances.csv"), header=["importance"])

    top15 = imp.head(15).sort_values()
    plt.figure(figsize=(8, 6))
    plt.barh(top15.index, top15.values)
    plt.title("Top 15 feature importances (XGBoost)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task5_xgb_feature_importance_top15.png"), dpi=260)
    plt.close()

    with open(os.path.join(out_dir, "meta_xgb.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved XGB outputs to:", out_dir)


if __name__ == "__main__":
    main()
