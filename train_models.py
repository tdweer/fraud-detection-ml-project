#!/usr/bin/env python3
"""Train and evaluate fraud detection models on the full dataset CSV.

"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)

import joblib
from xgboost import XGBClassifier

try:
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception:
    RandomUnderSampler = None
    ImbPipeline = None


DATA_CSV = '/mnt/data/creditcard (1).csv'
OUT_DIR = '/mnt/data/fraud_project/outputs'
RANDOM_STATE = 42
TEST_SIZE = 0.20

os.makedirs(OUT_DIR, exist_ok=True)


def plot_confusion(cm, title, out_path):
    plt.figure()
    plt.imshow(cm, aspect='auto')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks([0, 1], ['Legit(0)', 'Fraud(1)'])
    plt.yticks([0, 1], ['Legit(0)', 'Fraud(1)'])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(int(v)), ha='center', va='center')
    plt.tight_layout()
    plt.savefig(out_path, dpi=240)
    plt.close()


def cv_average_precision(model_builder, X, y, cv):
    """Return mean/std Average Precision over folds."""
    scores = []
    for tr_idx, va_idx in cv.split(X, y):
        m = model_builder()
        m.fit(X.iloc[tr_idx], y[tr_idx])
        p = m.predict_proba(X.iloc[va_idx])[:, 1]
        scores.append(average_precision_score(y[va_idx], p))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    # Load & de-duplicate (basic leakage mitigation)
    cols = pd.read_csv(DATA_CSV, nrows=0).columns.tolist()
    dtype_map = {c: 'float32' for c in cols if c != 'Class'}
    dtype_map['Class'] = 'int8'
    df = pd.read_csv(DATA_CSV, dtype=dtype_map, low_memory=False)
    raw_shape = df.shape
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    shape = df.shape

    # Feature engineering
    seconds_in_day = 24 * 60 * 60
    df['LogAmount'] = np.log1p(df['Amount'].astype('float32'))
    tod = (df['Time'] % seconds_in_day).astype('float32')
    df['TimeOfDay_sin'] = np.sin(2 * np.pi * tod / seconds_in_day).astype('float32')
    df['TimeOfDay_cos'] = np.cos(2 * np.pi * tod / seconds_in_day).astype('float32')
    df['TimeHours'] = (df['Time'] / 3600.0).astype('float32')

    y = df['Class'].astype(int).values
    X = df.drop(columns=['Class'])
    fraud_rate = float(y.mean())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Practical tuning subset: all fraud + 20k legit (keeps CV quick)
    train_df = X_train.copy()
    train_df['Class'] = y_train
    fraud_df = train_df[train_df['Class'] == 1]
    legit_df = train_df[train_df['Class'] == 0]
    n_legit_tune = min(20000, len(legit_df))
    tune_df = pd.concat([fraud_df, legit_df.sample(n=n_legit_tune, random_state=RANDOM_STATE)], axis=0)
    tune_df = tune_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    y_tune = tune_df['Class'].astype(int).values
    X_tune = tune_df.drop(columns=['Class'])

    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    # ----- Logistic Regression tuning (small grid) -----
    lr_candidates = []
    for C in [0.01, 0.1, 1.0]:
        for pen in ['l1', 'l2']:
            def make_lr(C=C, pen=pen):
                return Pipeline([
                    ('scaler', StandardScaler()),
                    ('lr', LogisticRegression(
                        solver='liblinear',
                        class_weight='balanced',
                        C=C,
                        penalty=pen,
                        max_iter=2000,
                    ))
                ])
            mean_ap, std_ap = cv_average_precision(make_lr, X_tune, y_tune, cv3)
            lr_candidates.append({'C': C, 'penalty': pen, 'mean_ap': mean_ap, 'std_ap': std_ap})

    lr_df = pd.DataFrame(lr_candidates).sort_values('mean_ap', ascending=False)
    lr_df.to_csv(os.path.join(OUT_DIR, 'task4_lr_cv_results.csv'), index=False)
    best_lr_params = lr_df.iloc[0].to_dict()

    best_lr = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            C=float(best_lr_params['C']),
            penalty=str(best_lr_params['penalty']),
            max_iter=2000,
        ))
    ])
    best_lr.fit(X_train, y_train)

    # Optional: Random undersampling + LR (Task 3 evidence)
    best_lr_under = None
    if RandomUnderSampler is not None and ImbPipeline is not None:
        best_lr_under = ImbPipeline([
            ('rus', RandomUnderSampler(random_state=RANDOM_STATE)),
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                solver='liblinear',
                class_weight=None,
                C=float(best_lr_params['C']),
                penalty=str(best_lr_params['penalty']),
                max_iter=2000,
            ))
        ])
        best_lr_under.fit(X_train, y_train)

    # ----- XGBoost tuning (manual small grid) -----
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = float(neg) / float(pos)

    xgb_grid = [
        {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.1},
        {'n_estimators': 300, 'max_depth': 3, 'learning_rate': 0.1},
        {'n_estimators': 500, 'max_depth': 4, 'learning_rate': 0.05},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
    ]

    xgb_rows = []
    for p in xgb_grid:
        def make_xgb(p=p):
            return XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                tree_method='hist',
                n_jobs=-1,
                random_state=RANDOM_STATE,
                scale_pos_weight=spw,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=1,
                reg_lambda=1.0,
                gamma=0.0,
                **p
            )
        mean_ap, std_ap = cv_average_precision(make_xgb, X_tune, y_tune, cv3)
        xgb_rows.append({**p, 'mean_ap': mean_ap, 'std_ap': std_ap})

    xgb_df = pd.DataFrame(xgb_rows).sort_values('mean_ap', ascending=False)
    xgb_df.to_csv(os.path.join(OUT_DIR, 'task4_xgb_cv_results.csv'), index=False)
    best_xgb_params = xgb_df.iloc[0].to_dict()

    best_xgb = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=spw,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_lambda=1.0,
        gamma=0.0,
        n_estimators=int(best_xgb_params['n_estimators']),
        max_depth=int(best_xgb_params['max_depth']),
        learning_rate=float(best_xgb_params['learning_rate']),
    )
    best_xgb.fit(X_train, y_train)

    # ----- Evaluation (test set) -----
    def eval_on_test(name, model):
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        return {
            'model': name,
            'precision@0.5': float(precision_score(y_test, pred, zero_division=0)),
            'recall@0.5': float(recall_score(y_test, pred, zero_division=0)),
            'f1@0.5': float(f1_score(y_test, pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, proba)),
            'pr_auc': float(average_precision_score(y_test, proba)),
            'proba': proba,
            'pred': pred,
        }

    res_lr = eval_on_test('LogisticRegression (balanced)', best_lr)
    res_xgb = eval_on_test('XGBoost (scale_pos_weight)', best_xgb)

    rows = [
        {k: v for k, v in res_lr.items() if k not in ('proba', 'pred')},
        {k: v for k, v in res_xgb.items() if k not in ('proba', 'pred')},
    ]

    res_lr_u = None
    if best_lr_under is not None:
        res_lr_u = eval_on_test('LogReg + RandomUnderSampler', best_lr_under)
        rows.append({k: v for k, v in res_lr_u.items() if k not in ('proba', 'pred')})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, 'task5_model_results.csv'), index=False)

    # Confusion matrices
    plot_confusion(confusion_matrix(y_test, res_lr['pred']), 'Confusion Matrix - LR (thr=0.5)', os.path.join(OUT_DIR, 'task5_cm_lr.png'))
    plot_confusion(confusion_matrix(y_test, res_xgb['pred']), 'Confusion Matrix - XGB (thr=0.5)', os.path.join(OUT_DIR, 'task5_cm_xgb.png'))

    # ROC curve comparison
    fpr_lr, tpr_lr, _ = roc_curve(y_test, res_lr['proba'])
    fpr_x, tpr_x, _ = roc_curve(y_test, res_xgb['proba'])
    plt.figure()
    plt.plot(fpr_lr, tpr_lr, label=f"LR AUC={res_lr['roc_auc']:.4f}")
    plt.plot(fpr_x, tpr_x, label=f"XGB AUC={res_xgb['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.title('ROC curve (test set)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'task5_roc_compare.png'), dpi=240)
    plt.close()

    # PR curve comparison
    p_lr, r_lr, _ = precision_recall_curve(y_test, res_lr['proba'])
    p_x, r_x, _ = precision_recall_curve(y_test, res_xgb['proba'])
    plt.figure()
    plt.plot(r_lr, p_lr, label=f"LR AP={res_lr['pr_auc']:.4f}")
    plt.plot(r_x, p_x, label=f"XGB AP={res_xgb['pr_auc']:.4f}")
    plt.title('Precision-Recall curve (test set)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'task5_pr_compare.png'), dpi=240)
    plt.close()

    # XGB feature importances
    imp = pd.Series(best_xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(OUT_DIR, 'task5_xgb_feature_importances.csv'), header=['importance'])
    top15 = imp.head(15).sort_values()
    plt.figure(figsize=(8, 6))
    plt.barh(top15.index, top15.values)
    plt.title('Top 15 feature importances (XGBoost)')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'task5_xgb_feature_importance_top15.png'), dpi=260)
    plt.close()

    # LR coefficients (after scaling)
    lr_model = best_lr.named_steps['lr']
    coef = pd.Series(lr_model.coef_.ravel(), index=X_train.columns)
    coef_df = pd.DataFrame({'feature': coef.index, 'coef': coef.values})
    coef_df['abs_coef'] = coef_df['coef'].abs()
    coef_df.sort_values('abs_coef', ascending=False).to_csv(os.path.join(OUT_DIR, 'task5_lr_coefficients.csv'), index=False)
    top_coef = coef_df.sort_values('abs_coef', ascending=False).head(15).sort_values('abs_coef')
    plt.figure(figsize=(8, 6))
    plt.barh(top_coef['feature'], top_coef['coef'])
    plt.title('Top 15 LR coefficients (by |coef|)')
    plt.xlabel('Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'task5_lr_top_coefficients.png'), dpi=260)
    plt.close()

    # Save models
    joblib.dump(best_lr, os.path.join(OUT_DIR, 'model_logreg.joblib'))
    joblib.dump(best_xgb, os.path.join(OUT_DIR, 'model_xgboost.joblib'))
    if best_lr_under is not None:
        joblib.dump(best_lr_under, os.path.join(OUT_DIR, 'model_logreg_undersample.joblib'))

    # Task 3/4 notes
    notes = []
    notes.append(f'Raw shape: {raw_shape}')
    notes.append(f'After dropping duplicates: {shape} (removed {dup_count})')
    notes.append(f'Fraud rate after de-duplication: {fraud_rate:.6f} ({fraud_rate*100:.4f}%)')
    notes.append('')
    notes.append('Feature engineering:')
    notes.append('- LogAmount = log1p(Amount) to reduce skew.')
    notes.append('- TimeOfDay_sin/cos = cyclic transform of Time (mod 24h) to capture periodicity.')
    notes.append('- TimeHours = Time/3600 for interpretability.')
    notes.append('')
    notes.append('Imbalance handling:')
    notes.append("- Logistic Regression: class_weight='balanced' (cost-sensitive learning).")
    notes.append('- XGBoost: scale_pos_weight = (#negative/#positive) on train split.')
    notes.append('- Optional RandomUnderSampler+LR to illustrate resampling effects.')
    with open(os.path.join(OUT_DIR, 'task3_task4_preprocessing_notes.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(notes))

    meta = {
        'data_csv': DATA_CSV,
        'raw_shape': raw_shape,
        'shape_after_dedup': shape,
        'duplicates_removed': dup_count,
        'fraud_rate': fraud_rate,
        'train_size': int(len(X_train)),
        'test_size': int(len(X_test)),
        'tuning_subset_size': int(len(X_tune)),
        'tuning_legit_sample': int(n_legit_tune),
        'tuning_fraud': int((y_tune == 1).sum()),
        'best_lr_params': {'C': float(best_lr_params['C']), 'penalty': str(best_lr_params['penalty'])},
        'best_xgb_params': {
            'n_estimators': int(best_xgb_params['n_estimators']),
            'max_depth': int(best_xgb_params['max_depth']),
            'learning_rate': float(best_xgb_params['learning_rate']),
        },
    }
    with open(os.path.join(OUT_DIR, 'run_metadata_task3_task5.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print('Done. Outputs in:', OUT_DIR)
    print(results_df)


if __name__ == '__main__':
    main()
