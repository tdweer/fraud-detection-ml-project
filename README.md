# Fraud Detection ML Project

This folder contains reproducible, well-commented Python code for the group coursework
(fraud detection using the `creditcard (1).csv` dataset).

## Structure
- `01_data_eda.py` : EDA + visualisations (Task 2)
- `02_preprocessing_imbalance.py` : preprocessing + imbalance strategy comparison (Task 3)
- `03_train_lr.py` : Logistic Regression training + tuning evidence (Task 4) and evaluation (Task 5)
- `04_train_xgb.py` : XGBoost training + tuning evidence (Task 4) and evaluation (Task 5)
- `05_compare_models.py` : combined ROC/PR curves + summary tables (Task 5)
- `run_all.py` : runs the whole pipeline end-to-end

## How to run
1. Place the dataset CSV in the project root (same level as this `code/` folder), named:
   `creditcard (1).csv`
2. Install requirements:
   `pip install -r requirements.txt`
3. Run:
   `python code/run_all.py`

Outputs are written to `outputs/`.

## Notes
- The dataset is extremely imbalanced (fraud ~= 0.17%). We therefore report PR-AUC (Average Precision)
  alongside ROC-AUC, precision, recall and F1.
- We use cost-sensitive learning (LR class_weight, XGBoost scale_pos_weight) and we compare SMOTE and
  undersampling in Task 3.
