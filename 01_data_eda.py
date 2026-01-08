"""
Task 2: Exploratory Data Analysis (EDA)

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal, leak-safe feature engineering used in later tasks."""
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

    df = pd.read_csv(data_csv, dtype=dtype_map, low_memory=False)
    raw_shape = df.shape
    dup = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    shape = df.shape

    # optional: feature engineering (not required for EDA plots but aligns with pipeline)
    df_fe = engineer_features(df)

    class_counts = df_fe["Class"].value_counts().sort_index()
    fraud_rate = float(class_counts.get(1, 0)) / float(len(df_fe))

    summary_path = os.path.join(out_dir, "task2_eda_summary_full.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("EDA summary (full dataset)\n")
        f.write(f"Raw shape: {raw_shape}\n")
        f.write(f"After dropping duplicates: {shape} (removed {dup})\n")
        f.write("Missing values (non-zero):\n")
        missing = df_fe.isna().sum()
        missing_nz = missing[missing > 0]
        f.write("None\n\n" if missing_nz.empty else missing_nz.to_string() + "\n\n")
        f.write("Class distribution:\n")
        f.write(class_counts.to_string() + "\n")
        f.write(f"\nFraud rate: {fraud_rate:.6f} ({fraud_rate*100:.4f}%)\n")

    # plots
    plt.figure()
    class_counts.plot(kind="bar")
    plt.title("Class Distribution (Full dataset)")
    plt.xlabel("Class (0=Legit, 1=Fraud)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_plot_class_distribution_full.png"), dpi=220)
    plt.close()

    plt.figure()
    np.log1p(df_fe["Amount"]).hist(bins=80)
    plt.title("Histogram of log1p(Amount) (Full dataset)")
    plt.xlabel("log1p(Amount)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_plot_hist_amount_log1p_full.png"), dpi=220)
    plt.close()

    plt.figure()
    df_fe["Time"].hist(bins=80)
    plt.title("Histogram of Time (Full dataset)")
    plt.xlabel("Time (seconds from first transaction)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_plot_hist_time_full.png"), dpi=220)
    plt.close()

    plt.figure()
    df_fe.boxplot(column="Amount", by="Class")
    plt.title("Amount by Class (Boxplot) (Full dataset)")
    plt.suptitle("")
    plt.xlabel("Class")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_plot_box_amount_by_class_full.png"), dpi=220)
    plt.close()

    # correlation on sample for speed
    sample_n = min(100000, len(df_fe))
    corr = df_fe.sample(n=sample_n, random_state=42).corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, aspect="auto")
    plt.title(f"Correlation Heatmap (sample n={sample_n})")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(corr.index)), corr.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "task2_plot_corr_heatmap_sample100k.png"), dpi=260)
    plt.close()

    print("Saved EDA outputs to:", out_dir)


if __name__ == "__main__":
    main()
