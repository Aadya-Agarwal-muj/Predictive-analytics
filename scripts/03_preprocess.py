"""
03_preprocess.py - data quality checks, descriptive statistics, outliers.

Stage: Data preprocessing (kept separate from modelling on purpose).

Principles we follow
    * The raw files in data/raw are never modified.
    * Zero is a valid count, never a "missing value".
    * Nothing is imputed unless a value is actually missing.
    * Outliers are FLAGGED and examined, not deleted: an unusually high
      accident day is a real event, and deleting it would make the model look
      better than it really is.
    * Counts are not standardised: the log link already handles scale, and
      keeping natural units keeps the coefficients interpretable.

Output: data/processed/swedish_speed_limit_92.csv
        outputs/metrics/data_quality_report.csv
        outputs/metrics/descriptive_statistics.csv
        outputs/metrics/iqr_outliers.csv
        outputs/figures/01_boxplots.png

Run   : python scripts/03_preprocess.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # draw to files, no window needed (works on any machine)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (  # noqa: E402
    BINARY_COLUMNS,
    COUNT_COLUMNS,
    FEATURES,
    FIG_DIR,
    METRICS_DIR,
    PROCESSED_CSV_PATH,
    RAW_CSV_PATH,
    TARGET,
    ensure_output_dirs,
)

REGIME_LABELS = {
    (0, 0): "No limit either year",
    (0, 1): "Limit in 1962 only",
    (1, 0): "Limit in 1961 only",
    (1, 1): "Limit both years",
}


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    ensure_output_dirs()
    raw = pd.read_csv(RAW_CSV_PATH)
    df = raw.copy()          # work on a copy; `raw` stays as loaded
    quality = []             # rows for data_quality_report.csv

    # ---------------------------------------------------------- 1. structure
    section("1. Dimensions, column names and data types (BEFORE conversion)")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(df.dtypes.to_string())
    quality.append(("Rows", df.shape[0]))
    quality.append(("Columns", df.shape[1]))

    # ------------------------------------------------------ 2. missing values
    section("2. Missing values")
    missing = df.isnull().sum()     # True for each NaN, summed per column
    print(missing.to_string())
    quality.append(("Missing values (total)", int(missing.sum())))
    if missing.sum() == 0:
        print("No missing values -> no imputation is needed or applied.")

    # ----------------------------------------------------------- 3. duplicates
    section("3. Duplicates")
    full_dups = int(df.duplicated().sum())
    day_dups = int(df["day"].duplicated().sum())
    value_dups = int(df.duplicated(subset=["lim1", "lim2", "y1", "y2"]).sum())
    print(f"Fully duplicated rows            : {full_dups}")
    print(f"Repeated day identifiers         : {day_dups}")
    print(f"Rows with identical values (no day): {value_dups}")
    print("Identical values on different days are legitimate (two days can have"
          " the same counts), so nothing is removed.")
    quality += [("Duplicated rows", full_dups), ("Duplicated day ids", day_dups),
                ("Rows with identical lim/y values", value_dups)]

    # ------------------------------------------------- 4. binary validation
    section("4. Binary-variable validation (lim1, lim2 must be 0 or 1)")
    for col in BINARY_COLUMNS:
        values = sorted(df[col].unique().tolist())
        ok = set(values) <= {0, 1}
        print(f"{col}: unique values {values} -> {'valid' if ok else 'INVALID'}")
        quality.append((f"{col} values only 0/1", ok))
        assert ok

    # --------------------------------------------- 5. count validation
    section("5. Count validation (non-negative whole numbers)")
    for col in COUNT_COLUMNS:
        non_negative = bool((df[col] >= 0).all())
        whole = bool(np.all(np.equal(np.mod(df[col], 1), 0)))
        zeros = int((df[col] == 0).sum())
        print(f"{col}: non-negative={non_negative}, whole numbers={whole}, zeros={zeros}")
        quality += [(f"{col} non-negative", non_negative), (f"{col} whole numbers", whole),
                    (f"{col} zero counts", zeros)]
        assert non_negative and whole

    # ------------------------------------------------ 6. type conversion
    section("6. Type conversion (an actual transformation: before -> after)")
    before = df.dtypes.astype(str)
    df["day"] = df["day"].astype("int64")
    for col in BINARY_COLUMNS + COUNT_COLUMNS:
        df[col] = df[col].astype("int64")
    after = df.dtypes.astype(str)
    print(pd.DataFrame({"before": before, "after": after}).to_string())
    print("Reason: lim1/lim2 arrive from R as floats (0.0/1.0); counts are whole"
          " numbers, so int64 states their true nature. Values are unchanged.")

    # ------------------------------------------ 7. descriptive statistics
    section("7. Descriptive statistics")
    stats_rows = {}
    for col in COUNT_COLUMNS:
        s = df[col]
        modes = s.mode().tolist()   # can be several values (multi-modal)
        stats_rows[col] = {
            "count": int(s.count()),
            "mean": s.mean(),
            "median": s.median(),
            "mode": ", ".join(map(str, modes)),
            "variance": s.var(),           # pandas uses ddof=1 (sample variance)
            "std_dev": s.std(),
            "min": s.min(),
            "Q1": s.quantile(0.25),
            "Q3": s.quantile(0.75),
            "max": s.max(),
            "variance/mean": s.var() / s.mean(),
        }
    for col in BINARY_COLUMNS:
        stats_rows[col] = {"count": int(df[col].count()), "mean": df[col].mean(),
                           "proportion of days with limit": df[col].mean()}
    desc = pd.DataFrame(stats_rows).T
    print(desc.to_string(float_format=lambda v: f"{v:.3f}"))
    print("\nvariance/mean > 1 for raw counts is a first hint of overdispersion,"
          " but the proper test uses the variance CONDITIONAL on predictors (script 05).")
    desc.to_csv(METRICS_DIR / "descriptive_statistics.csv")

    # ---------------------------------------------------- 8. IQR outliers
    section("8. Outlier detection with the IQR rule (flag, do not delete)")
    outlier_rows = []
    for col in COUNT_COLUMNS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        flagged = df[(df[col] < lower) | (df[col] > upper)]
        print(f"{col}: Q1={q1}, Q3={q3}, IQR={iqr}, fences=[{lower}, {upper}],"
              f" flagged days={flagged['day'].tolist()}")
        quality.append((f"{col} IQR outliers", len(flagged)))
        for _, row in flagged.iterrows():
            outlier_rows.append({"variable": col, "day": row["day"], "value": row[col],
                                 "lower_fence": lower, "upper_fence": upper})
    pd.DataFrame(outlier_rows).to_csv(METRICS_DIR / "iqr_outliers.csv", index=False)
    print("Decision: keep all flagged days. They are plausible high-accident days"
          " (no data-entry error is evident), and count models expect a long right tail.")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.boxplot([df["y1"], df["y2"]], tick_labels=["1961 (y1)", "1962 (y2)"], whis=1.5)
    ax.set_ylabel("Accidents per day")
    ax.set_title("Box plots of daily accident counts (whiskers = 1.5 x IQR)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_boxplots.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------- 9. derived columns
    section("9. Derived columns (justified transformations)")
    # log(y1): with a log link, log(mu) = b0 + b1*log(y1) means mu is
    # proportional to y1^b1 - an "elasticity" form. y1 >= 8, so log is safe.
    df["log_y1"] = np.log(df["y1"])
    # Human-readable regime label for charts (not used as a model input).
    df["regime"] = [REGIME_LABELS[(a, b)] for a, b in zip(df["lim1"], df["lim2"])]
    print(df[["day", "y1", "log_y1", "lim1", "lim2", "regime"]].head().to_string())
    print("\nRegime counts:")
    print(df["regime"].value_counts().to_string())
    print("\nNo scaling/standardisation is applied: GLM coefficients are then read"
          " directly in accident units (per extra 1961 accident).")

    # --------------------------------------- 10. feature/target separation
    section("10. Feature and target separation")
    X = df[FEATURES]
    y = df[TARGET]
    print(f"X (features) = {FEATURES}, shape {X.shape}")
    print(f"y (target)   = '{TARGET}', shape {y.shape}")
    print("`day` is excluded: it is an identifier code; using it would let the model"
          " learn position in the file rather than accident-related information.")

    # -------------------------------------------------------------- save
    df.to_csv(PROCESSED_CSV_PATH, index=False)
    pd.DataFrame(quality, columns=["check", "result"]).to_csv(
        METRICS_DIR / "data_quality_report.csv", index=False)
    print(f"\nSaved processed data -> {PROCESSED_CSV_PATH.name} {df.shape}")


if __name__ == "__main__":
    main()
