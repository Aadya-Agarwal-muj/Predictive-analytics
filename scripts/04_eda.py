"""
04_eda.py - exploratory data analysis: each chart answers one question.

Q1  What do the 1961 and 1962 count distributions look like?    -> 02
Q2  Is the 1961 count of a day related to its 1962 count?        -> 03
Q3  How do counts differ across speed-limit regimes?             -> 04
Q4  How did each matched day change from 1961 to 1962?           -> 05
Q5  Which variables are linearly related?                        -> 06
Q6  What does the `day` code look like (matters for validation)? -> 07
Q7  Does variance grow with the mean (overdispersion hint)?      -> 08

Run: python scripts/04_eda.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import FIG_DIR, METRICS_DIR, ensure_output_dirs  # noqa: E402
from src.modeling import load_processed_data  # noqa: E402

REGIME_ORDER = ["No limit either year", "Limit in 1962 only",
                "Limit in 1961 only", "Limit both years"]
sns.set_theme(style="whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    print(f"saved figure {name}")


def main() -> None:
    ensure_output_dirs()
    df = load_processed_data()

    # Q1 - distributions
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    bins = np.arange(5, 52, 3)
    for ax, col, year, colour in zip(axes, ["y1", "y2"], [1961, 1962], ["#4C72B0", "#DD8452"]):
        ax.hist(df[col], bins=bins, color=colour, edgecolor="white")
        ax.axvline(df[col].mean(), color="black", linestyle="--", label=f"mean = {df[col].mean():.2f}")
        ax.set_title(f"{year}: accidents per day (var = {df[col].var():.1f})")
        ax.set_xlabel("Accident count")
        ax.legend()
    axes[0].set_ylabel("Number of days")
    save(fig, "02_count_distributions.png")

    # Q2 - matched-year relationship
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lim2, colour, label in [(0, "#4C72B0", "No limit in 1962"), (1, "#C44E52", "Limit in 1962")]:
        sub = df[df["lim2"] == lim2]
        ax.scatter(sub["y1"], sub["y2"], color=colour, alpha=0.8, label=label)
    lims = [0, 52]
    ax.plot(lims, lims, color="grey", linestyle=":", label="y2 = y1 (no change)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("1961 accidents on the day (y1)")
    ax.set_ylabel("1962 accidents on matched day (y2)")
    ax.set_title(f"Matched days: Pearson r = {df['y1'].corr(df['y2']):.3f}")
    ax.legend()
    save(fig, "03_y1_vs_y2.png")

    # Q3 - counts by regime
    long = df.melt(id_vars=["day", "regime"], value_vars=["y1", "y2"],
                   var_name="year", value_name="accidents")
    long["year"] = long["year"].map({"y1": "1961", "y2": "1962"})
    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.boxplot(data=long, x="regime", y="accidents", hue="year", order=REGIME_ORDER, ax=ax)
    ax.set_xlabel("")
    ax.set_title("Accident counts by speed-limit regime")
    save(fig, "04_counts_by_regime.png")

    # Q4 - paired change
    df["change"] = df["y2"] - df["y1"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.stripplot(data=df, x="regime", y="change", order=REGIME_ORDER, ax=ax,
                  color="#55A868", size=6, jitter=0.2)
    means = df.groupby("regime")["change"].mean().reindex(REGIME_ORDER)
    ax.scatter(range(4), means.values, marker="_", s=900, color="black", label="mean change")
    ax.axhline(0, color="grey", linestyle=":")
    ax.set_xlabel("")
    ax.set_ylabel("y2 - y1 (change on matched day)")
    ax.set_title("Within-day change from 1961 to 1962")
    ax.legend()
    save(fig, "05_change_by_regime.png")

    # Q5 - correlation heatmap
    corr = df[["y1", "y2", "lim1", "lim2"]].corr()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Pearson correlation matrix")
    save(fig, "06_correlation_heatmap.png")

    # Q6 - structure of the `day` code
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["day"], df["y1"], marker="o", ms=3, label="1961 (y1)")
    ax.plot(df["day"], df["y2"], marker="o", ms=3, label="1962 (y2)")
    blocks = (df["lim2"] != df["lim2"].shift()) | (df["lim1"] != df["lim1"].shift())
    for d in df.loc[blocks, "day"]:
        ax.axvline(d - 0.5, color="grey", alpha=0.3)
    for _, r in df[df["lim2"] == 1].iterrows():
        ax.axvspan(r["day"] - 0.5, r["day"] + 0.5, color="#C44E52", alpha=0.08)
    ax.set_xlabel("day code (1-92)")
    ax.set_ylabel("Accidents")
    ax.set_title("Counts by day code; shaded = limit in 1962; vertical lines = regime changes")
    ax.legend()
    save(fig, "07_day_code_structure.png")

    # Q7 - mean vs variance within regimes
    grp = df.groupby("regime")["y2"].agg(["count", "mean", "var"]).reindex(REGIME_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(grp["mean"], grp["var"], s=grp["count"] * 8, color="#8172B3")
    for name, r in grp.iterrows():
        ax.annotate(f"{name}\n(n={int(r['count'])})", (r["mean"], r["var"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    line = np.linspace(10, 30, 10)
    ax.plot(line, line, linestyle=":", color="grey", label="variance = mean (Poisson)")
    ax.set_xlim(10, 30)
    ax.set_ylim(0, 140)
    ax.set_xlabel("Mean of y2 in regime")
    ax.set_ylabel("Variance of y2 in regime")
    ax.set_title("Variance vs mean of 1962 counts")
    ax.legend()
    save(fig, "08_mean_variance.png")

    # Tabular evidence used in the report/dashboard
    regime_table = df.groupby("regime").agg(
        days=("day", "count"), mean_y1=("y1", "mean"), mean_y2=("y2", "mean"),
        var_y2=("y2", "var"), mean_change=("change", "mean"),
    ).reindex(REGIME_ORDER)
    regime_table["ratio_y2_over_y1"] = regime_table["mean_y2"] / regime_table["mean_y1"]
    regime_table.to_csv(METRICS_DIR / "eda_regime_summary.csv")
    corr.to_csv(METRICS_DIR / "eda_correlations.csv")

    # Autocorrelation along the day code: if the code were an arbitrary label,
    # these would all be close to 0. A spike at lag 7 indicates a weekly cycle,
    # i.e. evidence that consecutive codes are consecutive days.
    acf = pd.DataFrame({
        "lag": range(1, 15),
        "y1_autocorr": [df["y1"].autocorr(lag) for lag in range(1, 15)],
        "y2_autocorr": [df["y2"].autocorr(lag) for lag in range(1, 15)],
    })
    acf.to_csv(METRICS_DIR / "eda_day_code_autocorrelation.csv", index=False)
    print("\nAutocorrelation of counts along the day code:")
    print(acf.round(3).to_string(index=False))

    runs = int(blocks.sum())
    print("\nRegime summary:")
    print(regime_table.round(3).to_string())
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())
    print(f"\nThe (lim1, lim2) pattern changes {runs - 1} times along the day code,"
          f" i.e. days come in {runs} contiguous blocks -> neighbouring day codes"
          " share conditions. This shapes the validation design in script 06.")


if __name__ == "__main__":
    main()
