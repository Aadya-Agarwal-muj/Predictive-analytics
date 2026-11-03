"""
06_evaluation.py - out-of-sample validation of the baselines and both GLMs.

Validation design (why it looks like this)
    * One row = one matched day (its 1961 and 1962 counts together). Splitting
      by ROW therefore keeps every matched pair intact: a day's 1961 count can
      never sit in training while its own 1962 count sits in the test set.
      We assert that train and test day codes never overlap.
    * The day code is documented only as a factor 1-92, so we do NOT treat it
      as a calendar ("train on Jan-Oct, test on Nov"). But EDA (figure 07 and
      eda_day_code_autocorrelation.csv) shows (i) days arrive in contiguous
      blocks with the same speed-limit regime and (ii) a strong lag-7
      autocorrelation - a weekly cycle - so consecutive codes are very likely
      consecutive days and neighbours share unrecorded conditions. A random
      split could therefore be slightly optimistic. We report BOTH:
        A. Stratified 80/20 hold-out split (simple, syllabus "train-test split").
        B. Blocked 5-fold CV over contiguous day-code ranges  (PRIMARY result;
           conservative - whole neighbourhoods are held out together).
        C. Repeated stratified 5-fold CV, 20 repeats (stability check: 100
           different test sets, report mean and spread).
    * With 92 rows a single test set has ~19 days, so one split is noisy -
      that is why CV is the primary evidence.

Compared predictors
    Mean baseline       : predict the training-set mean of y2 for every day.
    Previous-year       : predict y2 = y1 (same day last year).
    Poisson GLM         : y1 + lim1 + lim2, log link.
    NegBin (NB2)        : same predictors, extra dispersion parameter.

Run: python scripts/06_evaluation.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, train_test_split  # noqa: E402

from src.config import (  # noqa: E402
    FEATURE_SETS,
    FEATURES,
    FIG_DIR,
    METRICS_DIR,
    MODEL_DIR,
    RANDOM_STATE,
    TARGET,
    ensure_output_dirs,
)
from src.modeling import (  # noqa: E402
    design_matrix,
    fit_negbin,
    fit_poisson,
    load_processed_data,
    predictive_interval,
    regression_metrics,
)

MODEL_ORDER = ["Mean baseline", "Previous-year baseline", "Poisson GLM", "NegBin (NB2)"]


def predict_all(train: pd.DataFrame, test: pd.DataFrame, features=FEATURES,
                check_overlap: bool = True) -> dict:
    """Fit every model on `train` and return predicted means for `test`."""
    if check_overlap:
        assert set(train["day"]).isdisjoint(test["day"]), "train/test contamination!"
    pois = fit_poisson(train, features)
    nb_glm, alpha_hat, _ = fit_negbin(train, features)
    X_test = design_matrix(test, features)
    return {
        "Mean baseline": np.full(len(test), train[TARGET].mean()),
        "Previous-year baseline": test["y1"].to_numpy(dtype=float),
        "Poisson GLM": pois.predict(X_test).to_numpy(),
        "NegBin (NB2)": nb_glm.predict(X_test).to_numpy(),
        "_alpha": alpha_hat,
    }


def strata(df: pd.DataFrame) -> pd.Series:
    """Regime code 0-3 used for stratification (keeps regime mix in each fold)."""
    return (df["lim1"] * 2 + df["lim2"]).astype(int)


def run_cv(df, splitter, split_args, scheme_name, features=FEATURES, keep_oof=False):
    """Generic cross-validation loop returning per-fold metrics (+ coverage)."""
    rows, oof = [], []
    for fold, (tr, te) in enumerate(splitter.split(df, *split_args), start=1):
        train, test = df.iloc[tr], df.iloc[te]
        preds = predict_all(train, test, features)
        alpha_hat = preds.pop("_alpha")
        y = test[TARGET].to_numpy()
        for model, mu in preds.items():
            row = {"scheme": scheme_name, "fold": fold, "model": model, "n_test": len(te)}
            row.update(regression_metrics(y, mu))
            if model in ("Poisson GLM", "NegBin (NB2)"):
                kind = "poisson" if model == "Poisson GLM" else "negbin"
                lo, hi = predictive_interval(mu, kind, alpha_hat, level=0.90)
                row["90% PI coverage"] = float(np.mean((y >= lo) & (y <= hi)))
                row["90% PI mean width"] = float(np.mean(hi - lo))
            rows.append(row)
        if keep_oof:
            part = test[["day", "regime", "y1", "y2"]].copy()
            part["fold"] = fold
            for model, mu in preds.items():
                part[model] = mu
            part["negbin_alpha_fold"] = alpha_hat
            oof.append(part)
    out = pd.DataFrame(rows)
    return (out, pd.concat(oof)) if keep_oof else out


def summarise(cv: pd.DataFrame) -> pd.DataFrame:
    metrics = ["MAE", "RMSE", "Mean Poisson deviance"]
    s = cv.groupby(["scheme", "model"])[metrics].agg(["mean", "std"])
    s.columns = [f"{m} {stat}" for m, stat in s.columns]
    return s.reset_index()


def main() -> None:
    ensure_output_dirs()
    df = load_processed_data().sort_values("day").reset_index(drop=True)

    # ======================================================= A. hold-out split
    train, test = train_test_split(df, test_size=0.2, stratify=strata(df),
                                   random_state=RANDOM_STATE)
    print(f"A. Hold-out split: {len(train)} training days, {len(test)} test days "
          "(stratified by speed-limit regime, seed 42)")
    test_preds = predict_all(train, test)
    # In-sample predictions (train scored on itself) - compared with test
    # scores to check for overfitting.
    train_preds = predict_all(train, train, check_overlap=False)
    alpha_holdout = test_preds.pop("_alpha")
    train_preds.pop("_alpha")
    rows = []
    for model in MODEL_ORDER:
        for split, preds, data in [("train", train_preds, train), ("test", test_preds, test)]:
            r = {"split": split, "model": model}
            r.update(regression_metrics(data[TARGET], preds[model]))
            rows.append(r)
    holdout = pd.DataFrame(rows)
    holdout.to_csv(METRICS_DIR / "holdout_metrics.csv", index=False)
    print(holdout.round(3).to_string(index=False))

    hold_pred = test[["day", "regime", "y1", "y2"]].copy()
    for model in MODEL_ORDER:
        hold_pred[model] = test_preds[model]
    lo, hi = predictive_interval(hold_pred["NegBin (NB2)"], "negbin", alpha_holdout)
    hold_pred["NB 90% PI low"], hold_pred["NB 90% PI high"] = lo, hi
    hold_pred.to_csv(MODEL_DIR / "holdout_predictions.csv", index=False)

    # =================================================== B. blocked 5-fold CV
    blocked, oof = run_cv(df, KFold(n_splits=5, shuffle=False), (), "Blocked 5-fold",
                          keep_oof=True)
    oof.to_csv(MODEL_DIR / "oof_predictions_blocked_cv.csv", index=False)
    for fold, part in oof.groupby("fold"):
        print(f"  blocked fold {fold}: test day codes {part['day'].min()}-{part['day'].max()}")

    # ===================================== C. repeated stratified 5-fold CV
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=RANDOM_STATE)
    repeated = run_cv(df, rskf, (strata(df),), "Repeated stratified 5-fold x20")

    cv_all = pd.concat([blocked, repeated], ignore_index=True)
    cv_all.to_csv(METRICS_DIR / "cv_fold_metrics.csv", index=False)
    summary = summarise(cv_all)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    summary = summary.sort_values(["scheme", "model"])
    summary.to_csv(METRICS_DIR / "cv_summary.csv", index=False)
    print("\nB/C. Cross-validation summary (mean and std over folds):")
    print(summary.round(3).to_string(index=False))

    coverage = (cv_all.dropna(subset=["90% PI coverage"])
                .groupby(["scheme", "model"])[["90% PI coverage", "90% PI mean width"]]
                .mean().reset_index())
    coverage.to_csv(METRICS_DIR / "interval_coverage.csv", index=False)
    print("\n90% predictive-interval coverage (target = 0.90):")
    print(coverage.round(3).to_string(index=False))

    # ====================== D. feature-set selection (grid over specifications)
    fs_rows = []
    for name, feats in FEATURE_SETS.items():
        cv_fs = run_cv(df, KFold(n_splits=5, shuffle=False), (), "Blocked 5-fold", feats)
        cv_fs = cv_fs[cv_fs["model"].isin(["Poisson GLM", "NegBin (NB2)"])]
        agg = cv_fs.groupby("model")[["MAE", "RMSE", "Mean Poisson deviance"]].mean()
        for model, r in agg.iterrows():
            fs_rows.append({"feature_set": name, "model": model, **r.to_dict()})
    feature_sel = pd.DataFrame(fs_rows).sort_values(["model", "Mean Poisson deviance"])
    feature_sel.to_csv(METRICS_DIR / "feature_selection_cv.csv", index=False)
    print("\nD. Feature-set comparison (blocked 5-fold CV):")
    print(feature_sel.round(3).to_string(index=False))

    # ================================================================ figures
    blocked_means = summary[summary["scheme"] == "Blocked 5-fold"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colours = ["#BBBBBB", "#999999", "#4C72B0", "#C44E52"]
    for ax, metric in zip(axes, ["MAE", "RMSE", "Mean Poisson deviance"]):
        ax.bar(range(4), blocked_means[f"{metric} mean"], yerr=blocked_means[f"{metric} std"],
               color=colours, capsize=4)
        ax.set_xticks(range(4), ["Mean", "Prev-year", "Poisson", "NB2"])
        ax.set_title(f"{metric} (blocked 5-fold CV)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_cv_model_comparison.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(oof["y2"], oof["Poisson GLM"], alpha=0.7, label="Poisson")
    ax.scatter(oof["y2"], oof["NegBin (NB2)"], alpha=0.7, marker="x", label="NB2")
    ax.plot([5, 50], [5, 50], color="grey", linestyle=":")
    ax.set_xlabel("Actual 1962 accidents")
    ax.set_ylabel("Predicted (out-of-fold)")
    ax.set_title("Actual vs predicted, blocked CV (each day predicted once)")
    ax.legend()
    ax = axes[1]
    resid = oof["y2"] - oof["NegBin (NB2)"]
    ax.scatter(oof["NegBin (NB2)"], resid, alpha=0.7, color="#C44E52")
    ax.axhline(0, color="grey")
    ax.set_xlabel("Predicted mean (NB2)")
    ax.set_ylabel("Actual - predicted")
    ax.set_title("Out-of-fold residuals (NB2)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_actual_vs_predicted_oof.png", dpi=150)
    plt.close(fig)
    print("\nsaved figures 10_cv_model_comparison.png, 11_actual_vs_predicted_oof.png")


if __name__ == "__main__":
    main()
