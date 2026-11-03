"""
05_modeling.py - fit Poisson and Negative Binomial (NB2) regressions on all
92 days for INFERENCE (coefficients, IRRs, overdispersion tests).

Predictive accuracy is NOT judged here - fitting and scoring on the same rows
is optimistic. Out-of-sample evaluation is done in 06_evaluation.py.

Model:  log(E[y2 | X]) = b0 + b1*y1 + b2*lim1 + b3*lim2

Run: python scripts/05_modeling.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.config import FEATURES, FIG_DIR, METRICS_DIR, MODEL_DIR, TARGET, ensure_output_dirs  # noqa: E402
from src.modeling import (  # noqa: E402
    coefficient_table,
    fit_negbin,
    fit_poisson,
    load_processed_data,
    overdispersion_tests,
)


def main() -> None:
    ensure_output_dirs()
    df = load_processed_data()

    # ------------------------------------------------------------ Poisson
    pois = fit_poisson(df)
    print(pois.summary())
    (MODEL_DIR / "poisson_summary.txt").write_text(str(pois.summary()))

    # ------------------------------------------------------- Negative Binomial
    nb_glm, alpha_hat, nb_discrete = fit_negbin(df)
    print(nb_discrete.summary())
    (MODEL_DIR / "negbin_discrete_summary.txt").write_text(str(nb_discrete.summary()))
    (MODEL_DIR / "negbin_glm_summary.txt").write_text(str(nb_glm.summary()))
    print(f"\nEstimated NB2 dispersion alpha = {alpha_hat:.4f}")
    print(f"At the average count ({df[TARGET].mean():.1f}) NB2 variance = mu + alpha*mu^2 = "
          f"{df[TARGET].mean() + alpha_hat * df[TARGET].mean() ** 2:.1f} "
          f"vs Poisson variance = {df[TARGET].mean():.1f}")

    # ------------------------------------------------------ coefficient tables
    coefs = pd.concat([coefficient_table(pois, "Poisson"),
                       coefficient_table(nb_glm, "NegBin (NB2)")], ignore_index=True)
    coefs.to_csv(MODEL_DIR / "coefficients.csv", index=False)
    print("\nCoefficients and incidence-rate ratios (IRR = exp(coef)):")
    print(coefs[["model", "term", "coef", "std_err", "p_value", "IRR",
                 "IRR_ci_low", "IRR_ci_high", "pct_change"]].round(4).to_string(index=False))

    # ----------------------------------------------------- overdispersion
    tests = overdispersion_tests(df)
    pd.Series(tests, name="value").to_csv(METRICS_DIR / "overdispersion_tests.csv")
    print("\nOverdispersion diagnostics:")
    for k, v in tests.items():
        print(f"  {k:<36} {v:.4f}")

    # ---------------------------------------------------- fitted values
    fitted = df[["day", "lim1", "lim2", "y1", "y2"]].copy()
    fitted["poisson_fitted"] = pois.fittedvalues
    fitted["negbin_fitted"] = nb_glm.fittedvalues
    fitted["poisson_pearson_resid"] = pois.resid_pearson
    fitted["negbin_pearson_resid"] = nb_glm.resid_pearson
    fitted["poisson_deviance_resid"] = pois.resid_deviance
    fitted["negbin_deviance_resid"] = nb_glm.resid_deviance
    fitted.to_csv(MODEL_DIR / "fitted_values_full_data.csv", index=False)

    # --------------------------------------------- residual diagnostics plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, (name, res) in enumerate([("Poisson", pois), ("NegBin (NB2)", nb_glm)]):
        ax = axes[row, 0]
        ax.scatter(res.fittedvalues, res.resid_pearson, alpha=0.75)
        ax.axhline(0, color="grey")
        for b in (-2, 2):
            ax.axhline(b, color="red", linestyle=":")
        ax.set_xlabel("Fitted mean")
        ax.set_ylabel("Pearson residual")
        share = (abs(res.resid_pearson) > 2).mean() * 100
        ax.set_title(f"{name}: Pearson residuals ({share:.0f}% beyond +/-2)")
        ax = axes[row, 1]
        stats.probplot(res.resid_deviance, dist="norm", plot=ax)
        ax.set_title(f"{name}: Q-Q plot of deviance residuals")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_residual_diagnostics_full_fit.png", dpi=150)
    plt.close(fig)
    print("\nsaved figure 09_residual_diagnostics_full_fit.png")
    print(f"Features used: {FEATURES}; target: {TARGET}")


if __name__ == "__main__":
    main()
