"""
modeling.py - reusable modelling functions shared by scripts 05/06 and the dashboard.

Contents
    load_processed_data()   read the cleaned CSV
    design_matrix()         build X (with intercept column) for chosen features
    fit_poisson()           Poisson GLM with log link
    fit_negbin()            NB2 model: alpha by maximum likelihood, then GLM refit
    mean_poisson_deviance() evaluation metric for count predictions
    regression_metrics()    MAE, RMSE, mean Poisson deviance
    predictive_interval()   interval for a *new count* (not just the mean)
    overdispersion_tests()  Pearson dispersion, Cameron-Trivedi test, LR test

Why statsmodels (and not scikit-learn) for the models?
    statsmodels reports standard errors, p-values, confidence intervals, AIC,
    deviance and Pearson chi-square - everything needed to *defend* a GLM.
    scikit-learn's PoissonRegressor gives only point predictions and has no
    Negative Binomial model at all.

Why two statsmodels Negative Binomial classes?
    * statsmodels.discrete.discrete_model.NegativeBinomial (loglike_method="nb2")
      estimates the dispersion parameter alpha by maximum likelihood together
      with the betas.
    * statsmodels GLM with families.NegativeBinomial(alpha=...) treats alpha as a
      FIXED number you must supply (default 1.0, which is usually wrong).
    We therefore estimate alpha with the discrete model and then refit the GLM
    with that alpha. Given the same alpha, both give identical betas, but the
    GLM object also gives deviance, Pearson residuals and get_prediction().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.tools import add_constant

from src.config import FEATURES, PROCESSED_CSV_PATH, TARGET


# --------------------------------------------------------------------- data
def load_processed_data(path=PROCESSED_CSV_PATH) -> pd.DataFrame:
    """Read the processed dataset written by scripts/03_preprocess.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python run_all.py` (or scripts 02 and 03) first."
        )
    return pd.read_csv(path)


def design_matrix(df: pd.DataFrame, features=FEATURES) -> pd.DataFrame:
    """
    Return the predictor matrix X with an intercept column called 'const'.

    has_constant="add" forces the intercept to be added even when a small
    subset happens to have a column that looks constant (e.g. all lim1 = 0 in
    a single test row), so train and test matrices always have the same columns.
    """
    features = list(features)
    X = df[[f for f in features if f != "day_mod7"]].astype(float)
    if "day_mod7" in features:
        # One-hot encode position in the 7-day cycle with FIXED levels 1-6
        # (level 0 is the reference), so train and test always get the same
        # columns even if a small test fold lacks some level.
        for k in range(1, 7):
            X[f"wk_{k}"] = (df["day"] % 7 == k).astype(float)
    return add_constant(X, has_constant="add")


# ------------------------------------------------------------------- models
def fit_poisson(df: pd.DataFrame, features=FEATURES):
    """Poisson GLM:  log(E[y2|X]) = b0 + b1*y1 + b2*lim1 + b3*lim2."""
    X = design_matrix(df, features)
    y = df[TARGET].astype(float)
    # family=Poisson() uses the canonical log link by default.
    return GLM(y, X, family=families.Poisson()).fit()


def fit_negbin(df: pd.DataFrame, features=FEATURES):
    """
    Negative Binomial (NB2):  Var(Y|X) = mu + alpha * mu^2.

    Returns (glm_result, alpha_hat, discrete_result).
    """
    X = design_matrix(df, features)
    y = df[TARGET].astype(float)
    # Step 1: maximum-likelihood estimate of betas AND alpha.
    discrete = NegativeBinomial(y, X, loglike_method="nb2").fit(disp=0, maxiter=200)
    alpha_hat = float(discrete.params["alpha"])
    # Step 2: GLM refit with alpha fixed at its MLE (gives deviance, residuals,
    # prediction standard errors).
    glm = GLM(y, X, family=families.NegativeBinomial(alpha=alpha_hat)).fit()
    return glm, alpha_hat, discrete


# ------------------------------------------------------------------ metrics
def mean_poisson_deviance(y_true, y_pred) -> float:
    """
    Mean Poisson deviance = (2/n) * sum[ y*log(y/mu) - (y - mu) ].

    It is the natural loss for count predictions: it penalises errors relative
    to the size of the expected count (an error of 5 when mu = 10 is worse than
    an error of 5 when mu = 40). y*log(y/mu) is defined as 0 when y = 0.
    """
    y = np.asarray(y_true, dtype=float)
    mu = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(y > 0, y * np.log(y / mu), 0.0)
    return float(np.mean(2.0 * (term - (y - mu))))


def regression_metrics(y_true, y_pred) -> dict:
    """MAE, RMSE and mean Poisson deviance for one set of predictions."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "Mean Poisson deviance": mean_poisson_deviance(y, p),
    }


# --------------------------------------------------------------- intervals
def predictive_interval(mu, model: str, alpha: float | None = None, level=0.90):
    """
    Interval that should contain a NEW observed count with probability `level`.

    Poisson:  Y ~ Poisson(mu)
    NB2:      Y ~ NegBin(n = 1/alpha, p = n / (n + mu))  -> mean mu,
              variance mu + alpha*mu^2  (scipy's parameterisation of NB2)
    """
    mu = np.asarray(mu, dtype=float)
    lower_q, upper_q = (1 - level) / 2, 1 - (1 - level) / 2
    if model == "poisson":
        dist = stats.poisson(mu)
    elif model == "negbin":
        n = 1.0 / alpha
        dist = stats.nbinom(n, n / (n + mu))
    else:
        raise ValueError("model must be 'poisson' or 'negbin'")
    return dist.ppf(lower_q), dist.ppf(upper_q)


# ---------------------------------------------------------- overdispersion
def overdispersion_tests(df: pd.DataFrame, features=FEATURES) -> dict:
    """
    Three complementary checks of the Poisson assumption Var(Y|X) = E(Y|X).

    1. Pearson dispersion = Pearson chi2 / residual df. About 1 under Poisson;
       clearly above 1 means overdispersion.
    2. Cameron & Trivedi (1990) auxiliary regression:
           ((y - mu)^2 - y) / mu = alpha * mu + error   (no intercept)
       A significantly positive alpha means variance grows faster than mean
       (the NB2 form).
    3. Likelihood-ratio test of H0: alpha = 0 (Poisson) vs alpha > 0 (NB2).
       alpha = 0 lies on the boundary of the parameter space, so the p-value is
       HALF of the usual chi-square(1) tail probability.
    """
    pois = fit_poisson(df, features)
    nb_glm, alpha_hat, nb_discrete = fit_negbin(df, features)
    y = df[TARGET].to_numpy(dtype=float)
    mu = pois.fittedvalues.to_numpy()

    pearson_disp = pois.pearson_chi2 / pois.df_resid
    deviance_disp = pois.deviance / pois.df_resid

    # Cameron-Trivedi auxiliary OLS without intercept (closed form).
    z = ((y - mu) ** 2 - y) / mu
    ct_alpha = float(np.sum(z * mu) / np.sum(mu * mu))
    resid = z - ct_alpha * mu
    se = float(np.sqrt(np.sum(resid**2) / (len(y) - 1) / np.sum(mu * mu)))
    ct_t = ct_alpha / se
    ct_p_one_sided = float(stats.t.sf(ct_t, df=len(y) - 1))

    lr_stat = float(2 * (nb_discrete.llf - pois.llf))
    lr_p = float(0.5 * stats.chi2.sf(lr_stat, df=1))

    return {
        "y2 mean": float(y.mean()),
        "y2 variance": float(y.var(ddof=1)),
        "Poisson Pearson chi2": float(pois.pearson_chi2),
        "Poisson residual df": float(pois.df_resid),
        "Pearson dispersion (chi2/df)": float(pearson_disp),
        "Deviance dispersion (dev/df)": float(deviance_disp),
        "Cameron-Trivedi alpha": ct_alpha,
        "Cameron-Trivedi t": float(ct_t),
        "Cameron-Trivedi p (one-sided)": ct_p_one_sided,
        "NB2 alpha (MLE)": alpha_hat,
        "LR statistic (NB2 vs Poisson)": lr_stat,
        "LR p-value (boundary-corrected)": lr_p,
        "Poisson AIC": float(pois.aic),
        "NB2 AIC": float(nb_discrete.aic),
        "Poisson BIC": float(pois.bic_llf),
        "NB2 BIC": float(nb_discrete.bic),
    }


def coefficient_table(result, model_name: str) -> pd.DataFrame:
    """Coefficients, standard errors, p-values, 95% CI and incidence-rate ratios."""
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "model": model_name,
            "term": result.params.index,
            "coef": result.params.values,
            "std_err": result.bse.values,
            "z": result.tvalues.values,
            "p_value": result.pvalues.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
        }
    )
    # IRR = exp(coef): multiplicative change in the expected count for a
    # one-unit increase in the predictor, holding other predictors fixed.
    table["IRR"] = np.exp(table["coef"])
    table["IRR_ci_low"] = np.exp(table["ci_low"])
    table["IRR_ci_high"] = np.exp(table["ci_high"])
    table["pct_change"] = (table["IRR"] - 1) * 100
    return table
