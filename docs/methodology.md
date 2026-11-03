# Methodology and Results

Every number in this document comes from `python run_all.py`. The source files are in `outputs/metrics/` and `outputs/model_results/`.

## 1. Data acquisition
The CRAN source archive `SMPracticals_1.4-3.2.tar.gz` contains `SMPracticals/data/limits.rda` (839 bytes). That file is R's binary serialisation of a data frame. Script 01 lists the archive contents without unpacking anything. Script 02 extracts only that one file, decodes it with `pyreadr` and checks that the result has 92 rows and the columns `day, lim1, lim2, y1, y2`. As decoded, `day` is a factor, the `lim*` columns are floats and the `y*` columns are 32-bit integers.

## 2. Preprocessing (script 03)
| Check | Result | Action |
|---|---|---|
| Missing values | 0 | No imputation |
| Duplicated rows / day ids | 0 / 0 | None. Four days share identical lim/y values, which is legitimate. |
| `lim1`, `lim2` ∈ {0, 1} | Yes | Converted from float to int64 |
| Counts non-negative and whole | Yes; no zeros | Converted to int64 |
| IQR outliers (1.5 × IQR) | y1: day 61 (47). y2: days 40, 61, 68, 75 (above the 38.1 fence) | Kept: these are plausible busy days, not errors |
| Scaling | Not applied | Keeps coefficients in accident units; the log link already handles scale |

Descriptive statistics:

| | Mean | Median | Mode | Variance | SD | Min | Max |
|---|---|---|---|---|---|---|---|
| y1 | 22.76 | 21 | 18, 21 | 79.26 | 8.90 | 8 | 47 |
| y2 | 20.23 | 19 | 15, 16, 17 | 72.64 | 8.52 | 7 | 49 |

## 3. Exploratory findings (script 04)
* The correlation between y1 and y2 is 0.516. The 1961 count is the strongest single predictor.
* Ratio of mean y2 to mean y1 by regime:
  * No limit in either year: 1.10
  * Limit in 1962 only: 0.69
  * Limit in 1961 only: 1.15
  * Limit in both years: 1.09
* Days that received a limit in 1962 had higher 1961 counts (mean 26.0, compared with about 20 elsewhere). Part of their fall may therefore be regression to the mean. This is one reason no causal claim is made.
* In every regime the variance is well above the mean, so overdispersion is likely.
* Along the day code, the regime changes in 8 contiguous blocks, and the counts show strong lag-7 autocorrelation (0.47 / 0.58). Both point to consecutive days with a weekly cycle, which drives the validation design.

## 4. Models (script 05)
Both models use a log link: log μᵢ = β0 + β1·y1ᵢ + β2·lim1ᵢ + β3·lim2ᵢ.

* **Poisson GLM:** `statsmodels` `GLM(family=Poisson())`, fitted by IRLS. It assumes Var(Y|X) = μ.
* **NB2:** `statsmodels.discrete.NegativeBinomial(loglike_method="nb2")` estimates the β coefficients and α jointly by maximum likelihood. The model is then refitted as `GLM(family=NegativeBinomial(alpha=α̂))`, which gives deviance, residuals and prediction standard errors. The `GLM` NB family on its own would hold α fixed at 1, which would be an arbitrary choice. Var(Y|X) = μ + αμ².

| Term | Poisson coef (SE) | Poisson IRR | NB2 coef (SE) | NB2 IRR [95% CI] | NB2 p |
|---|---|---|---|---|---|
| const | 2.442 (0.070) | 11.50 | 2.458 (0.093) | 11.69 | <0.001 |
| y1 | 0.0298 (0.0026) | 1.030 | 0.0286 (0.0036) | 1.029 [1.022, 1.036] | <0.001 |
| lim1 | 0.112 (0.057) | 1.118 | 0.122 (0.077) | 1.130 [0.972, 1.313] | 0.112 |
| lim2 | −0.338 (0.052) | 0.714 | −0.317 (0.068) | 0.729 [0.637, 0.833] | <0.001 |

Interpretation of the NB2 model, holding the other inputs fixed:
* Each additional 1961 accident raises the expected 1962 count by 2.9%.
* A 1962 speed limit is associated with a 27% lower expected count.
* `lim1` is positive, which makes sense: a 1961 count recorded under a limit understates the day's usual level. Under Poisson `lim1` reaches p = 0.049, but under NB2 it is not significant. Poisson standard errors are too small when the data are overdispersed, and this is a concrete example.

### Overdispersion tests
| Diagnostic | Value |
|---|---|
| Pearson χ² / df (Poisson) | 157.84 / 88 = **1.79** |
| Deviance / df (Poisson) | 1.83 |
| Cameron–Trivedi auxiliary regression | α = 0.034, t = 3.24, one-sided p = 0.0008 |
| NB2 α (MLE) | 0.037 (SE 0.013) |
| LR test NB2 vs Poisson | 16.76, boundary-corrected p = 2 × 10⁻⁵ |
| AIC Poisson / NB2 | 607.8 / 593.0 |
| BIC Poisson / NB2 | 617.8 / 605.6 |

**Conclusion:** the overdispersion is moderate but statistically clear. At the mean count of 20.2, NB2 implies a variance of 35.4, while Poisson implies 20.2. Under Poisson, 15% of Pearson residuals fall outside ±2; under NB2 only 2% do. AIC and BIC are appropriate here because both models are fitted to the same 92 observations by maximum likelihood.

## 5. Validation (script 06)
* **Unit of splitting:** each row holds both years of one day, so splitting by row keeps every matched pair together. The code asserts that the training and test day codes never overlap.
* **Baselines:** (i) the training-set mean of y2; (ii) the previous year's count (ŷ = y1).
* **Schemes:**
  * **A.** A stratified 80/20 hold-out split (73/19 days, stratified by regime, seed 42).
  * **B.** Blocked 5-fold CV over contiguous day codes (1–19, 20–38, 39–56, 57–74, 75–92). This is the primary scheme: it holds out whole neighbourhoods, which respects the weekly and block structure.
  * **C.** Repeated stratified 5-fold CV with 20 repeats (100 test sets), used as a stability check.
* **Metrics:** MAE, RMSE, mean Poisson deviance, and coverage of the 90% predictive interval.

### Results
| Model | Blocked CV MAE | RMSE | Poisson dev. | Repeated CV MAE (± SD) |
|---|---|---|---|---|
| Mean baseline | 6.69 | 8.67 | 3.49 | 6.55 (± 1.23) |
| Previous-year baseline | 7.56 | 8.90 | 3.50 | 7.56 (± 0.99) |
| Poisson GLM | **4.87** | **6.12** | **1.90** | **4.96** (± 0.76) |
| NB2 | 4.90 | 6.16 | 1.91 | 4.98 (± 0.76) |

On the hold-out split, NB2 scored train MAE 4.73 and test MAE 4.73. Poisson scored 4.72 and 4.70. Train and test error are essentially the same, so there is no sign of overfitting. Both GLMs beat the baselines, so there is no sign of underfitting either. The test set has only 19 days, so a single split is noisy, and the CV results are the main evidence.

| 90% predictive interval | Coverage (blocked CV) | Mean width |
|---|---|---|
| Poisson | 0.76 | 14.5 accidents |
| NB2 | **0.88** | 19.2 accidents |

### Feature-set comparison (blocked CV, a grid search over model specifications)
| Features | Poisson MAE / RMSE / dev. | NB2 MAE / RMSE / dev. |
|---|---|---|
| y1 + lim1 + lim2 (main) | **4.87** / 6.12 / 1.90 | **4.90** / 6.16 / 1.91 |
| y1 + lim2 | 4.95 / 6.03 / 1.86 | 4.99 / 6.09 / 1.88 |
| log_y1 + lim1 + lim2 | 5.17 / 6.47 / 2.06 | 5.20 / 6.53 / 2.08 |
| lim1 + lim2 (no history) | 6.85 / 8.95 / 3.67 | 6.84 / 8.94 / 3.66 |
| y1 + lim1 + lim2 + weekly cycle | 4.95 / **5.95** / **1.80** | 4.97 / **5.98** / **1.81** |

* Removing `y1` roughly doubles the deviance, so historical frequency carries most of the predictive signal.
* The engineered `log(y1)` performs worse than raw `y1` and is not used.
* Adding a weekly-cycle indicator (day mod 7, inferred from EDA) improves RMSE and deviance slightly but worsens MAE, and it relies on an undocumented reading of `day`. The pre-specified main model is therefore kept, and the weekly cycle is reported as an extension.
* Model selection uses CV only and never the hold-out test set, so the test score is not biased by the choice.

## 6. Conclusion
Historical accident frequency and speed-limit conditions do predict 1962 accident counts. Both GLMs reduce the mean absolute error by about 27% compared with predicting the average. The data are overdispersed, so the **Negative Binomial (NB2) model is preferred**. Its point accuracy is the same as Poisson's, but its standard errors and prediction intervals are honest, with 88% coverage against a 90% target compared with 76% for Poisson. The `lim2` association (about −27%) is a predictive association from one non-randomised historical experiment and should not be read as a causal effect.
