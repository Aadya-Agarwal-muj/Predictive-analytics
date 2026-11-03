# Problem Statement

## Title
**Predictive Modelling of Traffic Accident Counts Using Poisson and Negative Binomial Regression: A Study of Historical Speed-Limit Conditions**

## Background
The number of road accidents on a given day is a count. It is a whole number, it can never be negative, and it is usually right-skewed. During 1961 and 1962, Sweden ran an experiment in which a general speed limit (90 or 100 km/h) was imposed on some days and not on others. Police-reported personal-injury accidents were recorded on 92 days in 1961 and on 92 matching days in 1962 (Svensson, 1981; distributed as dataset `limits` in the CRAN package SMPracticals).

## Problem
Given the accident count on a day in 1961, and whether a speed limit was in force on that day and on the matched day in 1962, **estimate the number of personal-injury accidents on the 1962 day**. The models are then checked against the counts that were actually recorded.

## Research question
*Can historical accident frequency and speed-limit conditions help predict the number of personal-injury traffic accidents recorded on corresponding motorway days?*

## Objectives
1. Acquire the original dataset from the official CRAN archive and document every variable.
2. Validate and preprocess the data in a separate, reproducible stage.
3. Explore distributions, matched-year relationships, speed-limit regimes and dispersion.
4. Fit a Poisson regression and a Negative Binomial (NB2) regression with a log link.
5. Test formally whether the data are overdispersed instead of assuming the answer.
6. Compare both models against a mean baseline and a previous-year baseline using held-out data and cross-validation (MAE, RMSE, Poisson deviance, interval coverage).
7. Present the data, the findings, the evaluation and a prediction sandbox in an interactive dashboard.

## Type of analytics
| Type | In this project |
|---|---|
| Descriptive | Summary statistics and charts of 1961 and 1962 counts. These describe the past. |
| **Predictive (focus)** | Models trained on some days predict the count for days they have never seen. The predictions are scored against actual records. |
| Prescriptive | Recommending speed-management actions. This is **out of scope**, because the data cannot support causal claims. |

## Why this is a count-regression problem
`y2` takes only the values 0, 1, 2, …. Ordinary linear regression can predict negative counts and assumes constant variance, but the variance of a count usually grows with its mean. Generalised linear models with a log link avoid both problems. The Poisson model assumes Var = Mean, while NB2 allows Var = μ + αμ².

## Assumptions
* Each row (one matched day) is treated as one unit of observation.
* The 1962 speed-limit status (`lim2`) is a planned policy condition known before the day happens, so it may legitimately be used as a predictor.
* Conditional on the predictors, the counts follow a Poisson or NB2 distribution. This assumption is checked in script 05.

## Limitations
* The data is historical Swedish data from 1961–62 with only 92 observations. It is not modern or Indian traffic data.
* There are no weather, traffic-volume, road-type or geographic variables.
* The speed limits were not randomised day by day, so the results are predictive associations and not causal effects.
* Small test sets give noisy metrics. Cross-validation and fold-to-fold spread are reported for that reason.
