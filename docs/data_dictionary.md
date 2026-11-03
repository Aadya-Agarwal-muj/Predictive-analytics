# Data Dictionary — `swedish_speed_limit_92.csv`

The processed file has 92 rows. **One row = one matched day**: a day in 1961 together with its corresponding day in 1962. A row is not a single accident.

| Column | Type (processed) | Original R type | Values observed | Meaning | Role |
|---|---|---|---|---|---|
| `day` | int64 | factor | 1–92 (unique) | Matched-day code | Identifier (excluded from the model) |
| `lim1` | int64 | numeric | 0 / 1 (21 days = 1) | 1 = general speed limit in force on the 1961 day | Predictor |
| `lim2` | int64 | numeric | 0 / 1 (48 days = 1) | 1 = speed limit in force on the matched 1962 day | Predictor |
| `y1` | int64 | integer | 8–47 | Police-reported personal-injury accidents on the 1961 day | Predictor |
| `y2` | int64 | integer | 7–49 | Same count on the matched 1962 day | **Target** |
| `log_y1` | float64 | derived | 2.08–3.85 | ln(`y1`), an engineered candidate feature | Tested in feature selection (not chosen) |
| `regime` | text | derived | 4 labels | Readable label for the (`lim1`, `lim2`) combination | Charts only |

## Speed-limit regimes

| Regime | lim1 | lim2 | Days |
|---|---|---|---|
| No limit either year | 0 | 0 | 28 |
| Limit in 1962 only | 0 | 1 | 43 |
| Limit in 1961 only | 1 | 0 | 16 |
| Limit both years | 1 | 1 | 5 |

## Example row
`day = 12, lim1 = 0, lim2 = 1, y1 = 40, y2 = 23` means: on day code 12 there was no speed limit in 1961 and 40 injury accidents were reported. On the matched day in 1962 a speed limit was in force and 23 accidents were reported.

## Notes from the analysis
* Day codes appear in 8 contiguous blocks that share a speed-limit pattern. The counts also show strong lag-7 autocorrelation (0.47 for y1, 0.58 for y2), which suggests a weekly cycle and therefore that consecutive codes are consecutive days. The documentation only calls `day` a factor, so it is not used as a numeric predictor. It is, however, taken into account when designing the validation.
* There are no missing values, no duplicated days and no zero counts.
