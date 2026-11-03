# Predictive Modelling of Traffic Accident Counts Using Poisson and Negative Binomial Regression

### A Study of Historical Speed-Limit Conditions (Sweden, 1961–1962)

**Course:** CSE3141 Predictive Analytics · B.Tech CSE, Manipal University Jaipur · Mini-project

This project predicts the number of police-reported personal-injury traffic accidents on a 1962 day (`y2`) from the count on the matched 1961 day (`y1`) and the speed-limit status in both years (`lim1`, `lim2`). It compares a Poisson GLM and a Negative Binomial (NB2) GLM against two simple baselines, validates them out-of-sample, and presents everything in an interactive Streamlit dashboard.

| Component | Where |
|---|---|
| 1. Problem statement | [`docs/problem_statement.md`](docs/problem_statement.md) |
| 2. Data acquisition & understanding | `scripts/01_inspect_archive.py`, `scripts/02_extract_dataset.py`, [`docs/data_dictionary.md`](docs/data_dictionary.md) |
| 3. Preprocessing & exploratory analytics | `scripts/03_preprocess.py`, `scripts/04_eda.py` |
| 4. Predictive modelling & validation | `scripts/05_modeling.py`, `scripts/06_evaluation.py`, [`docs/methodology.md`](docs/methodology.md) |
| 5. Interactive dashboard | `dashboard/app.py` |

---

## Quick start (any computer: Windows, macOS or Linux)

Requires **Python 3.10–3.13** and Git.

```bash
git clone https://github.com/Aadya-Agarwal-muj/Predictive-analytics.git
cd Predictive-analytics
```

Create and activate a virtual environment, then install the dependencies:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Reproduce the entire analysis (about 20 seconds), run the tests, and open the dashboard:

```bash
python run_all.py
python -m pytest -q
streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501>. All results are already in `outputs/`, so the dashboard also works straight after cloning. `run_all.py` regenerates every file from the original CRAN archive.

> If PowerShell blocks `Activate.ps1`, run `Set-ExecutionPolicy -Scope Process Bypass` once in that window.

---

## Data

* **Source:** dataset `limits` in the CRAN package [SMPracticals 1.4-3.2](https://cran.r-project.org/package=SMPracticals) (Davison, *Statistical Models*, 2003). The original data is from Svensson, A. (1981), *Annals of Statistics* 9, 697–704.
* **Content:** 92 matched days × 5 variables (`day`, `lim1`, `lim2`, `y1`, `y2`). It records police-reported personal-injury accidents on Swedish roads on 92 days in 1961 and 92 matching days in 1962. On some days a general speed limit of 90 or 100 km/h was in force.
* The original archive `data/raw/SMPracticals_1.4-3.2.tar.gz` (162,136 bytes) is kept unchanged. Only `SMPracticals/data/limits.rda` is extracted from it and read with `pyreadr`. See [`data/README.md`](data/README.md).

## Pipeline

| Step | Script | Main outputs |
|---|---|---|
| Inspect archive | `01_inspect_archive.py` | Confirms that `limits.rda` exists and prints the official documentation |
| Extract and convert | `02_extract_dataset.py` | `data/raw/limits.rda`, `data/raw/limits_raw.csv` |
| Preprocess | `03_preprocess.py` | `data/processed/swedish_speed_limit_92.csv`, quality report, descriptive statistics, IQR outliers, box plots |
| EDA | `04_eda.py` | Figures 02–08, regime summary, correlations, day-code autocorrelation |
| Model (inference) | `05_modeling.py` | Model summaries, coefficients and IRRs, overdispersion tests, residual plots |
| Validate | `06_evaluation.py` | Hold-out metrics, blocked and repeated CV, interval coverage, feature-set comparison |

Shared code lives in `src/config.py` (paths and constants) and `src/modeling.py` (model fitting and metrics). The scripts and the dashboard both use it, so there is no duplicated logic.

## Key results (all produced by the code in this repository)

**Model:** log E[y2] = β0 + β1·y1 + β2·lim1 + β3·lim2

**Blocked 5-fold cross-validation** (contiguous day-code blocks; every day is predicted by a model that never saw it):

| Model | MAE | RMSE | Mean Poisson deviance |
|---|---|---|---|
| Mean baseline | 6.69 | 8.67 | 3.49 |
| Previous-year baseline (ŷ = y1) | 7.56 | 8.90 | 3.50 |
| Poisson GLM | **4.87** | **6.12** | **1.90** |
| Negative Binomial (NB2) | 4.90 | 6.16 | 1.91 |

* Both GLMs cut MAE by about 27% compared with the mean baseline and by about 35% compared with simply repeating last year's count. Repeated stratified 5-fold CV (20 repeats) gives the same ranking.
* **Overdispersion is present.** For the Poisson model, Pearson χ²/df = 1.79. The Cameron–Trivedi test gives p = 0.0008, and the likelihood-ratio test of NB2 against Poisson gives LR = 16.8 (p < 0.001). The NB2 dispersion parameter is α = 0.037. AIC is 593.0 for NB2 and 607.8 for Poisson.
* **Why NB2 is the recommended model:** its point predictions are almost the same as Poisson's, but its 90% prediction intervals cover 88% of held-out days. Poisson's intervals cover only 76%, because they are too narrow.
* **NB2 incidence-rate ratios:** each extra 1961 accident multiplies the expected 1962 count by 1.029. A 1962 speed limit is associated with a 27% lower expected count (IRR 0.73, 95% CI 0.64–0.83). `lim1` is not significant under NB2 (p = 0.11).
* These results describe association within one historical experiment. They do not prove that speed limits *cause* fewer accidents.

Full discussion: [`docs/methodology.md`](docs/methodology.md).

## Repository structure

```
├── data/
│   ├── raw/          original CRAN archive, extracted limits.rda, faithful CSV
│   └── processed/    cleaned dataset used for modelling
├── scripts/          01–06 numbered pipeline stages
├── src/              shared config and modelling functions
├── dashboard/app.py  Streamlit + Plotly dashboard (4 sections)
├── outputs/          figures/, metrics/, model_results/ (regenerated by run_all.py)
├── docs/             problem statement, data dictionary, methodology and results
├── tests/            pytest checks for the data and models
├── run_all.py        runs the full pipeline
└── requirements.txt
```

## Deploying the dashboard online (optional)

To give someone a link instead of asking them to install anything, deploy on [Streamlit Community Cloud](https://share.streamlit.io):

1. Sign in with GitHub and choose **New app**.
2. Pick this repository and branch, and set the main file to `dashboard/app.py`.

`requirements.txt` is detected automatically.

## Limitations

* The data is historical (Sweden, 1961–62, 92 days). Results should not be transferred to modern or Indian traffic.
* The dataset has no weather, traffic-volume, road-type or location variables.
* Speed limits were not assigned at random to individual days, so the coefficients are predictive associations and not causal effects.
* With n = 92, every metric carries noticeable sampling uncertainty. That is why the fold-to-fold standard deviations are reported.

## Licence and attribution

The data and the original archive come from SMPracticals (GPL ≥ 2) by A. C. Davison and A. R. Brazzale and are redistributed unchanged. The project code was written for academic coursework.
