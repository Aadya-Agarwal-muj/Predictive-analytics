"""
Streamlit dashboard for the Swedish speed-limit count-regression project.

Start it from the project root:
    streamlit run dashboard/app.py

Sections
    1. Dataset Overview            - size, dictionary, quality checks, statistics
    2. Predictive Analytics (EDA)  - distributions, matched years, regimes, correlations
    3. Model Evaluation            - baselines vs Poisson vs NB2, CV, dispersion, residuals
    4. Prediction Sandbox          - enter a scenario, get an expected count + intervals

The dashboard reads the processed data and the evaluation tables written by
`python run_all.py`, and refits the two final models live (92 rows -> instant).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import METRICS_DIR, MODEL_DIR, PROCESSED_CSV_PATH  # noqa: E402
from src.modeling import (  # noqa: E402
    coefficient_table,
    design_matrix,
    fit_negbin,
    fit_poisson,
    load_processed_data,
    predictive_interval,
)

st.set_page_config(page_title="Accident Count Prediction", page_icon="🚦", layout="wide")

REGIME_ORDER = ["No limit either year", "Limit in 1962 only",
                "Limit in 1961 only", "Limit both years"]
MODEL_ORDER = ["Mean baseline", "Previous-year baseline", "Poisson GLM", "NegBin (NB2)"]
COLOURS = {"1961": "#4C72B0", "1962": "#DD8452", "Poisson GLM": "#4C72B0",
           "NegBin (NB2)": "#C44E52", "Mean baseline": "#BBBBBB",
           "Previous-year baseline": "#888888"}

DATA_DICTIONARY = pd.DataFrame(
    [
        ["day", "Matched-day code 1-92 (an R factor in the original)", "int", "Identifier - not a predictor"],
        ["lim1", "1 = general speed limit (90/100 km/h) imposed on that day in 1961, 0 = not", "binary", "Predictor"],
        ["lim2", "1 = speed limit imposed on the matched day in 1962, 0 = not", "binary", "Predictor"],
        ["y1", "Police-reported personal-injury accidents on Swedish roads that day in 1961", "count", "Predictor"],
        ["y2", "Police-reported personal-injury accidents on the matched day in 1962", "count", "TARGET"],
        ["log_y1", "Natural log of y1 (engineered, tested in feature selection)", "float", "Candidate feature"],
        ["regime", "Text label of the (lim1, lim2) combination, for charts", "text", "Derived label"],
    ],
    columns=["Column", "Meaning", "Type", "Role"],
)


# ------------------------------------------------------------------ loaders
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_processed_data()


@st.cache_data
def read_table(path: Path):
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def get_models():
    df = get_data()
    pois = fit_poisson(df)
    nb_glm, alpha_hat, nb_discrete = fit_negbin(df)
    return pois, nb_glm, alpha_hat, nb_discrete


if not PROCESSED_CSV_PATH.exists():
    st.error("Processed data not found. From the project folder run:  `python run_all.py`  "
             "then restart:  `streamlit run dashboard/app.py`")
    st.stop()

df = get_data()
pois, nb_glm, alpha_hat, nb_discrete = get_models()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("🚦 Accident Count Prediction")
st.sidebar.caption("CSE3141 Predictive Analytics - mini project")
section = st.sidebar.radio(
    "Section",
    ["1. Dataset Overview", "2. Predictive Analytics (EDA)",
     "3. Model Evaluation", "4. Prediction Sandbox"],
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Academic demonstration on **historical Swedish data (1961-62)**. "
    "It is not an operational traffic-forecasting service and does not "
    "establish causal effects of speed limits."
)
st.sidebar.caption("Data: `limits` in CRAN package SMPracticals 1.4-3.2 "
                   "(Svensson, 1981, *Annals of Statistics* 9, 697-704).")

st.title("Predictive Modelling of Traffic Accident Counts")
st.caption("Poisson and Negative Binomial regression on historical speed-limit conditions")


# ======================================================== 1. DATASET OVERVIEW
if section.startswith("1"):
    st.header("1. Dataset Overview")
    core = df[["day", "lim1", "lim2", "y1", "y2"]]
    c = st.columns(5)
    c[0].metric("Observations (matched days)", len(core))
    c[1].metric("Original variables", core.shape[1])
    c[2].metric("Missing values", int(core.isnull().sum().sum()))
    c[3].metric("Duplicated rows", int(core.duplicated().sum()))
    c[4].metric("Days with 1962 limit", int(core["lim2"].sum()))

    st.subheader("Data dictionary")
    st.dataframe(DATA_DICTIONARY, hide_index=True, width="stretch")
    st.markdown(
        "**One row = one matched day**: the accident count on a day in 1961 (`y1`), the count on "
        "the corresponding day in 1962 (`y2`), and whether a speed limit applied on each. "
        "A row is *not* one accident."
    )

    st.subheader("Descriptive statistics")
    desc = read_table(METRICS_DIR / "descriptive_statistics.csv")
    if desc is not None:
        st.dataframe(desc.rename(columns={"Unnamed: 0": "variable"}).round(3),
                     hide_index=True, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data-quality checks")
        quality = read_table(METRICS_DIR / "data_quality_report.csv")
        if quality is not None:
            st.dataframe(quality.astype(str), hide_index=True, width="stretch")
    with col2:
        st.subheader("IQR outliers (flagged, kept)")
        out = read_table(METRICS_DIR / "iqr_outliers.csv")
        if out is not None:
            st.dataframe(out, hide_index=True, width="stretch")
        st.caption("High-accident days are plausible real events; deleting them would bias "
                   "the model towards under-predicting busy days.")
        fig = go.Figure()
        for col, year in [("y1", "1961"), ("y2", "1962")]:
            fig.add_box(y=df[col], name=f"{year} ({col})", marker_color=COLOURS[year],
                        boxpoints="outliers")
        fig.update_layout(height=320, margin=dict(t=20, b=20), yaxis_title="Accidents per day")
        st.plotly_chart(fig, width="stretch")

    with st.expander("Show all 92 observations"):
        st.dataframe(core, hide_index=True, width="stretch")


# ================================================== 2. PREDICTIVE ANALYTICS
elif section.startswith("2"):
    st.header("2. Predictive Analytics - exploratory evidence")
    long = df.melt(id_vars=["day", "regime", "lim1", "lim2"], value_vars=["y1", "y2"],
                   var_name="variable", value_name="accidents")
    long["year"] = long["variable"].map({"y1": "1961", "y2": "1962"})

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribution of daily counts")
        fig = px.histogram(long, x="accidents", color="year", barmode="overlay", nbins=16,
                           opacity=0.65, color_discrete_map=COLOURS)
        fig.update_layout(height=380, yaxis_title="Number of days")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"1961: mean {df.y1.mean():.2f}, variance {df.y1.var():.2f}.  "
                   f"1962: mean {df.y2.mean():.2f}, variance {df.y2.var():.2f}. "
                   "Right-skewed, strictly non-negative counts -> count regression.")
    with col2:
        st.subheader("Matched-year comparison")
        plot_df = df.assign(**{"1962 limit": df["lim2"].map({0: "No", 1: "Yes"})})
        fig = px.scatter(plot_df, x="y1", y="y2", color="1962 limit", hover_data=["day", "regime"],
                         color_discrete_map={"No": "#4C72B0", "Yes": "#C44E52"})
        fig.add_trace(go.Scatter(x=[0, 52], y=[0, 52], mode="lines", name="y2 = y1",
                                 line=dict(dash="dot", color="grey")))
        fig.update_layout(height=380, xaxis_title="1961 count (y1)", yaxis_title="1962 count (y2)")
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Pearson r(y1, y2) = {df.y1.corr(df.y2):.3f}. Red points (1962 limit) sit mostly "
                   "below the no-change line.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Counts by speed-limit regime")
        fig = px.box(long, x="regime", y="accidents", color="year", points="all",
                     category_orders={"regime": REGIME_ORDER}, color_discrete_map=COLOURS)
        fig.update_layout(height=400, xaxis_title="")
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Within-day change (y2 - y1)")
        ch = df.assign(change=df.y2 - df.y1)
        fig = px.strip(ch, x="regime", y="change", category_orders={"regime": REGIME_ORDER},
                       hover_data=["day", "y1", "y2"])
        fig.add_hline(y=0, line_dash="dot", line_color="grey")
        fig.update_layout(height=400, xaxis_title="")
        st.plotly_chart(fig, width="stretch")

    regime = read_table(METRICS_DIR / "eda_regime_summary.csv")
    if regime is not None:
        st.dataframe(regime.round(3), hide_index=True, width="stretch")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Correlations")
        corr = df[["y1", "y2", "lim1", "lim2"]].corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(height=380)
        st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Structure of the day code")
        fig = px.line(long, x="day", y="accidents", color="year", markers=True,
                      color_discrete_map=COLOURS)
        for _, r in df[df.lim2 == 1].iterrows():
            fig.add_vrect(x0=r.day - 0.5, x1=r.day + 0.5, fillcolor="red", opacity=0.06, line_width=0)
        fig.update_layout(height=380, xaxis_title="day code (shaded = limit in 1962)")
        st.plotly_chart(fig, width="stretch")
        acf = read_table(METRICS_DIR / "eda_day_code_autocorrelation.csv")
        if acf is not None:
            lag7 = acf.loc[acf.lag == 7]
            st.caption(f"Lag-7 autocorrelation: y1 = {lag7.y1_autocorr.iloc[0]:.2f}, "
                       f"y2 = {lag7.y2_autocorr.iloc[0]:.2f} -> a weekly cycle, so neighbouring "
                       "codes are consecutive days. This is why blocked cross-validation is used.")

    st.subheader("Main analytical findings")
    means = df.groupby("regime")[["y1", "y2"]].mean()
    r = means.loc["Limit in 1962 only"]
    n = means.loc["No limit either year"]
    st.markdown(
        f"""
- The 1961 count of a day is the strongest single signal for its 1962 count (r = {df.y1.corr(df.y2):.2f}).
- Days with a limit **only in 1962** averaged {r.y1:.1f} accidents in 1961 but {r.y2:.1f} in 1962
  (ratio {r.y2 / r.y1:.2f}); days with **no limit in either year** went from {n.y1:.1f} to {n.y2:.1f}
  (ratio {n.y2 / n.y1:.2f}).
- Variance exceeds the mean within every regime -> Poisson's equal mean-variance assumption must be tested.
- These are **associations** in one historical experiment; they are not proof that limits caused the change.
"""
    )


# ===================================================== 3. MODEL EVALUATION
elif section.startswith("3"):
    st.header("3. Predictive Model Evaluation")
    cv = read_table(METRICS_DIR / "cv_summary.csv")
    if cv is None:
        st.error("Evaluation tables not found - run `python run_all.py` first.")
        st.stop()

    scheme = st.selectbox("Validation scheme", cv["scheme"].unique().tolist())
    sub = cv[cv.scheme == scheme].set_index("model").reindex(MODEL_ORDER).reset_index()
    best = sub.loc[sub["MAE mean"].idxmin()]
    c = st.columns(4)
    c[0].metric("Best model (CV MAE)", best["model"])
    c[1].metric("Its MAE", f"{best['MAE mean']:.2f}")
    mean_mae = sub.loc[sub.model == "Mean baseline", "MAE mean"].iloc[0]
    c[2].metric("Improvement vs mean baseline", f"{(1 - best['MAE mean'] / mean_mae) * 100:.0f}%")
    c[3].metric("NB2 dispersion alpha", f"{alpha_hat:.4f}")

    metric = st.radio("Metric", ["MAE", "RMSE", "Mean Poisson deviance"], horizontal=True)
    fig = px.bar(sub, x="model", y=f"{metric} mean", error_y=f"{metric} std", color="model",
                 color_discrete_map=COLOURS)
    fig.update_layout(height=360, showlegend=False, xaxis_title="", yaxis_title=f"{metric} (lower is better)")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sub.round(3), hide_index=True, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Actual vs predicted (out-of-fold, blocked CV)")
        oof = read_table(MODEL_DIR / "oof_predictions_blocked_cv.csv")
        if oof is not None:
            choice = st.selectbox("Model", MODEL_ORDER, index=3)
            fig = px.scatter(oof, x="y2", y=choice, hover_data=["day", "regime", "y1"],
                             color="regime", category_orders={"regime": REGIME_ORDER})
            fig.add_trace(go.Scatter(x=[5, 50], y=[5, 50], mode="lines", name="perfect",
                                     line=dict(dash="dot", color="grey")))
            fig.update_layout(height=420, xaxis_title="Actual 1962 count",
                              yaxis_title="Predicted (never trained on this day)")
            st.plotly_chart(fig, width="stretch")
    with col2:
        st.subheader("Hold-out split: train vs test")
        hold = read_table(METRICS_DIR / "holdout_metrics.csv")
        if hold is not None:
            st.dataframe(hold.round(3), hide_index=True, width="stretch")
            st.caption("Similar train and test errors -> no sign of overfitting. With only 19 test "
                       "days a single split is noisy, so cross-validation is the main evidence.")

    st.subheader("Dispersion: is Poisson's Var = Mean assumption violated?")
    od = read_table(METRICS_DIR / "overdispersion_tests.csv")
    col1, col2 = st.columns(2)
    with col1:
        if od is not None:
            od.columns = ["statistic", "value"]
            st.dataframe(od.round(4), hide_index=True, width="stretch")
    with col2:
        cov = read_table(METRICS_DIR / "interval_coverage.csv")
        if cov is not None:
            st.markdown("**90% prediction-interval coverage (target 0.90)**")
            st.dataframe(cov.round(3), hide_index=True, width="stretch")
        st.info("Poisson and NB2 make almost identical point predictions, but Poisson's intervals "
                "are too narrow (they ignore the extra variance). NB2's intervals cover close to "
                "the nominal 90% - this is the practical benefit of modelling overdispersion.")

    st.subheader("Coefficients and incidence-rate ratios (fit on all 92 days)")
    coefs = pd.concat([coefficient_table(pois, "Poisson"), coefficient_table(nb_glm, "NegBin (NB2)")])
    show = coefs[["model", "term", "coef", "std_err", "p_value", "IRR", "IRR_ci_low",
                  "IRR_ci_high", "pct_change"]].round(4)
    st.dataframe(show, hide_index=True, width="stretch")
    nb = coefs[coefs.model == "NegBin (NB2)"].set_index("term")
    st.markdown(
        f"""
**Reading the NB2 coefficients (exp(coef) = IRR):**
- `y1`: each extra 1961 accident multiplies the expected 1962 count by **{nb.loc['y1', 'IRR']:.3f}**
  (+{nb.loc['y1', 'pct_change']:.1f}%), other inputs fixed.
- `lim2`: a speed limit on the 1962 day is associated with an expected count **{abs(nb.loc['lim2', 'pct_change']):.1f}% lower**
  (IRR {nb.loc['lim2', 'IRR']:.3f}, 95% CI {nb.loc['lim2', 'IRR_ci_low']:.3f}-{nb.loc['lim2', 'IRR_ci_high']:.3f}).
- `lim1`: a limit on the 1961 day means y1 was recorded under a limit; IRR {nb.loc['lim1', 'IRR']:.3f}
  (p = {nb.loc['lim1', 'p_value']:.3f}, not significant at 5% under NB2).
"""
    )

    st.subheader("Residual diagnostics")
    which = st.radio("Model", ["Poisson GLM", "NegBin (NB2)"], horizontal=True)
    res = pois if which == "Poisson GLM" else nb_glm
    rd = pd.DataFrame({"fitted": res.fittedvalues, "pearson": res.resid_pearson, "day": df.day})
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(rd, x="fitted", y="pearson", hover_data=["day"])
        for b in (-2, 0, 2):
            fig.add_hline(y=b, line_dash="dot" if b else "solid", line_color="red" if b else "grey")
        fig.update_layout(height=360, title=f"Pearson residuals ({(rd.pearson.abs() > 2).mean():.0%} beyond ±2)")
        st.plotly_chart(fig, width="stretch")
    with col2:
        dev = np.sort(np.asarray(res.resid_deviance))
        theo = stats.norm.ppf((np.arange(1, len(dev) + 1) - 0.5) / len(dev))
        fig = px.scatter(x=theo, y=dev, labels={"x": "Theoretical normal quantile", "y": "Deviance residual"})
        fig.add_trace(go.Scatter(x=[-3, 3], y=[-3, 3], mode="lines", line=dict(dash="dot", color="grey"),
                                 showlegend=False))
        fig.update_layout(height=360, title="Q-Q plot of deviance residuals")
        st.plotly_chart(fig, width="stretch")

    fs = read_table(METRICS_DIR / "feature_selection_cv.csv")
    if fs is not None:
        with st.expander("Feature-set comparison (blocked 5-fold CV)"):
            st.dataframe(fs.round(3), hide_index=True, width="stretch")


# =================================================== 4. PREDICTION SANDBOX
else:
    st.header("4. Prediction Sandbox")
    st.warning("Academic historical demonstration only - not a validated real-time traffic "
               "forecasting tool. Predictions apply to conditions like Sweden in 1961-62.")
    col1, col2 = st.columns([1, 2])
    with col1:
        y1 = st.number_input("Accidents on the matched 1961 day (y1)", min_value=0,
                             max_value=100, value=int(round(df.y1.median())), step=1)
        lim1 = st.radio("Speed limit on the 1961 day?", ["No", "Yes"], horizontal=True)
        lim2 = st.radio("Speed limit on the 1962 day?", ["No", "Yes"], horizontal=True)
        model_name = st.radio("Model", ["NegBin (NB2) - recommended", "Poisson GLM"])
        if not df.y1.min() <= y1 <= df.y1.max():
            st.error(f"y1 = {y1} is outside the observed range {df.y1.min()}-{df.y1.max()}: "
                     "this is extrapolation and the prediction is unreliable.")

    scenario = pd.DataFrame({"y1": [y1], "lim1": [int(lim1 == "Yes")], "lim2": [int(lim2 == "Yes")]})
    X_new = design_matrix(scenario)
    is_nb = model_name.startswith("NegBin")
    res = nb_glm if is_nb else pois
    pred = res.get_prediction(X_new).summary_frame(alpha=0.05)
    mu = float(pred["mean"].iloc[0])
    lo90, hi90 = predictive_interval([mu], "negbin" if is_nb else "poisson", alpha_hat, 0.90)

    with col2:
        c = st.columns(3)
        c[0].metric("Predicted expected accidents", f"{mu:.1f}")
        c[1].metric("95% CI for the mean", f"{pred['mean_ci_lower'].iloc[0]:.1f} - {pred['mean_ci_upper'].iloc[0]:.1f}")
        c[2].metric("90% prediction interval (count)", f"{int(lo90[0])} - {int(hi90[0])}")
        ks = np.arange(0, int(max(60, mu * 2.5)))
        if is_nb:
            n = 1 / alpha_hat
            pmf = stats.nbinom.pmf(ks, n, n / (n + mu))
        else:
            pmf = stats.poisson.pmf(ks, mu)
        fig = px.bar(x=ks, y=pmf, labels={"x": "Accident count on the 1962 day", "y": "Probability"})
        fig.add_vline(x=mu, line_dash="dash", line_color="black", annotation_text="expected")
        fig.update_layout(height=340, title=f"Predicted count distribution - {model_name.split(' -')[0]}")
        st.plotly_chart(fig, width="stretch")

    b = res.params
    st.markdown(
        f"""
**How this number is computed**
log(μ) = {b['const']:.4f} + {b['y1']:.4f}·y1 + {b['lim1']:.4f}·lim1 + {b['lim2']:.4f}·lim2
= {b['const']:.4f} + {b['y1']:.4f}×{y1} + {b['lim1']:.4f}×{scenario.lim1[0]} + {b['lim2']:.4f}×{scenario.lim2[0]}
= {np.log(mu):.4f}  →  μ = exp({np.log(mu):.4f}) = **{mu:.2f}**

*Confidence interval* = uncertainty about the **average** count for days like this.
*Prediction interval* = range for **one actual day's** count, which is wider because individual days
also vary randomly{' (NB2 adds extra variance αμ² with α = ' + f'{alpha_hat:.4f})' if is_nb else ' (Poisson variance = μ)'}.
"""
    )
    similar = df[(df.lim1 == scenario.lim1[0]) & (df.lim2 == scenario.lim2[0])]
    st.caption(f"Historical days with the same speed-limit pattern: {len(similar)} "
               f"(mean y2 = {similar.y2.mean():.1f})." if len(similar) else
               "No historical days share this speed-limit pattern.")
