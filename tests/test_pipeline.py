"""
Automated checks (run with:  python -m pytest -q).

They confirm the processed data still matches the documented dataset and that
both models fit and behave sensibly. Run `python run_all.py` first.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EXPECTED_COLUMNS, EXPECTED_ROWS  # noqa: E402
from src.modeling import (  # noqa: E402
    fit_negbin,
    fit_poisson,
    load_processed_data,
    mean_poisson_deviance,
    predictive_interval,
)


@pytest.fixture(scope="module")
def df():
    return load_processed_data()


def test_shape_and_columns(df):
    assert len(df) == EXPECTED_ROWS
    assert list(df.columns[:5]) == EXPECTED_COLUMNS


def test_day_is_unique_identifier(df):
    assert sorted(df["day"]) == list(range(1, 93))


def test_data_quality(df):
    assert df[EXPECTED_COLUMNS].isnull().sum().sum() == 0
    assert set(df["lim1"]) <= {0, 1} and set(df["lim2"]) <= {0, 1}
    assert (df[["y1", "y2"]] >= 0).all().all()


def test_models_fit_and_predict_positive(df):
    pois = fit_poisson(df)
    nb, alpha, _ = fit_negbin(df)
    assert pois.converged and nb.converged
    assert alpha > 0
    assert (pois.fittedvalues > 0).all() and (nb.fittedvalues > 0).all()
    # A Poisson GLM with intercept reproduces the total count exactly.
    assert np.isclose(pois.fittedvalues.sum(), df["y2"].sum())


def test_deviance_zero_for_perfect_prediction():
    y = np.array([0, 3, 10])
    assert mean_poisson_deviance(y, np.where(y == 0, 1e-12, y)) == pytest.approx(0, abs=1e-9)


def test_negbin_interval_wider_than_poisson():
    lo_p, hi_p = predictive_interval([20.0], "poisson")
    lo_n, hi_n = predictive_interval([20.0], "negbin", alpha=0.05)
    assert (hi_n - lo_n) >= (hi_p - lo_p)
