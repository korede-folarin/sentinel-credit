"""
Drift monitoring: the Population Stability Index (PSI), v1 Decision 20.

Plain English: PSI measures how far a feature's distribution has moved
between a reference population and a newer one. Near 0 means no change;
the configured thresholds band it as stable, moderate or material shift.
"""

import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pytest

from conftest import load_module

# monitor_drift imports MLflow for its logging step only; the PSI maths does
# not use it. Where MLflow is not installed (the slim CI image), a placeholder
# module lets the script import so its pure functions can be tested.
if importlib.util.find_spec("mlflow") is None:
    sys.modules.setdefault("mlflow", types.ModuleType("mlflow"))

drift = load_module("monitor_drift", "pipeline/monitor_drift.py")
psi = drift.population_stability_index


def test_identical_distributions_have_near_zero_psi():
    x = pd.Series(np.random.default_rng(0).normal(size=5000))
    assert psi(x, x.copy(), 10) == pytest.approx(0.0, abs=1e-9)


def test_a_large_shift_is_material():
    rng = np.random.default_rng(1)
    ref, new = pd.Series(rng.normal(0, 1, 5000)), pd.Series(rng.normal(1.5, 1, 5000))
    value = psi(ref, new, 10)
    assert value > drift.PSI_HIGH and drift.psi_band(value) == "material_shift"


def test_a_small_shift_is_smaller_than_a_large_one():
    rng = np.random.default_rng(2)
    ref = pd.Series(rng.normal(0, 1, 5000))
    small = psi(ref, pd.Series(rng.normal(0.1, 1, 5000)), 10)
    large = psi(ref, pd.Series(rng.normal(1.0, 1, 5000)), 10)
    assert 0 < small < large


def test_psi_is_never_negative():
    rng = np.random.default_rng(3)
    for _ in range(5):
        assert psi(pd.Series(rng.exponential(1, 2000)), pd.Series(rng.exponential(1.3, 2000)), 10) >= 0


def test_categorical_features_use_category_shares():
    ref = pd.Series([0] * 700 + [1] * 300)
    same = pd.Series([0] * 70 + [1] * 30)
    flipped = pd.Series([0] * 300 + [1] * 700)
    assert psi(ref, same, 10) == pytest.approx(0.0, abs=1e-9)
    assert psi(ref, flipped, 10) > drift.PSI_HIGH


def test_missing_values_are_their_own_bucket():
    ref = pd.Series([1.0, 2.0, None] * 100)
    new = pd.Series([1.0, 2.0, 2.0] * 100)      # the missing values disappeared
    assert psi(ref, new, 10) > 0


def test_bands_follow_the_configured_thresholds():
    assert drift.psi_band(drift.PSI_MODERATE - 1e-9) == "stable"
    assert drift.psi_band(drift.PSI_MODERATE) == "moderate_shift"
    assert drift.psi_band(drift.PSI_HIGH) == "material_shift"
