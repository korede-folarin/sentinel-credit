"""
v2 Phase 1 (V2-02): the PyTorch challenger's building blocks.

Plain English: the neural-network classifier behaves like a scikit-learn
model (so the champion's Platt calibration can wrap it), repeats exactly with
the same seed, and learns; the autoencoder gives unusual rows a higher
reconstruction error; v2 metrics use v1's definitions; and v2 can only write
to its own schema and can never record a promotion.

PyTorch and MLflow are not in the slim CI image, so these tests skip there
(and say why); they run locally.
"""

import os

import numpy as np
import pytest

from conftest import ROOT, load_module

torch = pytest.importorskip("torch", reason="PyTorch is not installed (slim CI image)")
tm = load_module("torch_models", "v2/phase1/torch_models.py")


def data(n=3000, seed=0):
    from sklearn.datasets import make_classification
    return make_classification(n, 13, n_informative=6, weights=[0.93], random_state=seed)


def test_mlp_learns_and_returns_valid_probabilities():
    from sklearn.metrics import roc_auc_score
    X, y = data()
    m = tm.MLPClassifierTorch(hidden=(16,), max_epochs=30).fit(X[:2000], y[:2000])
    p = m.predict_proba(X[2000:])
    assert p.shape == (1000, 2) and np.allclose(p.sum(axis=1), 1.0)
    assert roc_auc_score(y[2000:], p[:, 1]) > 0.8


def test_mlp_is_deterministic_with_a_fixed_seed():
    X, y = data()
    a = tm.MLPClassifierTorch(hidden=(16,), max_epochs=10, random_state=7).fit(X, y).predict_proba(X)
    b = tm.MLPClassifierTorch(hidden=(16,), max_epochs=10, random_state=7).fit(X, y).predict_proba(X)
    assert np.array_equal(a, b)


def test_mlp_works_inside_the_champions_calibration():
    from sklearn.calibration import CalibratedClassifierCV
    X, y = data()
    cal = CalibratedClassifierCV(tm.MLPClassifierTorch(hidden=(16,), max_epochs=10), method="sigmoid", cv=5)
    p = cal.fit(X, y).predict_proba(X)[:, 1]
    assert len(cal.calibrated_classifiers_) == 5 and ((p >= 0) & (p <= 1)).all()


def test_autoencoder_flags_unusual_rows():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(3000, 13))
    scaler, model, epochs, _ = tm.train_autoencoder(normal, [8, 4], 1e-3, 256, 40, 0.15, 5)
    err_normal = tm.reconstruction_error(scaler, model, rng.normal(0, 1, size=(500, 13)))
    err_odd = tm.reconstruction_error(scaler, model, rng.normal(4, 1, size=(500, 13)))
    assert epochs >= 1 and err_odd.mean() > 2 * err_normal.mean()


def test_v2_metrics_use_v1_definitions():
    pytest.importorskip("mlflow", reason="MLflow is not installed (slim CI image)")
    common = load_module("v2_phase1_common", "v2/phase1/common.py")
    rng = np.random.default_rng(1)
    y = (rng.random(4000) < 0.05).astype(int)
    p = np.clip(0.03 + 0.1 * y + rng.normal(0, 0.03, 4000), 0.001, 0.999)
    m = common.evaluate(y, p)
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1, abs=1e-3)
    assert m["business_cutoff"] == 0.0657
    flagged = p >= 0.0657
    assert m["share_sent_to_review"] == pytest.approx(flagged.mean(), abs=1e-4)
    assert m["recall_at_cutoff"] == pytest.approx((flagged & (y == 1)).sum() / y.sum(), abs=1e-4)


def test_v2_writes_only_to_its_own_schema():
    pytest.importorskip("mlflow", reason="MLflow is not installed (slim CI image)")
    import pandas as pd
    common = load_module("v2_phase1_common", "v2/phase1/common.py")
    for table in ("feature_store.customer_features", "reporting.feature_drift_log", "raw.customers"):
        with pytest.raises(ValueError):
            common.write_frame(table, pd.DataFrame({"a": [1]}))


def test_gate_outcomes_can_never_record_a_promotion():
    with open(os.path.join(ROOT, "v2", "sql", "phase1_results.sql"), encoding="utf-8") as f:
        ddl = f.read()
    assert "CHECK (promoted = false)" in ddl
