import numpy as np

from rcd.complexity import fractal_volatility, regional_summary


def test_regional_sign_convention() -> None:
    values = {"F1": 2.0, "F2": 4.0, "O1": 1.0, "O2": 1.0}
    summary = regional_summary(values, ["F1", "F2"], ["O1", "O2"])
    assert summary["rostral"] == 3.0
    assert summary["caudal"] == 1.0
    assert summary["rostrocaudal"] == 2.0


def test_fractal_volatility_is_finite() -> None:
    rng = np.random.default_rng(2)
    dimension, standard_error = fractal_volatility(rng.normal(size=2048))
    assert 1.0 < dimension < 2.1
    assert np.isfinite(standard_error)


def test_constant_signal_returns_plugin_fallback() -> None:
    dimension, standard_error = fractal_volatility(np.ones(128))
    assert dimension == 0.5
    assert np.isnan(standard_error)
