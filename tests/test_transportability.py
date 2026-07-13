import numpy as np

from transportability.eeg_features import _subject_features


def test_transportability_features_are_finite_for_synthetic_recording():
    rng = np.random.default_rng(7)
    features = _subject_features(rng.normal(size=(19, 5000)), sfreq=500.0)
    assert set(features) >= {
        "posterior_delta_alpha_ratio",
        "posterior_alpha_relative",
        "global_aperiodic_exponent",
    }
    assert all(np.isfinite(value) for value in features.values())
