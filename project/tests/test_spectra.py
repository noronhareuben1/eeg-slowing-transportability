import numpy as np

from rcd.spectra import compute_spectral_features


def test_spectral_features_are_finite_for_alpha_signal() -> None:
    sfreq = 250.0
    time = np.arange(int(sfreq * 15)) / sfreq
    rng = np.random.default_rng(4)
    signal = np.sin(2 * np.pi * 10 * time) + 0.25 * rng.normal(size=time.size)
    features = compute_spectral_features(signal, sfreq)
    values = np.asarray(list(features.__dict__.values()), dtype=float)
    assert np.isfinite(values).all()
    assert features.alpha_relative > features.theta_relative
    assert -1.0 < features.aperiodic_exponent < 4.0
