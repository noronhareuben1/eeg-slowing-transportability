import numpy as np

from transportability.photic_response_features import interval_features


def test_interval_features_detect_stimulus_entrainment() -> None:
    sfreq = 250.0
    time = np.arange(int(3.0 * sfreq)) / sfreq
    rng = np.random.default_rng(42)
    signal = np.sin(2 * np.pi * 10.0 * time) + 0.10 * rng.normal(size=time.size)

    features = interval_features(signal, sfreq=sfreq, stimulus_hz=10.0)

    assert features["stimulus_snr_db"] > 15.0
    assert features["alpha_relative"] > features["theta_relative"]
    assert np.isfinite(list(features.values())).all()
