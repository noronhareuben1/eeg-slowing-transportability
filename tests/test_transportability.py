import numpy as np
import pandas as pd

from transportability.eeg_features import _subject_features
from transportability.derive_features import build_derivation_features


def test_transportability_features_are_finite_for_synthetic_recording():
    rng = np.random.default_rng(7)
    features = _subject_features(rng.normal(size=(19, 5000)), sfreq=500.0)
    assert set(features) >= {
        "posterior_delta_alpha_ratio",
        "posterior_alpha_relative",
        "global_aperiodic_exponent",
    }
    assert all(np.isfinite(value) for value in features.values())


def test_derivation_feature_builder_uses_locked_regional_features():
    regional = pd.DataFrame(
        {
            "participant_id": ["sub-001"],
            "diagnosis": ["AD"],
            "delta_relative_caudal": [0.25],
            "alpha_relative_caudal": [0.5],
            "aperiodic_exponent_global": [1.2],
        }
    )
    features = build_derivation_features(regional)
    assert features.loc[0, "posterior_delta_alpha_ratio"] == 0.5
    assert features.loc[0, "posterior_alpha_relative"] == 0.5
    assert features.loc[0, "global_aperiodic_exponent"] == 1.2
