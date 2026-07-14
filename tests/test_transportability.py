import numpy as np
import pandas as pd

from transportability.derive_features import build_derivation_features
from transportability.eeg_features import _subject_features
from transportability.run_amendment_analysis import (
    CLASSES,
    multiclass_metrics,
    threshold_at_specificity,
)


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


def test_multiclass_metrics_are_participant_level_and_perfect_when_probabilities_match():
    y = np.asarray(["AD", "CN", "FTD", "AD", "CN", "FTD"])
    probability = np.full((len(y), len(CLASSES)), 0.05)
    for row, label in enumerate(y):
        probability[row, int(np.flatnonzero(CLASSES == label)[0])] = 0.90
    metrics = multiclass_metrics(y, probability)
    assert metrics["macro_roc_auc_ovr"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_threshold_at_specificity_uses_derivation_predictions_only():
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    probability = np.asarray([0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 0.90])
    threshold = threshold_at_specificity(y, probability, target=0.75)
    prediction = probability >= threshold
    specificity = ((~prediction) & (y == 0)).sum() / (y == 0).sum()
    assert specificity >= 0.75
