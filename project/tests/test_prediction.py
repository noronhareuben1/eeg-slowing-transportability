import numpy as np

from rcd.prediction import CLASSES, _metrics


def test_prediction_metrics_perfect_probabilities() -> None:
    y_true = np.asarray(["AD", "CN", "FTD", "AD", "CN", "FTD"])
    probability = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]
    )
    assert CLASSES.tolist() == ["AD", "CN", "FTD"]
    metrics = _metrics(y_true, probability)
    assert metrics["macro_roc_auc_ovr"] == 1.0
    assert metrics["accuracy"] == 1.0
