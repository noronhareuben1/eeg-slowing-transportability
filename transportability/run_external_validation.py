"""Fit the locked panel in derivation data and evaluate once externally."""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = (
    "posterior_delta_alpha_ratio",
    "posterior_alpha_relative",
    "global_aperiodic_exponent",
)


def _bootstrap(y: np.ndarray, p: np.ndarray, metric, seed: int = 19) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(float(metric(y[idx], p[idx])))
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def evaluate(derivation: pd.DataFrame, external: pd.DataFrame) -> dict[str, object]:
    derivation = derivation[derivation.diagnosis.isin(["AD", "CN"])].copy()
    external = external[external.diagnosis.isin(["AD", "CN"])].copy()
    derivation = derivation.dropna(subset=list(FEATURES))
    external = external.dropna(subset=list(FEATURES))
    x_train = derivation.loc[:, FEATURES].to_numpy(float)
    y_train = (derivation.diagnosis == "AD").astype(int).to_numpy()
    x_test = external.loc[:, FEATURES].to_numpy(float)
    y_test = (external.diagnosis == "AD").astype(int).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=19),
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)
    fpr, tpr, thresholds = roc_curve(y_train, model.predict_proba(x_train)[:, 1])
    valid = np.flatnonzero(fpr <= 0.10)
    threshold = float(thresholds[valid[-1]]) if len(valid) else float(thresholds[0])
    predictions = probabilities >= threshold
    result = {
        "n_derivation": int(len(derivation)),
        "n_external": int(len(external)),
        "external_auc": float(auc),
        "external_auc_ci95": _bootstrap(y_test, probabilities, roc_auc_score),
        "external_average_precision": float(average_precision_score(y_test, probabilities)),
        "external_brier": float(brier_score_loss(y_test, probabilities)),
        "external_sensitivity_at_derivation_threshold": float((predictions & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)),
        "external_specificity_at_derivation_threshold": float((~predictions & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)),
        "derivation_selected_threshold": threshold,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derivation", type=pathlib.Path, required=True)
    parser.add_argument("--external", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = evaluate(pd.read_csv(args.derivation), pd.read_csv(args.external))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
