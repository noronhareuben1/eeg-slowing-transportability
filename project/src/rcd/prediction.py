from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from rcd.manifest import write_manifest

CLASSES = np.asarray(["AD", "CN", "FTD"])


def _assemble_features(project_root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    mechanistic = pd.read_csv(project_root / "outputs" / "mechanistic" / "regional_features.csv")
    reproduction = pd.read_csv(
        project_root / "outputs" / "reproduction" / "regional_complexity.csv"
    )
    metadata = mechanistic[
        ["participant_id", "diagnosis", "Age", "Gender", "MMSE"]
    ].copy()
    metadata["male"] = (metadata["Gender"].str.upper() == "M").astype(float)

    spectral_bases = [
        "delta_relative",
        "theta_relative",
        "alpha_relative",
        "beta_relative",
        "gamma_relative",
        "spectral_edge_95_hz",
        "median_frequency_hz",
        "loglog_slope",
        "aperiodic_offset",
        "aperiodic_exponent",
    ]
    spectral_columns = [
        f"{base}_{suffix}"
        for base in spectral_bases
        for suffix in ("global", "rostrocaudal")
    ]
    metadata = metadata.merge(
        mechanistic[["participant_id", *spectral_columns]],
        on="participant_id",
        validate="one_to_one",
    )

    complexity_wide = reproduction.pivot(
        index="participant_id",
        columns="metric",
        values=["global", "rostrocaudal"],
    )
    complexity_wide.columns = [
        f"complexity_{metric}_{region}" for region, metric in complexity_wide.columns
    ]
    complexity_wide = complexity_wide.reset_index()
    surrogate_columns = [
        "surrogate_hfd_z_global",
        "surrogate_hfd_z_rostrocaudal",
    ]
    complexity_wide = complexity_wide.merge(
        mechanistic[["participant_id", *surrogate_columns]],
        on="participant_id",
        validate="one_to_one",
    )
    metadata = metadata.merge(complexity_wide, on="participant_id", validate="one_to_one")
    complexity_columns = [
        column
        for column in metadata.columns
        if column.startswith("complexity_") or column.startswith("surrogate_hfd_z_")
    ]
    feature_sets = {
        "demographic": ["Age", "male", "MMSE"],
        "spectral": spectral_columns,
        "complexity": complexity_columns,
        "feature_fusion": [*spectral_columns, *complexity_columns],
    }
    return metadata.sort_values("participant_id").reset_index(drop=True), feature_sets


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = probability / probability.sum(axis=1, keepdims=True)
    predicted = CLASSES[np.argmax(probability, axis=1)]
    return {
        "macro_roc_auc_ovr": float(
            roc_auc_score(y_true, probability, labels=CLASSES, multi_class="ovr", average="macro")
        ),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=CLASSES, average="macro")),
        "log_loss": float(log_loss(y_true, probability, labels=CLASSES)),
    }


def _bootstrap_auc(
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
    iterations: int,
) -> tuple[dict[str, tuple[float, float]], dict[str, tuple[float, float]]]:
    def interval(values: np.ndarray) -> tuple[float, float]:
        quantiles = np.quantile(values, [0.025, 0.975])
        return float(quantiles[0]), float(quantiles[1])

    rng = np.random.default_rng(seed)
    model_names = list(predictions)
    y_true = predictions[model_names[0]][0]
    class_indices = {label: np.flatnonzero(y_true == label) for label in CLASSES}
    bootstrap = {name: np.empty(iterations, dtype=float) for name in model_names}
    for iteration in range(iterations):
        indices = np.concatenate(
            [
                rng.choice(class_indices[label], size=class_indices[label].size, replace=True)
                for label in CLASSES
            ]
        )
        for name, (_, probability) in predictions.items():
            bootstrap[name][iteration] = roc_auc_score(
                y_true[indices],
                probability[indices],
                labels=CLASSES,
                multi_class="ovr",
                average="macro",
            )
    intervals = {
        name: interval(values)
        for name, values in bootstrap.items()
    }
    fusion = bootstrap["feature_fusion"]
    differences = {
        name: interval(fusion - values)
        for name, values in bootstrap.items()
        if name != "feature_fusion"
    }
    return intervals, differences


def run_classical_prediction(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path]:
    table, feature_sets = _assemble_features(project_root)
    y = table["diagnosis"].to_numpy()
    settings = config["classification"]
    outer_folds = int(settings["outer_folds"])
    outer_repeats = int(settings["outer_repeats"])
    inner_folds = int(settings["inner_folds"])
    seed = int(config["study"]["seed"])

    split_records: list[dict[str, Any]] = []
    frozen_splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(outer_repeats):
        splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=seed + repeat,
        )
        for fold, (train, test) in enumerate(splitter.split(table, y)):
            frozen_splits.append((repeat, fold, train, test))
            train_ids = set(table.iloc[train]["participant_id"])
            test_ids = set(table.iloc[test]["participant_id"])
            if train_ids & test_ids:
                raise RuntimeError("Participant leakage detected in outer split")
            split_records.extend(
                [
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": participant_id,
                        "role": role,
                    }
                    for role, indices in (("train", train), ("test", test))
                    for participant_id in table.iloc[indices]["participant_id"]
                ]
            )

    output_dir = project_root / "outputs" / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    splits_path = output_dir / "frozen_subject_splits.csv"
    pd.DataFrame(split_records).to_csv(splits_path, index=False)

    prediction_records: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []
    parameter_grid = {
        "classifier__C": [0.01, 0.1, 1.0],
        "classifier__l1_ratio": [0.0, 0.5, 1.0],
    }
    for model_name, columns in feature_sets.items():
        values = table[columns].to_numpy(dtype=float)
        for repeat, fold, train, test in frozen_splits:
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            solver="saga",
                            class_weight="balanced",
                            max_iter=5000,
                            tol=1e-3,
                            random_state=seed + repeat * 100 + fold,
                        ),
                    ),
                ]
            )
            inner = StratifiedKFold(
                n_splits=inner_folds,
                shuffle=True,
                random_state=seed + 1000 + repeat * 100 + fold,
            )
            search = GridSearchCV(
                pipeline,
                parameter_grid,
                scoring="roc_auc_ovr",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(values[train], y[train])
            raw_probability = search.predict_proba(values[test])
            probability = np.zeros((test.size, CLASSES.size), dtype=float)
            for source_index, label in enumerate(search.best_estimator_.classes_):
                target_index = int(np.flatnonzero(CLASSES == label)[0])
                probability[:, target_index] = raw_probability[:, source_index]
            for row_index, participant_index in enumerate(test):
                prediction_records.append(
                    {
                        "model": model_name,
                        "repeat": repeat,
                        "fold": fold,
                        "participant_id": table.iloc[participant_index]["participant_id"],
                        "diagnosis": y[participant_index],
                        **{
                            f"probability_{label}": float(probability[row_index, class_index])
                            for class_index, label in enumerate(CLASSES)
                        },
                    }
                )
            tuning_records.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "inner_best_macro_auc": float(search.best_score_),
                    "C": float(search.best_params_["classifier__C"]),
                    "l1_ratio": float(search.best_params_["classifier__l1_ratio"]),
                }
            )

    predictions = pd.DataFrame(prediction_records)
    tuning = pd.DataFrame(tuning_records)
    prediction_path = output_dir / "classical_outer_predictions.csv"
    tuning_path = output_dir / "classical_inner_tuning.csv"
    predictions.to_csv(prediction_path, index=False)
    tuning.to_csv(tuning_path, index=False)

    probability_columns = [f"probability_{label}" for label in CLASSES]
    aggregated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    summary: dict[str, Any] = {"models": {}, "feature_sets": feature_sets}
    for model_name, frame in predictions.groupby("model", sort=False):
        participant = (
            frame.groupby(["participant_id", "diagnosis"], as_index=False)[probability_columns]
            .mean()
            .sort_values("participant_id")
        )
        y_true = participant["diagnosis"].to_numpy()
        probability = participant[probability_columns].to_numpy(dtype=float)
        aggregated[model_name] = (y_true, probability)
        summary["models"][model_name] = _metrics(y_true, probability)
    intervals, differences = _bootstrap_auc(
        aggregated,
        seed=seed + 5000,
        iterations=int(config["statistics"]["bootstrap_iterations"]),
    )
    for model_name, (low, high) in intervals.items():
        summary["models"][model_name]["macro_auc_bootstrap_ci95_low"] = low
        summary["models"][model_name]["macro_auc_bootstrap_ci95_high"] = high
    summary["feature_fusion_macro_auc_difference_ci95"] = {
        comparator: {"low": low, "high": high}
        for comparator, (low, high) in differences.items()
    }
    summary["outer_folds"] = outer_folds
    summary["outer_repeats"] = outer_repeats
    summary["inner_folds"] = inner_folds
    summary["parameter_grid"] = parameter_grid
    summary["participant_level_aggregation"] = "mean probability across outer repeats"
    summary_path = output_dir / "classical_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    write_manifest(
        project_root / "outputs" / "manifests" / "classical_prediction.json",
        stage="classical_prediction",
        project_root=project_root,
        config_path=config_path,
        inputs=[
            project_root / "outputs" / "mechanistic" / "regional_features.csv",
            project_root / "outputs" / "reproduction" / "regional_complexity.csv",
        ],
        outputs=[splits_path, prediction_path, tuning_path, summary_path],
        extra={
            "outer_folds": outer_folds,
            "outer_repeats": outer_repeats,
            "inner_folds": inner_folds,
            "participant_count": int(table.shape[0]),
        },
    )
    return splits_path, prediction_path, tuning_path, summary_path
