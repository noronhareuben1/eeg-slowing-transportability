"""Run the compact spectral/complexity amendment without participant leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = np.asarray(["AD", "CN", "FTD"])
BASELINE = [
    "posterior_delta_alpha_ratio",
    "posterior_alpha_relative",
    "global_aperiodic_exponent",
]
SPECTRAL_ADDITIONS = [
    "theta_relative_global",
    "median_frequency_hz_global",
    "aperiodic_offset_global",
    "alpha_relative_rostrocaudal",
    "aperiodic_exponent_rostrocaudal",
]
COMPLEXITY_ADDITIONS = [
    "higuchi_fd_global",
    "higuchi_fd_rostrocaudal",
    "surrogate_hfd_z_global",
    "surrogate_hfd_z_rostrocaudal",
    "box_count_fd_rostrocaudal",
]
EXTERNAL_SPECTRAL_ADDITIONS = ["global_delta_relative", "global_alpha_relative"]
MODEL_ORDER = ["locked_baseline", "spectral", "complexity", "combined"]


def assemble_internal_features(project_root: Path) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    baseline = pd.read_csv(project_root / "outputs/transportability/derivation_features.csv")
    mechanistic = pd.read_csv(project_root / "outputs/mechanistic/regional_features.csv")
    needed = ["participant_id", *SPECTRAL_ADDITIONS, *COMPLEXITY_ADDITIONS]
    table = baseline.merge(
        mechanistic[needed],
        on="participant_id",
        how="inner",
        validate="one_to_one",
    )
    feature_sets = {
        "locked_baseline": BASELINE,
        "spectral": [*BASELINE, *SPECTRAL_ADDITIONS],
        "complexity": [*BASELINE, *COMPLEXITY_ADDITIONS],
        "combined": [*BASELINE, *SPECTRAL_ADDITIONS, *COMPLEXITY_ADDITIONS],
    }
    required = sorted({column for columns in feature_sets.values() for column in columns})
    table = table.dropna(subset=required).sort_values("participant_id").reset_index(drop=True)
    if table["participant_id"].duplicated().any():
        raise ValueError("internal feature table contains duplicate participants")
    return table, feature_sets


def multiclass_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    probability = probability / probability.sum(axis=1, keepdims=True)
    predicted = CLASSES[np.argmax(probability, axis=1)]
    per_class = {
        label: float(roc_auc_score(y_true == label, probability[:, index]))
        for index, label in enumerate(CLASSES)
    }
    return {
        "macro_roc_auc_ovr": float(
            roc_auc_score(
                y_true,
                probability,
                labels=CLASSES,
                multi_class="ovr",
                average="macro",
            )
        ),
        "per_class_roc_auc_ovr": per_class,
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "macro_f1": float(f1_score(y_true, predicted, labels=CLASSES, average="macro")),
        "log_loss": float(log_loss(y_true, probability, labels=CLASSES)),
        "confusion_matrix_rows_true": confusion_matrix(
            y_true, predicted, labels=CLASSES, normalize="true"
        ).tolist(),
    }


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(y == label), size=int((y == label).sum()), replace=True)
            for label in np.unique(y)
        ]
    )


def bootstrap_multiclass_auc(
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    rng = np.random.default_rng(seed)
    values = {name: np.empty(iterations) for name in probabilities}
    for iteration in range(iterations):
        indices = _stratified_indices(y_true, rng)
        for name, probability in probabilities.items():
            values[name][iteration] = roc_auc_score(
                y_true[indices],
                probability[indices],
                labels=CLASSES,
                multi_class="ovr",
                average="macro",
            )
    intervals = {
        name: np.quantile(samples, [0.025, 0.975]).astype(float).tolist()
        for name, samples in values.items()
    }
    baseline = values["locked_baseline"]
    differences = {
        name: np.quantile(samples - baseline, [0.025, 0.975]).astype(float).tolist()
        for name, samples in values.items()
        if name != "locked_baseline"
    }
    return intervals, differences


def run_internal_analysis(
    project_root: Path,
    *,
    outer_repeats: int,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    table, feature_sets = assemble_internal_features(project_root)
    y = table["diagnosis"].to_numpy()
    prediction_rows: list[dict[str, Any]] = []
    tuning_rows: list[dict[str, Any]] = []
    frozen_splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for repeat in range(outer_repeats):
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(splitter.split(table, y)):
            train_ids = set(table.iloc[train]["participant_id"])
            test_ids = set(table.iloc[test]["participant_id"])
            if train_ids & test_ids:
                raise RuntimeError("participant leakage detected")
            frozen_splits.append((repeat, fold, train, test))

    for model_name in MODEL_ORDER:
        values = table[feature_sets[model_name]].to_numpy(float)
        for repeat, fold, train, test in frozen_splits:
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            solver="lbfgs",
                            class_weight="balanced",
                            max_iter=3000,
                            random_state=seed + repeat * 100 + fold,
                        ),
                    ),
                ]
            )
            inner = StratifiedKFold(
                n_splits=4,
                shuffle=True,
                random_state=seed + 1000 + repeat * 100 + fold,
            )
            search = GridSearchCV(
                model,
                {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
                scoring="roc_auc_ovr",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise",
            )
            search.fit(values[train], y[train])
            raw = search.predict_proba(values[test])
            probability = np.zeros((len(test), len(CLASSES)), dtype=float)
            for source_index, label in enumerate(search.best_estimator_.classes_):
                probability[:, int(np.flatnonzero(CLASSES == label)[0])] = raw[:, source_index]
            for row_index, participant_index in enumerate(test):
                prediction_rows.append(
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
            tuning_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "fold": fold,
                    "selected_C": float(search.best_params_["classifier__C"]),
                    "inner_macro_auc": float(search.best_score_),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    tuning = pd.DataFrame(tuning_rows)
    probability_columns = [f"probability_{label}" for label in CLASSES]
    aggregated_probabilities: dict[str, np.ndarray] = {}
    participant_tables: list[pd.DataFrame] = []
    summary: dict[str, Any] = {
        "analysis_role": "exploratory internally validated AD/FTD/CN analysis",
        "n_participants": int(len(table)),
        "class_counts": table["diagnosis"].value_counts().sort_index().to_dict(),
        "outer_repeats": outer_repeats,
        "outer_folds": 5,
        "inner_folds": 4,
        "models": {},
        "feature_sets": feature_sets,
    }
    reference_y: np.ndarray | None = None
    for model_name in MODEL_ORDER:
        frame = predictions[predictions["model"] == model_name]
        participant = (
            frame.groupby(["participant_id", "diagnosis"], as_index=False)[probability_columns]
            .mean()
            .sort_values("participant_id")
        )
        participant.insert(0, "model", model_name)
        participant_tables.append(participant)
        current_y = participant["diagnosis"].to_numpy()
        if reference_y is None:
            reference_y = current_y
        elif not np.array_equal(reference_y, current_y):
            raise RuntimeError("models do not share identical participant ordering")
        probability = participant[probability_columns].to_numpy(float)
        aggregated_probabilities[model_name] = probability
        summary["models"][model_name] = multiclass_metrics(current_y, probability)
        summary["models"][model_name]["n_features"] = len(feature_sets[model_name])

    assert reference_y is not None
    intervals, differences = bootstrap_multiclass_auc(
        reference_y,
        aggregated_probabilities,
        iterations=bootstrap_iterations,
        seed=seed + 5000,
    )
    baseline_auc = summary["models"]["locked_baseline"]["macro_roc_auc_ovr"]
    for model_name in MODEL_ORDER:
        summary["models"][model_name]["macro_auc_ci95"] = intervals[model_name]
        summary["models"][model_name]["macro_auc_difference_vs_baseline"] = float(
            summary["models"][model_name]["macro_roc_auc_ovr"] - baseline_auc
        )
        if model_name != "locked_baseline":
            summary["models"][model_name]["macro_auc_difference_ci95"] = differences[
                model_name
            ]
    return summary, pd.concat(participant_tables, ignore_index=True), tuning


def _calibration(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, max_iter=2000).fit(logits, y)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def threshold_at_specificity(y: np.ndarray, probability: np.ndarray, target: float = 0.90) -> float:
    fpr, _, thresholds = roc_curve(y, probability)
    valid = np.flatnonzero(fpr <= 1 - target + 1e-12)
    return float(thresholds[valid[-1]]) if len(valid) else float(thresholds[0])


def _binary_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    prediction = probability >= threshold
    result = {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "sensitivity_at_derivation_threshold": float(
            (prediction & (y == 1)).sum() / max((y == 1).sum(), 1)
        ),
        "specificity_at_derivation_threshold": float(
            ((~prediction) & (y == 0)).sum() / max((y == 0).sum(), 1)
        ),
        "derivation_threshold": threshold,
    }
    result.update(_calibration(y, probability))
    return result


def run_external_analysis(
    project_root: Path,
    *,
    bootstrap_iterations: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    derivation = pd.read_csv(project_root / "outputs/transportability/derivation_features.csv")
    mechanistic = pd.read_csv(project_root / "outputs/mechanistic/regional_features.csv")
    shared = mechanistic[
        ["participant_id", "delta_relative_global", "alpha_relative_global"]
    ].rename(
        columns={
            "delta_relative_global": "global_delta_relative",
            "alpha_relative_global": "global_alpha_relative",
        }
    )
    derivation = derivation.merge(shared, on="participant_id", validate="one_to_one")
    external = pd.read_csv(project_root / "outputs/transportability/external_padic_features.csv")
    derivation = derivation[derivation["diagnosis"].isin(["AD", "CN"])].copy()
    external = external[external["diagnosis"].isin(["AD", "CN"])].copy()
    feature_sets = {
        "locked_baseline": BASELINE,
        "shared_spectral_extension": [*BASELINE, *EXTERNAL_SPECTRAL_ADDITIONS],
    }
    all_features = sorted({feature for features in feature_sets.values() for feature in features})
    derivation = derivation.dropna(subset=all_features).reset_index(drop=True)
    external = external.dropna(subset=all_features).reset_index(drop=True)
    y_train = (derivation["diagnosis"] == "AD").astype(int).to_numpy()
    y_test = (external["diagnosis"] == "AD").astype(int).to_numpy()
    probabilities: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "analysis_role": "exploratory external AD/CN amendment",
        "n_derivation": int(len(derivation)),
        "n_external": int(len(external)),
        "models": {},
        "feature_sets": feature_sets,
    }
    for model_name, columns in feature_sets.items():
        x_train = derivation[columns].to_numpy(float)
        x_test = external[columns].to_numpy(float)
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=seed,
                    ),
                ),
            ]
        )
        selected_c = 1.0
        if model_name != "locked_baseline":
            inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            search = GridSearchCV(
                pipeline,
                {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
                error_score="raise",
            ).fit(x_train, y_train)
            fitted = search.best_estimator_
            selected_c = float(search.best_params_["classifier__C"])
        else:
            fitted = pipeline.fit(x_train, y_train)
        train_probability = fitted.predict_proba(x_train)[:, 1]
        test_probability = fitted.predict_proba(x_test)[:, 1]
        probabilities[model_name] = test_probability
        threshold = threshold_at_specificity(y_train, train_probability)
        summary["models"][model_name] = _binary_metrics(
            y_test, test_probability, threshold=threshold
        )
        summary["models"][model_name]["n_features"] = len(columns)
        summary["models"][model_name]["selected_C"] = selected_c
        for index, probability in enumerate(test_probability):
            rows.append(
                {
                    "model": model_name,
                    "recording_index": external.iloc[index]["recording_index"],
                    "diagnosis": external.iloc[index]["diagnosis"],
                    "probability_AD": float(probability),
                }
            )

    rng = np.random.default_rng(seed + 7000)
    bootstrap = {name: np.empty(bootstrap_iterations) for name in probabilities}
    for iteration in range(bootstrap_iterations):
        indices = _stratified_indices(y_test, rng)
        for model_name, probability in probabilities.items():
            bootstrap[model_name][iteration] = roc_auc_score(
                y_test[indices], probability[indices]
            )
    for model_name, samples in bootstrap.items():
        summary["models"][model_name]["roc_auc_ci95"] = (
            np.quantile(samples, [0.025, 0.975]).astype(float).tolist()
        )
    difference = bootstrap["shared_spectral_extension"] - bootstrap["locked_baseline"]
    summary["paired_auc_difference_extension_minus_baseline"] = float(
        summary["models"]["shared_spectral_extension"]["roc_auc"]
        - summary["models"]["locked_baseline"]["roc_auc"]
    )
    summary["paired_auc_difference_ci95"] = (
        np.quantile(difference, [0.025, 0.975]).astype(float).tolist()
    )
    return summary, pd.DataFrame(rows)


def build_figure(internal: dict[str, Any], external: dict[str, Any], output: Path) -> None:
    labels = {
        "locked_baseline": "Locked baseline",
        "spectral": "+ spectral",
        "complexity": "+ complexity",
        "combined": "+ both",
        "shared_spectral_extension": "+ shared spectral",
    }
    colors = ["#3B6FB6", "#D7852A", "#6E9F58", "#8C5A9E"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for index, model_name in enumerate(MODEL_ORDER):
        model = internal["models"][model_name]
        low, high = model["macro_auc_ci95"]
        axes[0].errorbar(
            model["macro_roc_auc_ovr"],
            index,
            xerr=[[model["macro_roc_auc_ovr"] - low], [high - model["macro_roc_auc_ovr"]]],
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    axes[0].set_yticks(range(len(MODEL_ORDER)), [labels[name] for name in MODEL_ORDER])
    axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    axes[0].set(xlim=(0.45, 0.9), xlabel="Macro ROC-AUC (95% CI)", title="A  Internal AD/FTD/CN")

    external_order = ["locked_baseline", "shared_spectral_extension"]
    for index, model_name in enumerate(external_order):
        model = external["models"][model_name]
        low, high = model["roc_auc_ci95"]
        axes[1].errorbar(
            model["roc_auc"],
            index,
            xerr=[[model["roc_auc"] - low], [high - model["roc_auc"]]],
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    axes[1].set_yticks(range(2), [labels[name] for name in external_order])
    axes[1].invert_yaxis()
    axes[1].axvline(0.5, color="0.5", linestyle="--", linewidth=0.8)
    axes[1].set(xlim=(0.45, 0.9), xlabel="ROC-AUC (95% CI)", title="B  External P-ADIC AD/CN")
    for axis in axes:
        axis.grid(axis="x", color="0.9", linewidth=0.8)
        axis.grid(axis="y", visible=False)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def write_summary_tables(
    internal: dict[str, Any], external: dict[str, Any], output_dir: Path
) -> None:
    internal_rows = []
    for model_name in MODEL_ORDER:
        model = internal["models"][model_name]
        difference_ci = model.get("macro_auc_difference_ci95", [np.nan, np.nan])
        internal_rows.append(
            {
                "model": model_name,
                "n_features": model["n_features"],
                "macro_roc_auc_ovr": model["macro_roc_auc_ovr"],
                "ci95_low": model["macro_auc_ci95"][0],
                "ci95_high": model["macro_auc_ci95"][1],
                "difference_vs_baseline": model["macro_auc_difference_vs_baseline"],
                "difference_ci95_low": difference_ci[0],
                "difference_ci95_high": difference_ci[1],
                "balanced_accuracy": model["balanced_accuracy"],
                "macro_f1": model["macro_f1"],
                "auc_AD": model["per_class_roc_auc_ovr"]["AD"],
                "auc_FTD": model["per_class_roc_auc_ovr"]["FTD"],
                "auc_CN": model["per_class_roc_auc_ovr"]["CN"],
            }
        )
    pd.DataFrame(internal_rows).to_csv(output_dir / "table_internal_models.csv", index=False)

    external_rows = []
    for model_name in ("locked_baseline", "shared_spectral_extension"):
        model = external["models"][model_name]
        external_rows.append(
            {
                "model": model_name,
                "n_features": model["n_features"],
                "roc_auc": model["roc_auc"],
                "ci95_low": model["roc_auc_ci95"][0],
                "ci95_high": model["roc_auc_ci95"][1],
                "average_precision": model["average_precision"],
                "brier": model["brier"],
                "calibration_intercept": model["calibration_intercept"],
                "calibration_slope": model["calibration_slope"],
                "sensitivity_at_derivation_threshold": model[
                    "sensitivity_at_derivation_threshold"
                ],
                "specificity_at_derivation_threshold": model[
                    "specificity_at_derivation_threshold"
                ],
            }
        )
    pd.DataFrame(external_rows).to_csv(output_dir / "table_external_models.csv", index=False)


def write_report(internal: dict[str, Any], external: dict[str, Any], output: Path) -> None:
    best_name = max(
        MODEL_ORDER,
        key=lambda name: internal["models"][name]["macro_roc_auc_ovr"],
    )
    best = internal["models"][best_name]
    baseline = internal["models"]["locked_baseline"]
    external_baseline = external["models"]["locked_baseline"]
    external_extension = external["models"]["shared_spectral_extension"]
    best_difference_ci = best["macro_auc_difference_ci95"]
    extension_specificity = external_extension["specificity_at_derivation_threshold"]
    text = f"""# Compact EEG amendment results

## Main research question

Can a low-parameter, physiologically interpretable EEG model distinguish AD,
FTD, and cognitively normal participants, and do compact spectral or
rostrocaudal-complexity additions improve accuracy without weakening external
AD-versus-control transportability?

## Hypotheses

The working hypothesis was that the locked three-feature slowing model would
provide a transferable baseline, and that compact spectral parameterization and
rostrocaudal complexity features might improve three-class discrimination while
preserving external performance.

## Results

The internally validated three-feature baseline reached macro ROC-AUC
{baseline['macro_roc_auc_ovr']:.3f} (95% CI {baseline['macro_auc_ci95'][0]:.3f} to
{baseline['macro_auc_ci95'][1]:.3f}) across {internal['n_participants']} AHEPA
participants. The highest internal point estimate was **{best_name.replace('_', ' ')}**
with macro ROC-AUC {best['macro_roc_auc_ovr']:.3f} (95% CI
{best['macro_auc_ci95'][0]:.3f} to {best['macro_auc_ci95'][1]:.3f}), a difference
of {best['macro_auc_difference_vs_baseline']:+.3f} (95% CI
{best_difference_ci[0]:+.3f} to {best_difference_ci[1]:+.3f}) from the baseline.
FTD remained the weakest class even in the best model (one-versus-rest AUC
{best['per_class_roc_auc_ovr']['FTD']:.3f}).

In independent P-ADIC AD/CN validation, the locked baseline reached ROC-AUC
{external_baseline['roc_auc']:.3f} (95% CI {external_baseline['roc_auc_ci95'][0]:.3f} to
{external_baseline['roc_auc_ci95'][1]:.3f}). The shared five-feature spectral
extension reached {external_extension['roc_auc']:.3f} (95% CI
{external_extension['roc_auc_ci95'][0]:.3f} to
{external_extension['roc_auc_ci95'][1]:.3f}). The paired AUC difference was
{external['paired_auc_difference_extension_minus_baseline']:+.3f} (95% CI
{external['paired_auc_difference_ci95'][0]:+.3f} to
{external['paired_auc_difference_ci95'][1]:+.3f}). The extension reduced the
Brier score from {external_baseline['brier']:.3f} to {external_extension['brier']:.3f},
but transported specificity remained poor ({extension_specificity:.3f}).

## Interpretation

These are exploratory amendment results. The spectral panel is a reasonable
candidate for a future independently locked test because its internal estimate
improved, but this dataset provides no evidence that it improves external AD/CN
discrimination. The complexity additions did not improve on the spectral-only
model. The external cohort validates AD versus controls only; it does not
externally validate FTD classification. The next confirmatory study requires a
new independent cohort with FTD and must lock the selected panel before testing.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=19)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_dir = root / "outputs/amendment_v1_1"
    output_dir.mkdir(parents=True, exist_ok=True)
    internal, internal_predictions, tuning = run_internal_analysis(
        root,
        outer_repeats=args.outer_repeats,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    external, external_predictions = run_external_analysis(
        root,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
    )
    internal_predictions.to_csv(output_dir / "internal_participant_predictions.csv", index=False)
    tuning.to_csv(output_dir / "internal_tuning.csv", index=False)
    external_predictions.to_csv(output_dir / "external_participant_predictions.csv", index=False)
    (output_dir / "internal_results.json").write_text(
        json.dumps(internal, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "external_results.json").write_text(
        json.dumps(external, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    build_figure(internal, external, output_dir / "figure_amendment_results.png")
    write_summary_tables(internal, external, output_dir)
    write_report(internal, external, root / "docs/amendment_v1_1_results.md")
    print(json.dumps({"internal": internal, "external": external}, indent=2))


if __name__ == "__main__":
    main()
