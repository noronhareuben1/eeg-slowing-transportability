"""Repeated nested validation of a paired resting-plus-photic EEG fingerprint."""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC

from transportability.run_amendment_analysis import CLASSES, multiclass_metrics

METRICS = [
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
    "spectral_entropy",
    "stimulus_snr_db",
    "harmonic_snr_db",
    "higuchi_fd",
]
RESTING_METRICS = [
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
    "higuchi_fd",
    "surrogate_hfd_z",
]
AP = {
    "Fp1": 0.0, "Fp2": 0.0, "F7": 1.0, "F3": 1.0, "Fz": 1.0,
    "F4": 1.0, "F8": 1.0, "T3": 2.0, "C3": 2.0, "Cz": 2.0,
    "C4": 2.0, "T4": 2.0, "T5": 3.0, "P3": 3.0, "Pz": 3.0,
    "P4": 3.0, "T6": 3.0, "O1": 4.0, "O2": 4.0,
}
LR = {
    "Fp1": -1.0, "Fp2": 1.0, "F7": -1.5, "F3": -0.7, "Fz": 0.0,
    "F4": 0.7, "F8": 1.5, "T3": -1.5, "C3": -0.7, "Cz": 0.0,
    "C4": 0.7, "T4": 1.5, "T5": -1.5, "P3": -0.7, "Pz": 0.0,
    "P4": 0.7, "T6": 1.5, "O1": -0.7, "O2": 0.7,
}
ROSTRAL = {"Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8"}
CAUDAL = {"T5", "P3", "Pz", "P4", "T6", "O1", "O2"}
MODELS = ["resting_direct", "paired_direct", "paired_hierarchical", "paired_hybrid"]


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[valid], y[valid], 1)[0]) if valid.sum() >= 2 else np.nan


def build_resting_table(project_root: Path) -> tuple[pd.DataFrame, list[str]]:
    channel = pd.read_csv(project_root / "outputs/mechanistic/channel_features.csv")
    records: list[dict[str, object]] = []
    for (participant_id, diagnosis), frame in channel.groupby(["participant_id", "diagnosis"]):
        record: dict[str, object] = {
            "participant_id": participant_id,
            "diagnosis": diagnosis,
        }
        ap = frame["channel"].map(AP).to_numpy(float)
        lr = frame["channel"].map(LR).to_numpy(float)
        for metric in RESTING_METRICS:
            values = frame[metric].to_numpy(float)
            record[f"resting__{metric}__mean"] = float(np.mean(values))
            record[f"resting__{metric}__ap_slope"] = _slope(ap, values)
            record[f"resting__{metric}__lr_slope"] = _slope(lr, values)
            record[f"resting__{metric}__front_minus_back"] = float(
                np.mean(values[ap <= 1]) - np.mean(values[ap >= 3])
            )
        records.append(record)
    structured = pd.DataFrame(records)
    wide = channel.pivot(
        index=["participant_id", "diagnosis"], columns="channel", values=RESTING_METRICS
    )
    wide.columns = [
        f"resting__{metric}__ch_{channel_name}"
        for metric, channel_name in wide.columns
    ]
    wide = wide.reset_index()
    table = structured.merge(wide, on=["participant_id", "diagnosis"], validate="one_to_one")

    regional = pd.read_csv(project_root / "outputs/mechanistic/regional_features.csv")
    excluded = {"participant_id", "diagnosis", "Group", "Age", "Gender", "MMSE"}
    regional_columns = [column for column in regional.columns if column not in excluded]
    regional = regional[["participant_id", *regional_columns]].rename(
        columns={column: f"resting__regional__{column}" for column in regional_columns}
    )
    table = table.merge(regional, on="participant_id", validate="one_to_one")
    columns = [column for column in table if column not in {"participant_id", "diagnosis"}]
    return table.sort_values("participant_id").reset_index(drop=True), columns


def build_photic_table(project_root: Path) -> tuple[pd.DataFrame, list[str]]:
    long = pd.read_csv(project_root / "outputs/amendment_v1_2/photic_response_features.csv")
    records: list[dict[str, object]] = []
    for (participant_id, diagnosis, frequency), frame in long.groupby(
        ["participant_id", "diagnosis", "frequency_hz"]
    ):
        record: dict[str, object] = {
            "participant_id": participant_id,
            "diagnosis": diagnosis,
            "frequency_hz": frequency,
        }
        ap = frame["channel"].map(AP).to_numpy(float)
        lr = frame["channel"].map(LR).to_numpy(float)
        rostral = frame["channel"].isin(ROSTRAL).to_numpy()
        caudal = frame["channel"].isin(CAUDAL).to_numpy()
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            record[f"{metric}__global"] = float(np.nanmean(values))
            record[f"{metric}__rostral"] = float(np.nanmean(values[rostral]))
            record[f"{metric}__caudal"] = float(np.nanmean(values[caudal]))
            record[f"{metric}__rostrocaudal"] = (
                record[f"{metric}__rostral"] - record[f"{metric}__caudal"]
            )
            record[f"{metric}__ap_slope"] = _slope(ap, values)
            record[f"{metric}__lr_slope"] = _slope(lr, values)
        records.append(record)
    spatial_long = pd.DataFrame(records)
    spatial_metrics = [
        column
        for column in spatial_long
        if column not in {"participant_id", "diagnosis", "frequency_hz"}
    ]
    structured = spatial_long.pivot(
        index=["participant_id", "diagnosis"], columns="frequency_hz", values=spatial_metrics
    )
    structured.columns = [
        f"photic__{metric}__{int(frequency)}hz" for metric, frequency in structured.columns
    ]
    structured = structured.reset_index()
    trends: dict[str, list[float]] = {}
    frequencies = np.asarray([5.0, 10.0, 15.0, 20.0])
    for metric in spatial_metrics:
        columns = [f"photic__{metric}__{int(frequency)}hz" for frequency in frequencies]
        values = structured[columns].to_numpy(float)
        trends[f"photic__{metric}__frequency_mean"] = np.nanmean(values, axis=1).tolist()
        trends[f"photic__{metric}__frequency_sd"] = np.nanstd(values, axis=1).tolist()
        trends[f"photic__{metric}__frequency_slope"] = [
            _slope(frequencies, row) for row in values
        ]
    structured = pd.concat([structured, pd.DataFrame(trends, index=structured.index)], axis=1)

    raw = long.pivot(
        index=["participant_id", "diagnosis"],
        columns=["frequency_hz", "channel"],
        values=METRICS,
    )
    raw.columns = [
        f"photic__{metric}__{int(frequency)}hz__ch_{channel}"
        for metric, frequency, channel in raw.columns
    ]
    raw = raw.reset_index()
    table = structured.merge(raw, on=["participant_id", "diagnosis"], validate="one_to_one")
    columns = [column for column in table if column not in {"participant_id", "diagnosis"}]
    return table, columns


def usable_columns(table: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if table[column].notna().mean() >= 0.60 and table[column].nunique(dropna=True) > 1
    ]


def assemble_analysis_table(
    project_root: Path,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    resting, resting_columns = build_resting_table(project_root)
    photic, photic_columns = build_photic_table(project_root)
    table = resting.merge(photic, on=["participant_id", "diagnosis"], validate="one_to_one")
    resting_columns = usable_columns(table, resting_columns)
    paired_columns = usable_columns(table, [*resting_columns, *photic_columns])
    if table["participant_id"].duplicated().any():
        raise ValueError("paired feature table contains duplicate participants")
    table = table.sort_values("participant_id").reset_index(drop=True)
    return table, resting_columns, paired_columns


def _search(n_features: int, *, binary: bool, seed: int, n_jobs: int) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("classifier", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    )
    ks = sorted({5, 10, 20, min(40, n_features)})
    grid = [
        {
            "scale": [StandardScaler(), RobustScaler()],
            "select__k": ks,
            "classifier": [
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)
            ],
            "classifier__C": [0.01, 0.1, 1.0],
        },
        {
            "scale": [StandardScaler()],
            "select__k": [k for k in ks if k <= 20],
            "classifier": [LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")],
        },
        {
            "scale": [StandardScaler(), RobustScaler()],
            "select__k": ks,
            "classifier": [SVC(probability=True, class_weight="balanced", random_state=seed)],
            "classifier__C": [0.1, 1.0, 10.0],
            "classifier__gamma": ["scale"],
        },
    ]
    return GridSearchCV(
        pipeline,
        grid,
        scoring="roc_auc" if binary else "roc_auc_ovr",
        cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=seed + 1000),
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )


def _aligned_probability(estimator: Any, values: pd.DataFrame) -> np.ndarray:
    raw = estimator.predict_proba(values)
    probability = np.zeros((len(values), len(CLASSES)))
    for source, label in enumerate(estimator.classes_):
        probability[:, int(np.flatnonzero(CLASSES == label)[0])] = raw[:, source]
    return probability


def _binary_probability(estimator: Any, values: pd.DataFrame) -> np.ndarray:
    positive = int(np.flatnonzero(estimator.classes_ == 1)[0])
    return estimator.predict_proba(values)[:, positive]


def _selection_rows(
    estimator: Pipeline,
    columns: list[str],
    *,
    component: str,
    repeat: int,
    fold: int,
) -> list[dict[str, object]]:
    support = estimator.named_steps["select"].get_support()
    return [
        {"component": component, "repeat": repeat, "fold": fold, "feature": feature}
        for feature, selected in zip(columns, support, strict=True)
        if selected
    ]


def _prediction_rows(
    table: pd.DataFrame,
    test: np.ndarray,
    probability: np.ndarray,
    *,
    model: str,
    repeat: int,
    fold: int,
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "repeat": repeat,
            "fold": fold,
            "participant_id": table.iloc[index]["participant_id"],
            "diagnosis": table.iloc[index]["diagnosis"],
            **{
                f"probability_{label}": float(probability[row, class_index])
                for class_index, label in enumerate(CLASSES)
            },
        }
        for row, index in enumerate(test)
    ]


def aggregate_predictions(predictions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    probability_columns = [f"probability_{label}" for label in CLASSES]
    output = {}
    for model in sorted(predictions["model"].unique()):
        output[model] = (
            predictions.loc[predictions["model"] == model]
            .groupby(["participant_id", "diagnosis"], as_index=False)[probability_columns]
            .mean()
            .sort_values("participant_id")
            .reset_index(drop=True)
        )
    return output


def _stratified_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate(
        [
            rng.choice(np.flatnonzero(y == label), size=int((y == label).sum()), replace=True)
            for label in CLASSES
        ]
    )


def _bootstrap(
    y: np.ndarray,
    probabilities: dict[str, np.ndarray],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    macro = {model: np.empty(iterations) for model in probabilities}
    ftd = {model: np.empty(iterations) for model in probabilities}
    for iteration in range(iterations):
        indices = _stratified_indices(y, rng)
        for model, probability in probabilities.items():
            macro[model][iteration] = roc_auc_score(
                y[indices], probability[indices], labels=CLASSES, multi_class="ovr", average="macro"
            )
            ftd[model][iteration] = roc_auc_score(
                y[indices] == "FTD", probability[indices, int(np.flatnonzero(CLASSES == "FTD")[0])]
            )
    baseline = "resting_direct"
    return {
        "macro_auc_ci95": {
            model: np.quantile(values, [0.025, 0.975]).astype(float).tolist()
            for model, values in macro.items()
        },
        "ftd_auc_ci95": {
            model: np.quantile(values, [0.025, 0.975]).astype(float).tolist()
            for model, values in ftd.items()
        },
        "macro_difference_vs_resting_ci95": {
            model: np.quantile(values - macro[baseline], [0.025, 0.975]).astype(float).tolist()
            for model, values in macro.items()
            if model != baseline
        },
        "ftd_difference_vs_resting_ci95": {
            model: np.quantile(values - ftd[baseline], [0.025, 0.975]).astype(float).tolist()
            for model, values in ftd.items()
            if model != baseline
        },
    }


def run_analysis(
    project_root: Path,
    *,
    outer_repeats: int,
    bootstrap_iterations: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    table, resting_columns, paired_columns = assemble_analysis_table(project_root)
    y = table["diagnosis"].to_numpy()
    prediction_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for repeat in range(outer_repeats):
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + repeat)
        for fold, (train, test) in enumerate(outer.split(table, y)):
            train_ids = set(table.iloc[train]["participant_id"])
            test_ids = set(table.iloc[test]["participant_id"])
            if train_ids & test_ids:
                raise RuntimeError("participant leakage detected")
            fold_seed = seed + repeat * 100 + fold
            fitted: dict[str, GridSearchCV] = {}
            for component, columns in (
                ("resting_direct", resting_columns),
                ("paired_direct", paired_columns),
            ):
                fitted[component] = _search(
                    len(columns), binary=False, seed=fold_seed, n_jobs=n_jobs
                ).fit(table.loc[train, columns], y[train])
                estimator = fitted[component].best_estimator_
                probability = _aligned_probability(estimator, table.loc[test, columns])
                prediction_rows.extend(
                    _prediction_rows(
                        table, test, probability, model=component, repeat=repeat, fold=fold
                    )
                )
                selection_rows.extend(
                    _selection_rows(
                        estimator,
                        columns,
                        component=component,
                        repeat=repeat,
                        fold=fold,
                    )
                )

            dementia_train = (y[train] != "CN").astype(int)
            stage1 = _search(
                len(paired_columns), binary=True, seed=fold_seed + 20000, n_jobs=n_jobs
            ).fit(table.loc[train, paired_columns], dementia_train)
            subtype_mask = y[train] != "CN"
            subtype_train = (y[train][subtype_mask] == "AD").astype(int)
            stage2 = _search(
                len(paired_columns), binary=True, seed=fold_seed + 30000, n_jobs=n_jobs
            ).fit(table.loc[train[subtype_mask], paired_columns], subtype_train)
            p_dementia = _binary_probability(
                stage1.best_estimator_, table.loc[test, paired_columns]
            )
            p_ad_given_dementia = _binary_probability(
                stage2.best_estimator_, table.loc[test, paired_columns]
            )
            hierarchical = np.column_stack(
                [
                    p_dementia * p_ad_given_dementia,
                    1.0 - p_dementia,
                    p_dementia * (1.0 - p_ad_given_dementia),
                ]
            )
            direct_rows = [
                row
                for row in prediction_rows
                if row["model"] == "paired_direct"
                and row["repeat"] == repeat
                and row["fold"] == fold
            ]
            direct = np.asarray(
                [[row[f"probability_{label}"] for label in CLASSES] for row in direct_rows],
                dtype=float,
            )
            hybrid = 0.5 * direct + 0.5 * hierarchical
            hybrid /= hybrid.sum(axis=1, keepdims=True)
            prediction_rows.extend(
                _prediction_rows(
                    table,
                    test,
                    hierarchical,
                    model="paired_hierarchical",
                    repeat=repeat,
                    fold=fold,
                )
            )
            prediction_rows.extend(
                _prediction_rows(
                    table, test, hybrid, model="paired_hybrid", repeat=repeat, fold=fold
                )
            )
            for component, search in (
                ("resting_direct", fitted["resting_direct"]),
                ("paired_direct", fitted["paired_direct"]),
                ("dementia_vs_cn", stage1),
                ("ad_vs_ftd", stage2),
            ):
                estimator = search.best_estimator_
                tuning_rows.append(
                    {
                        "component": component,
                        "repeat": repeat,
                        "fold": fold,
                        "classifier": type(estimator.named_steps["classifier"]).__name__,
                        "selected_k": int(estimator.named_steps["select"].k),
                        "inner_auc": float(search.best_score_),
                    }
                )
            selection_rows.extend(
                _selection_rows(
                    stage1.best_estimator_,
                    paired_columns,
                    component="dementia_vs_cn",
                    repeat=repeat,
                    fold=fold,
                )
            )
            selection_rows.extend(
                _selection_rows(
                    stage2.best_estimator_,
                    paired_columns,
                    component="ad_vs_ftd",
                    repeat=repeat,
                    fold=fold,
                )
            )
        print(f"completed repeat {repeat + 1}/{outer_repeats}", flush=True)

    predictions = pd.DataFrame(prediction_rows)
    tuning = pd.DataFrame(tuning_rows)
    selected = pd.DataFrame(selection_rows)
    aggregated = aggregate_predictions(predictions)
    reference = aggregated[MODELS[0]]["diagnosis"].to_numpy()
    probabilities: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {
        "analysis_role": "exploratory repeated nested AD/FTD/CN validation",
        "n_participants": int(len(table)),
        "class_counts": table["diagnosis"].value_counts().sort_index().to_dict(),
        "outer_repeats": outer_repeats,
        "outer_folds": 5,
        "inner_folds": 4,
        "n_resting_features_available": len(resting_columns),
        "n_paired_features_available": len(paired_columns),
        "models": {},
    }
    participant_rows = []
    for model in MODELS:
        participant = aggregated[model]
        if not np.array_equal(reference, participant["diagnosis"].to_numpy()):
            raise RuntimeError("model predictions do not share participant ordering")
        probability = participant[[f"probability_{label}" for label in CLASSES]].to_numpy(float)
        probabilities[model] = probability
        summary["models"][model] = multiclass_metrics(reference, probability)
        participant_rows.append(participant.assign(model=model))
    bootstrap = _bootstrap(
        reference,
        probabilities,
        iterations=bootstrap_iterations,
        seed=seed + 5000,
    )
    baseline = summary["models"]["resting_direct"]
    for model in MODELS:
        values = summary["models"][model]
        values["macro_auc_ci95"] = bootstrap["macro_auc_ci95"][model]
        values["ftd_auc_ci95"] = bootstrap["ftd_auc_ci95"][model]
        values["macro_auc_difference_vs_resting"] = float(
            values["macro_roc_auc_ovr"] - baseline["macro_roc_auc_ovr"]
        )
        values["ftd_auc_difference_vs_resting"] = float(
            values["per_class_roc_auc_ovr"]["FTD"]
            - baseline["per_class_roc_auc_ovr"]["FTD"]
        )
        if model != "resting_direct":
            values["macro_auc_difference_ci95"] = bootstrap[
                "macro_difference_vs_resting_ci95"
            ][model]
            values["ftd_auc_difference_ci95"] = bootstrap[
                "ftd_difference_vs_resting_ci95"
            ][model]
    feature_frequency = (
        selected.groupby(["component", "feature"]).size().rename("selection_count").reset_index()
    )
    denominators = selected[["component", "repeat", "fold"]].drop_duplicates().groupby(
        "component"
    ).size()
    feature_frequency["selection_fraction"] = [
        count / denominators.loc[component]
        for component, count in feature_frequency[["component", "selection_count"]].itertuples(
            index=False, name=None
        )
    ]
    summary["classifier_choice_counts"] = {
        component: dict(Counter(frame["classifier"]))
        for component, frame in tuning.groupby("component")
    }
    return summary, predictions, tuning, feature_frequency


def write_outputs(
    project_root: Path,
    summary: dict[str, Any],
    predictions: pd.DataFrame,
    tuning: pd.DataFrame,
    feature_frequency: pd.DataFrame,
) -> None:
    output = project_root / "outputs" / "amendment_v1_2"
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "outer_predictions.csv", index=False)
    tuning.to_csv(output / "inner_tuning.csv", index=False)
    feature_frequency.sort_values(
        ["component", "selection_fraction", "feature"], ascending=[True, False, True]
    ).to_csv(output / "feature_selection_frequency.csv", index=False)
    (output / "results.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = []
    for model in MODELS:
        values = summary["models"][model]
        rows.append(
            {
                "model": model,
                "macro_auc": values["macro_roc_auc_ovr"],
                "macro_ci95_low": values["macro_auc_ci95"][0],
                "macro_ci95_high": values["macro_auc_ci95"][1],
                "auc_AD": values["per_class_roc_auc_ovr"]["AD"],
                "auc_CN": values["per_class_roc_auc_ovr"]["CN"],
                "auc_FTD": values["per_class_roc_auc_ovr"]["FTD"],
                "balanced_accuracy": values["balanced_accuracy"],
                "macro_f1": values["macro_f1"],
                "macro_difference_vs_resting": values["macro_auc_difference_vs_resting"],
                "ftd_difference_vs_resting": values["ftd_auc_difference_vs_resting"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "table_models.csv", index=False)

    colors = ["#6B7280", "#1D4ED8", "#B45309", "#047857"]
    labels = ["Resting", "Paired direct", "Paired two-stage", "Paired hybrid"]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    for index, row in table.iterrows():
        axes[0].errorbar(
            row["macro_auc"],
            index,
            xerr=[
                [row["macro_auc"] - row["macro_ci95_low"]],
                [row["macro_ci95_high"] - row["macro_auc"]],
            ],
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="Macro ROC-AUC (95% CI)",
        title="A  Overall discrimination",
        xlim=(0.45, 0.9),
    )
    x = np.arange(3)
    width = 0.18
    for index, row in table.iterrows():
        axes[1].bar(
            x + (index - 1.5) * width,
            [row["auc_AD"], row["auc_CN"], row["auc_FTD"]],
            width,
            label=labels[index],
            color=colors[index],
        )
    axes[1].set_xticks(x, ["AD", "CN", "FTD"])
    axes[1].set(
        ylim=(0.35, 1.0),
        ylabel="One-vs-rest ROC-AUC",
        title="B  Class-specific performance",
    )
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        if axis is axes[1]:
            axis.axhline(0.5, color="0.75", linewidth=0.7, linestyle="--")
        else:
            axis.axvline(0.5, color="0.75", linewidth=0.7, linestyle="--")
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(output / "figure_paired_response_results.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--outer-repeats", type=int, default=10)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    root = args.project_root.resolve()
    summary, predictions, tuning, feature_frequency = run_analysis(
        root,
        outer_repeats=args.outer_repeats,
        bootstrap_iterations=args.bootstrap_iterations,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )
    write_outputs(root, summary, predictions, tuning, feature_frequency)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
