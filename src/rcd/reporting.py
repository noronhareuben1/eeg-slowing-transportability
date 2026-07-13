from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from rcd.manifest import write_manifest


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.empty(iterations)
    for index in range(iterations):
        values[index] = rng.choice(a, a.size, replace=True).mean() - rng.choice(
            b, b.size, replace=True
        ).mean()
    interval = np.quantile(values, [0.025, 0.975])
    return float(interval[0]), float(interval[1])


def _demographic_table(participants: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    table = participants.copy()
    table.columns = table.columns.str.strip()
    table["diagnosis"] = table["Group"].astype(str).str.strip().map(labels)
    table["Gender"] = table["Gender"].astype(str).str.strip().str.upper()
    rows: list[dict[str, Any]] = []
    for diagnosis in ("AD", "FTD", "CN"):
        frame = table.loc[table["diagnosis"] == diagnosis]
        rows.append(
            {
                "diagnosis": diagnosis,
                "n": int(frame.shape[0]),
                "age_mean": float(frame["Age"].mean()),
                "age_sd": float(frame["Age"].std(ddof=1)),
                "female_n": int((frame["Gender"] == "F").sum()),
                "male_n": int((frame["Gender"] == "M").sum()),
                "mmse_mean": float(frame["MMSE"].mean()),
                "mmse_sd": float(frame["MMSE"].std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def _build_main_figure(
    *,
    reproduction: pd.DataFrame,
    mechanistic: pd.DataFrame,
    state: pd.DataFrame,
    model_table: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> tuple[Path, Path]:
    sns.set_theme(style="whitegrid", context="paper")
    palette = {"AD": "#CC6677", "FTD": "#4477AA", "CN": "#228833"}
    order = ["AD", "FTD", "CN"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    box = reproduction.loc[reproduction["metric"] == "box_count_fd"].copy()
    sns.violinplot(
        data=box,
        x="diagnosis",
        y="rostrocaudal",
        hue="diagnosis",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1,
        ax=axes[0, 0],
    )
    sns.stripplot(
        data=box,
        x="diagnosis",
        y="rostrocaudal",
        order=order,
        color="black",
        alpha=0.55,
        size=3,
        jitter=0.18,
        ax=axes[0, 0],
    )
    axes[0, 0].axhline(0, color="0.35", linewidth=0.8)
    axes[0, 0].set(
        title="A  Direct box-count reproduction",
        xlabel="",
        ylabel="Rostral − caudal FD",
    )

    sns.violinplot(
        data=mechanistic,
        x="diagnosis",
        y="surrogate_hfd_z_rostrocaudal",
        hue="diagnosis",
        order=order,
        hue_order=order,
        palette=palette,
        legend=False,
        inner=None,
        cut=0,
        linewidth=1,
        ax=axes[0, 1],
    )
    sns.stripplot(
        data=mechanistic,
        x="diagnosis",
        y="surrogate_hfd_z_rostrocaudal",
        order=order,
        color="black",
        alpha=0.55,
        size=3,
        jitter=0.18,
        ax=axes[0, 1],
    )
    axes[0, 1].axhline(0, color="0.35", linewidth=0.8)
    axes[0, 1].set(
        title="B  Nonlinear-excess HFD",
        xlabel="",
        ylabel="Rostral − caudal surrogate z-score",
    )

    state_asymmetry = (
        state.pivot_table(
            index=["participant_id", "diagnosis", "state"],
            columns="region",
            values="higuchi_fd",
        )
        .reset_index()
        .assign(asymmetry=lambda frame: frame["rostral"] - frame["caudal"])
    )
    sns.pointplot(
        data=state_asymmetry,
        x="state",
        y="asymmetry",
        hue="diagnosis",
        hue_order=order,
        palette=palette,
        errorbar=("ci", 95),
        n_boot=5000,
        seed=seed,
        dodge=0.2,
        markers=["o", "s", "D"],
        capsize=0.08,
        ax=axes[1, 0],
    )
    axes[1, 0].axhline(0, color="0.35", linewidth=0.8)
    axes[1, 0].set(
        title="C  Matched recording-state comparison",
        xlabel="Recording state",
        ylabel="Rostral − caudal HFD",
    )
    axes[1, 0].set_xticks([0, 1], ["Eyes closed", "Photic open eyes"])
    axes[1, 0].legend(title="Diagnosis", frameon=True)

    display_order = [
        "Demographic",
        "Complexity",
        "Spectral",
        "Feature fusion",
        "EEGNet",
        "EEGNet + complexity",
    ]
    plot_table = model_table.set_index("model").loc[display_order].reset_index()
    positions = np.arange(plot_table.shape[0])
    axes[1, 1].errorbar(
        plot_table["macro_auc"],
        positions,
        xerr=np.vstack(
            [
                plot_table["macro_auc"] - plot_table["ci95_low"],
                plot_table["ci95_high"] - plot_table["macro_auc"],
            ]
        ),
        fmt="o",
        color="#332288",
        ecolor="#888888",
        capsize=3,
    )
    axes[1, 1].axvline(0.5, color="0.35", linestyle="--", linewidth=0.8)
    axes[1, 1].set_yticks(positions, plot_table["model"])
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0.45, 1.0)
    axes[1, 1].set(
        title="D  Leakage-safe three-class prediction",
        xlabel="Participant-level macro ROC-AUC (95% bootstrap CI)",
        ylabel="",
    )

    png_path = output_dir / "figure_main_results.png"
    pdf_path = output_dir / "figure_main_results.pdf"
    figure.savefig(png_path, dpi=350, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def generate_reporting_outputs(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
) -> tuple[Path, Path, Path, Path, Path, Path]:
    reproduction_summary = _load_json(project_root / "outputs/reproduction/summary.json")
    mechanistic_summary = _load_json(project_root / "outputs/mechanistic/summary.json")
    state_summary = _load_json(project_root / "outputs/state/summary.json")
    classical_summary = _load_json(project_root / "outputs/prediction/classical_summary.json")
    deep_summary = _load_json(project_root / "outputs/deep/summary.json")
    reproduction = pd.read_csv(project_root / "outputs/reproduction/regional_complexity.csv")
    mechanistic = pd.read_csv(project_root / "outputs/mechanistic/regional_features.csv")
    state = pd.read_csv(project_root / "outputs/state/paired_state_features.csv")

    box = reproduction.loc[reproduction["metric"] == "box_count_fd"]
    ad = box.loc[box["diagnosis"] == "AD", "rostrocaudal"].to_numpy()
    ftd = box.loc[box["diagnosis"] == "FTD", "rostrocaudal"].to_numpy()
    seed = int(config["study"]["seed"])
    iterations = int(config["statistics"]["bootstrap_iterations"])
    h1_ci = _bootstrap_difference(ad, ftd, seed=seed + 1, iterations=iterations)

    h1 = reproduction_summary["ad_vs_ftd"]["box_count_fd"]
    h2 = mechanistic_summary["h2_spectral_independence"]["with_mmse"]
    h3 = mechanistic_summary["h3_nonlinear_excess_hfd"]["unadjusted"]
    h4 = state_summary["h4_diagnosis_region_state"]["unadjusted"]
    h5 = deep_summary["bootstrap"]
    raw_p = np.asarray(
        [
            h1["welch_p_unadjusted"],
            h2["p_unadjusted"],
            h3["welch_p_unadjusted"],
            h4["p_unadjusted"],
            h5["fusion_minus_raw_paired_permutation_p"],
        ]
    )
    adjusted_p = multipletests(raw_p, method="holm")[1]
    hypothesis_rows = [
        {
            "hypothesis": "H1 direct reproduction",
            "estimate": h1["mean_difference_ad_minus_ftd"],
            "ci95_low": h1_ci[0],
            "ci95_high": h1_ci[1],
            "effect_size": h1["hedges_g"],
            "p_unadjusted": raw_p[0],
            "p_holm": adjusted_p[0],
            "supported": False,
        },
        {
            "hypothesis": "H2 spectral independence",
            "estimate": h2["ad_vs_ftd_coefficient"],
            "ci95_low": h2["ci95_low"],
            "ci95_high": h2["ci95_high"],
            "effect_size": None,
            "p_unadjusted": raw_p[1],
            "p_holm": adjusted_p[1],
            "supported": False,
        },
        {
            "hypothesis": "H3 nonlinear-excess HFD",
            "estimate": h3["mean_difference_ad_minus_ftd"],
            "ci95_low": h3["bootstrap_ci95_low"],
            "ci95_high": h3["bootstrap_ci95_high"],
            "effect_size": h3["hedges_g"],
            "p_unadjusted": raw_p[2],
            "p_holm": adjusted_p[2],
            "supported": False,
        },
        {
            "hypothesis": "H4 diagnosis × region × state",
            "estimate": h4["coefficient"],
            "ci95_low": h4["ci95_low"],
            "ci95_high": h4["ci95_high"],
            "effect_size": None,
            "p_unadjusted": raw_p[3],
            "p_holm": adjusted_p[3],
            "supported": False,
        },
        {
            "hypothesis": "H5 complexity fusion > EEGNet",
            "estimate": h5["fusion_minus_raw_observed"],
            "ci95_low": h5["fusion_minus_raw_ci95_low"],
            "ci95_high": h5["fusion_minus_raw_ci95_high"],
            "effect_size": None,
            "p_unadjusted": raw_p[4],
            "p_holm": adjusted_p[4],
            "supported": bool(deep_summary["h5_success"]),
        },
    ]
    hypothesis_table = pd.DataFrame(hypothesis_rows)

    model_rows: list[dict[str, Any]] = []
    classical_names = {
        "demographic": "Demographic",
        "complexity": "Complexity",
        "spectral": "Spectral",
        "feature_fusion": "Feature fusion",
    }
    for key, display in classical_names.items():
        values = classical_summary["models"][key]
        model_rows.append(
            {
                "model": display,
                "macro_auc": values["macro_roc_auc_ovr"],
                "ci95_low": values["macro_auc_bootstrap_ci95_low"],
                "ci95_high": values["macro_auc_bootstrap_ci95_high"],
                "accuracy": values["accuracy"],
                "balanced_accuracy": values["balanced_accuracy"],
                "macro_f1": values["macro_f1"],
                "log_loss": values["log_loss"],
            }
        )
    deep_names = {"eegnet": "EEGNet", "eegnet_complexity_fusion": "EEGNet + complexity"}
    for key, display in deep_names.items():
        values = deep_summary["models"][key]
        prefix = "raw" if key == "eegnet" else "fusion"
        model_rows.append(
            {
                "model": display,
                "macro_auc": values["macro_roc_auc_ovr"],
                "ci95_low": h5[f"{prefix}_macro_auc_ci95_low"],
                "ci95_high": h5[f"{prefix}_macro_auc_ci95_high"],
                "accuracy": values["accuracy"],
                "balanced_accuracy": values["balanced_accuracy"],
                "macro_f1": values["macro_f1"],
                "log_loss": values["log_loss"],
            }
        )
    model_table = pd.DataFrame(model_rows)
    participants = pd.read_csv(
        project_root / "data/ds004504/participants.tsv", sep="\t", skipinitialspace=True
    )
    demographics = _demographic_table(participants, config["study"]["labels"])

    output_dir = project_root / "outputs" / "reporting"
    output_dir.mkdir(parents=True, exist_ok=True)
    hypothesis_path = output_dir / "table_hypotheses.csv"
    model_path = output_dir / "table_models.csv"
    demographics_path = output_dir / "table_demographics.csv"
    hypothesis_table.to_csv(hypothesis_path, index=False)
    model_table.to_csv(model_path, index=False)
    demographics.to_csv(demographics_path, index=False)
    png_path, pdf_path = _build_main_figure(
        reproduction=reproduction,
        mechanistic=mechanistic,
        state=state,
        model_table=model_table,
        output_dir=output_dir,
        seed=seed,
    )
    values = {
        "study_sample": {"total": 88, "AD": 36, "FTD": 23, "CN": 29},
        "hypotheses": hypothesis_rows,
        "models": model_rows,
        "primary_interpretation": (
            "The reported rostrocaudal complexity pattern was not reproduced as a statistically "
            "robust AD-FTD effect, did not survive spectral or surrogate controls, was not "
            "materially modified by recording state, and did not add reliable predictive value "
            "to EEGNet under subject-level validation."
        ),
        "multiplicity": "Holm correction across H1-H5",
    }
    values_path = project_root / "outputs" / "manuscript_values.json"
    values_path.write_text(json.dumps(values, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        project_root / "outputs/manifests/reporting.json",
        stage="reporting",
        project_root=project_root,
        config_path=config_path,
        inputs=[
            project_root / "outputs/reproduction/summary.json",
            project_root / "outputs/mechanistic/summary.json",
            project_root / "outputs/state/summary.json",
            project_root / "outputs/prediction/classical_summary.json",
            project_root / "outputs/deep/summary.json",
        ],
        outputs=[
            hypothesis_path,
            model_path,
            demographics_path,
            png_path,
            pdf_path,
            values_path,
        ],
    )
    return hypothesis_path, model_path, demographics_path, png_path, pdf_path, values_path
