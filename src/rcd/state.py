from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import antropy as ant
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed

from rcd.data import (
    events_path,
    fixed_segment,
    load_participants,
    load_preprocessed_raw,
    photic_open_intervals,
    set_path,
)
from rcd.download import iter_expected_set_files
from rcd.manifest import write_manifest


def _regional_hfd(
    values: np.ndarray,
    channel_names: list[str],
    config: dict[str, Any],
) -> dict[str, float]:
    hfd = {
        channel: float(
            ant.higuchi_fd(signal, kmax=int(config["complexity"]["hfd_kmax_primary"]))
        )
        for channel, signal in zip(channel_names, values, strict=True)
    }
    groups = config["study"]["channels"]
    return {
        "rostral": float(np.mean([hfd[channel] for channel in groups["rostral"]])),
        "caudal": float(np.mean([hfd[channel] for channel in groups["caudal"]])),
    }


def _participant_state_features(
    row: pd.Series,
    *,
    eyes_root: Path,
    photic_root: Path,
    config: dict[str, Any],
    interval_seconds: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    preprocessing = config["preprocessing"]
    channels = config["study"]["channels"]["canonical"]
    eyes_settings = config["datasets"]["ds004504"]
    photic_settings = config["datasets"]["ds006036"]
    metadata = {
        "participant_id": row["participant_id"],
        "diagnosis": row["diagnosis"],
        "Group": row["Group"],
        "Age": float(row["Age"]),
        "Gender": row["Gender"],
        "MMSE": float(row["MMSE"]),
    }

    eyes_raw = load_preprocessed_raw(
        set_path(
            eyes_root,
            row["participant_id"],
            eyes_settings["task"],
            eyes_settings.get("derivative_pipeline"),
        ),
        channels,
        l_freq=float(preprocessing["l_freq"]),
        h_freq=float(preprocessing["h_freq"]),
        resample_hz=float(preprocessing["resample_hz"]),
        rereference=str(preprocessing["rereference"]),
    )
    eyes_regions = _regional_hfd(
        fixed_segment(eyes_raw, interval_seconds),
        eyes_raw.ch_names,
        config,
    )
    state_records = [
        {**metadata, "state": "eyes_closed", "region": region, "higuchi_fd": value}
        for region, value in eyes_regions.items()
    ]

    event_file = events_path(photic_root, row["participant_id"], photic_settings["task"])
    intervals = photic_open_intervals(event_file)
    intervals = intervals.loc[
        intervals["frequency_hz"].notna() & (intervals["duration"] >= interval_seconds)
    ].copy()
    if intervals.empty:
        return state_records, []
    photic_raw = load_preprocessed_raw(
        set_path(
            photic_root,
            row["participant_id"],
            photic_settings["task"],
            photic_settings.get("derivative_pipeline"),
        ),
        channels,
        l_freq=float(preprocessing["l_freq"]),
        h_freq=float(preprocessing["h_freq"]),
        resample_hz=float(preprocessing["resample_hz"]),
        rereference=str(preprocessing["rereference"]),
    )
    interval_records: list[dict[str, Any]] = []
    for interval in intervals.itertuples(index=False):
        if float(interval.onset) + interval_seconds > photic_raw.times[-1]:
            continue
        segment = fixed_segment(photic_raw, interval_seconds, float(interval.onset))
        regions = _regional_hfd(segment, photic_raw.ch_names, config)
        for region, value in regions.items():
            interval_records.append(
                {
                    **metadata,
                    "frequency_hz": float(interval.frequency_hz),
                    "region": region,
                    "higuchi_fd": value,
                }
            )
    if not interval_records:
        return state_records, []
    interval_table = pd.DataFrame(interval_records)
    frequency_table = (
        interval_table.groupby(
            [
                "participant_id",
                "diagnosis",
                "Group",
                "Age",
                "Gender",
                "MMSE",
                "frequency_hz",
                "region",
            ],
            as_index=False,
        )["higuchi_fd"]
        .mean()
        .to_dict("records")
    )
    aggregate = interval_table.groupby("region", as_index=False)["higuchi_fd"].mean()
    state_records.extend(
        [
            {
                **metadata,
                "state": "photic_open",
                "region": record.region,
                "higuchi_fd": float(record.higuchi_fd),
            }
            for record in aggregate.itertuples(index=False)
        ]
    )
    return state_records, frequency_table


def _prepare_model_table(table: pd.DataFrame) -> pd.DataFrame:
    subset = table.loc[table["diagnosis"].isin(["AD", "FTD"])].copy()
    subset["AD_vs_FTD"] = (subset["diagnosis"] == "AD").astype(float)
    subset["rostral"] = (subset["region"] == "rostral").astype(float)
    subset["male"] = (subset["Gender"].str.upper() == "M").astype(float)
    for column in ("Age", "MMSE"):
        subset[f"{column}_z"] = (subset[column] - subset[column].mean()) / subset[column].std(
            ddof=1
        )
    return subset


def _mixed_model_summary(
    table: pd.DataFrame,
    *,
    formula: str,
    term: str,
) -> dict[str, Any]:
    model = smf.mixedlm(
        formula,
        table,
        groups=table["participant_id"],
        re_formula="1",
    )
    fit = None
    optimizer = None
    errors: list[str] = []
    for method in ("lbfgs", "powell"):
        try:
            candidate = model.fit(reml=False, method=method, maxiter=1000, disp=False)
            if np.isfinite(candidate.params[term]) and np.isfinite(candidate.bse[term]):
                fit = candidate
                optimizer = method
                break
        except (np.linalg.LinAlgError, ValueError) as error:
            errors.append(f"{method}: {type(error).__name__}: {error}")
    if fit is None:
        fit = smf.gee(
            formula,
            groups="participant_id",
            data=table,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        confidence = fit.conf_int().loc[term]
        return {
            "method": "Gaussian GEE with exchangeable participant correlation",
            "mixed_model_errors": errors,
            "n_participants": int(table["participant_id"].nunique()),
            "n_observations": int(table.shape[0]),
            "formula": formula,
            "term": term,
            "coefficient": float(fit.params[term]),
            "standard_error": float(fit.bse[term]),
            "ci95_low": float(confidence.iloc[0]),
            "ci95_high": float(confidence.iloc[1]),
            "z": float(fit.tvalues[term]),
            "p_unadjusted": float(fit.pvalues[term]),
            "converged": bool(fit.converged),
            "exchangeable_correlation": float(fit.cov_struct.dep_params),
        }
    confidence = fit.conf_int().loc[term]
    return {
        "method": f"random-intercept mixed model ({optimizer})",
        "mixed_model_errors": errors,
        "n_participants": int(table["participant_id"].nunique()),
        "n_observations": int(table.shape[0]),
        "formula": formula,
        "term": term,
        "coefficient": float(fit.params[term]),
        "standard_error": float(fit.bse[term]),
        "ci95_low": float(confidence.iloc[0]),
        "ci95_high": float(confidence.iloc[1]),
        "z": float(fit.tvalues[term]),
        "p_unadjusted": float(fit.pvalues[term]),
        "converged": bool(fit.converged),
        "random_intercept_variance": float(fit.cov_re.iloc[0, 0]),
    }


def _frequency_model_summary(table: pd.DataFrame) -> dict[str, Any]:
    common = table.loc[table["frequency_hz"].isin([5.0, 10.0, 15.0, 20.0])].copy()
    common = _prepare_model_table(common)
    formula = (
        "higuchi_fd ~ AD_vs_FTD * rostral * C(frequency_hz, Treatment(reference=5.0)) "
        "+ Age_z + male + MMSE_z"
    )
    model = smf.mixedlm(
        formula,
        common,
        groups=common["participant_id"],
        re_formula="1",
    )
    fit = None
    method_label = None
    errors: list[str] = []
    for method in ("lbfgs", "powell"):
        try:
            candidate = model.fit(reml=False, method=method, maxiter=1000, disp=False)
            if np.isfinite(candidate.params).all() and np.isfinite(candidate.bse).all():
                fit = candidate
                method_label = f"random-intercept mixed model ({method})"
                break
        except (np.linalg.LinAlgError, ValueError) as error:
            errors.append(f"{method}: {type(error).__name__}: {error}")
    if fit is None:
        fit = smf.gee(
            formula,
            groups="participant_id",
            data=common,
            family=sm.families.Gaussian(),
            cov_struct=sm.cov_struct.Exchangeable(),
        ).fit()
        method_label = "Gaussian GEE with exchangeable participant correlation"
    terms = [
        term
        for term in fit.params.index
        if "AD_vs_FTD:rostral:C(frequency_hz" in term
    ]
    coefficients: dict[str, Any] = {}
    confidence = fit.conf_int()
    for term in terms:
        coefficients[term] = {
            "coefficient": float(fit.params[term]),
            "standard_error": float(fit.bse[term]),
            "ci95_low": float(confidence.loc[term].iloc[0]),
            "ci95_high": float(confidence.loc[term].iloc[1]),
            "z": float(fit.tvalues[term]),
            "p_unadjusted": float(fit.pvalues[term]),
        }
    return {
        "method": method_label,
        "mixed_model_errors": errors,
        "n_participants": int(common["participant_id"].nunique()),
        "n_observations": int(common.shape[0]),
        "frequencies_hz": [5.0, 10.0, 15.0, 20.0],
        "formula": formula,
        "converged": bool(fit.converged),
        "frequency_specific_three_way_terms_vs_5hz": coefficients,
    }


def run_state_analysis(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    interval_seconds: float,
    n_jobs: int,
) -> tuple[Path, Path, Path]:
    eyes_root = project_root / "data" / "ds004504"
    photic_root = project_root / "data" / "ds006036"
    participants = load_participants(eyes_root, config["study"]["labels"])
    nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_participant_state_features)(
            row,
            eyes_root=eyes_root,
            photic_root=photic_root,
            config=config,
            interval_seconds=interval_seconds,
        )
        for _, row in participants.iterrows()
    )
    state_table = pd.DataFrame([record for states, _ in nested for record in states])
    frequency_table = pd.DataFrame([record for _, frequencies in nested for record in frequencies])
    paired_ids = set(
        state_table.loc[state_table["state"] == "photic_open", "participant_id"].unique()
    )
    paired = state_table.loc[state_table["participant_id"].isin(paired_ids)].copy()
    paired["photic"] = (paired["state"] == "photic_open").astype(float)
    model_table = _prepare_model_table(paired)
    triple_term = "AD_vs_FTD:rostral:photic"
    unadjusted_formula = "higuchi_fd ~ AD_vs_FTD * rostral * photic"
    adjusted_formula = unadjusted_formula + " + Age_z + male + MMSE_z"
    summary = {
        "interval_seconds": interval_seconds,
        "paired_participants_all_diagnoses": len(paired_ids),
        "participants_without_valid_photic_interval": sorted(
            set(participants["participant_id"]) - paired_ids
        ),
        "group_means": paired.groupby(["diagnosis", "state", "region"])["higuchi_fd"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .to_dict("records"),
        "h4_diagnosis_region_state": {
            "unadjusted": _mixed_model_summary(
                model_table,
                formula=unadjusted_formula,
                term=triple_term,
            ),
            "age_sex_mmse_adjusted": _mixed_model_summary(
                model_table,
                formula=adjusted_formula,
                term=triple_term,
            ),
        },
        "frequency_sensitivity": _frequency_model_summary(frequency_table),
    }

    output_dir = project_root / "outputs" / "state"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "paired_state_features.csv"
    frequency_path = output_dir / "photic_frequency_features.csv"
    summary_path = output_dir / "summary.json"
    state_table.to_csv(state_path, index=False)
    frequency_table.to_csv(frequency_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    event_inputs = sorted(photic_root.glob("sub-*/eeg/*_events.tsv"))
    inputs = [
        eyes_root / "participants.tsv",
        photic_root / "participants.tsv",
        *iter_expected_set_files(eyes_root),
        *iter_expected_set_files(photic_root),
        *event_inputs,
    ]
    write_manifest(
        project_root / "outputs" / "manifests" / "state.json",
        stage="state",
        project_root=project_root,
        config_path=config_path,
        inputs=inputs,
        outputs=[state_path, frequency_path, summary_path],
        extra={
            "interval_seconds": interval_seconds,
            "participant_workers": n_jobs,
            "paired_participants": len(paired_ids),
        },
    )
    return state_path, frequency_path, summary_path
