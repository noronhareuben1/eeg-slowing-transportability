from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import antropy as ant
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

from rcd.data import fixed_segment, load_participants, load_preprocessed_raw, set_path
from rcd.download import iter_expected_set_files
from rcd.manifest import write_manifest
from rcd.spectra import compute_spectral_features
from rcd.surrogates import iaaft, surrogate_zscore


def _stable_seed(seed: int, participant_id: str, channel: str) -> int:
    digest = hashlib.sha256(f"{seed}:{participant_id}:{channel}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _channel_features(
    values: np.ndarray,
    *,
    channel: str,
    participant_id: str,
    sfreq: float,
    config: dict[str, Any],
    n_surrogates: int,
) -> dict[str, float | str]:
    spectral_config = config["spectra"]
    complexity_config = config["complexity"]
    peak_width_limits = spectral_config["peak_width_limits"]
    spectral = compute_spectral_features(
        values,
        sfreq,
        peak_width_limits=(float(peak_width_limits[0]), float(peak_width_limits[1])),
        max_n_peaks=int(spectral_config["max_n_peaks"]),
        min_peak_height=float(spectral_config["min_peak_height"]),
    )
    observed_hfd = float(
        ant.higuchi_fd(values, kmax=int(complexity_config["hfd_kmax_primary"]))
    )
    rng = np.random.default_rng(
        _stable_seed(int(config["study"]["seed"]), participant_id, channel)
    )
    surrogate_hfd = np.asarray(
        [
            ant.higuchi_fd(
                iaaft(
                    values,
                    rng=rng,
                    max_iterations=int(complexity_config["iaaft_max_iterations"]),
                    tolerance=float(complexity_config["iaaft_tolerance"]),
                ),
                kmax=int(complexity_config["hfd_kmax_primary"]),
            )
            for _ in range(n_surrogates)
        ],
        dtype=float,
    )
    return {
        "channel": channel,
        **asdict(spectral),
        "higuchi_fd": observed_hfd,
        "surrogate_hfd_mean": float(surrogate_hfd.mean()),
        "surrogate_hfd_sd": float(surrogate_hfd.std(ddof=1)),
        "surrogate_hfd_z": surrogate_zscore(observed_hfd, surrogate_hfd),
    }


def _participant_features(
    row: pd.Series,
    *,
    dataset_root: Path,
    task: str,
    derivative_pipeline: str | None,
    config: dict[str, Any],
    n_surrogates: int,
    n_jobs: int,
) -> list[dict[str, Any]]:
    preprocessing = config["preprocessing"]
    channels = config["study"]["channels"]
    raw = load_preprocessed_raw(
        set_path(dataset_root, row["participant_id"], task, derivative_pipeline),
        channels["canonical"],
        l_freq=float(preprocessing["l_freq"]),
        h_freq=float(preprocessing["h_freq"]),
        resample_hz=float(preprocessing["resample_hz"]),
        rereference=str(preprocessing["rereference"]),
    )
    segment = fixed_segment(raw, float(preprocessing["reproduction_epoch_seconds"]))
    nested = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_channel_features)(
            values,
            channel=channel,
            participant_id=row["participant_id"],
            sfreq=float(raw.info["sfreq"]),
            config=config,
            n_surrogates=n_surrogates,
        )
        for channel, values in zip(raw.ch_names, segment, strict=True)
    )
    metadata = {
        "participant_id": row["participant_id"],
        "diagnosis": row["diagnosis"],
        "Group": row["Group"],
        "Age": float(row["Age"]),
        "Gender": row["Gender"],
        "MMSE": float(row["MMSE"]),
    }
    return [{**metadata, **record} for record in nested]


def _regional_table(channel_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    channel_groups = config["study"]["channels"]
    metadata = ["participant_id", "diagnosis", "Group", "Age", "Gender", "MMSE"]
    value_columns = [
        column
        for column in channel_table.columns
        if column not in {*metadata, "channel"}
    ]
    records: list[dict[str, Any]] = []
    for keys, frame in channel_table.groupby(metadata, dropna=False, sort=False):
        record: dict[str, Any] = dict(zip(metadata, keys, strict=True))
        indexed = frame.set_index("channel")
        for column in value_columns:
            values = indexed[column]
            rostral = float(values.loc[channel_groups["rostral"]].mean())
            caudal = float(values.loc[channel_groups["caudal"]].mean())
            record[f"{column}_global"] = float(values.mean())
            record[f"{column}_rostral"] = rostral
            record[f"{column}_caudal"] = caudal
            record[f"{column}_rostrocaudal"] = rostral - caudal
        records.append(record)
    return pd.DataFrame(records)


def _bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
    iterations: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations, dtype=float)
    for index in range(iterations):
        differences[index] = rng.choice(a, size=a.size, replace=True).mean() - rng.choice(
            b, size=b.size, replace=True
        ).mean()
    low, high = np.quantile(differences, [0.025, 0.975])
    return float(low), float(high)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt(
        ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
        / (a.size + b.size - 2)
    )
    correction = 1 - 3 / (4 * (a.size + b.size) - 9)
    return float(correction * (a.mean() - b.mean()) / pooled)


def _contrast(
    regional: pd.DataFrame,
    outcome: str,
    *,
    seed: int,
    bootstrap_iterations: int,
) -> dict[str, float | int]:
    ad = regional.loc[regional["diagnosis"] == "AD", outcome].to_numpy(dtype=float)
    ftd = regional.loc[regional["diagnosis"] == "FTD", outcome].to_numpy(dtype=float)
    ad, ftd = ad[np.isfinite(ad)], ftd[np.isfinite(ftd)]
    test = stats.ttest_ind(ad, ftd, equal_var=False)
    ci_low, ci_high = _bootstrap_mean_difference(
        ad,
        ftd,
        seed=seed,
        iterations=bootstrap_iterations,
    )
    return {
        "n_ad": int(ad.size),
        "n_ftd": int(ftd.size),
        "mean_ad": float(ad.mean()),
        "mean_ftd": float(ftd.mean()),
        "mean_difference_ad_minus_ftd": float(ad.mean() - ftd.mean()),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "hedges_g": _hedges_g(ad, ftd),
        "welch_t": float(test.statistic),
        "welch_p_unadjusted": float(test.pvalue),
    }


def _adjusted_model(
    table: pd.DataFrame,
    outcome: str,
    covariates: list[str],
) -> dict[str, Any]:
    subset = table.loc[table["diagnosis"].isin(["AD", "FTD"])].copy()
    subset["AD_vs_FTD"] = (subset["diagnosis"] == "AD").astype(float)
    subset["male"] = (subset["Gender"].str.upper() == "M").astype(float)
    columns = [outcome, "AD_vs_FTD", "male", *covariates]
    subset = subset[columns].replace([np.inf, -np.inf], np.nan).dropna()
    continuous = [column for column in covariates if column not in {"male"}]
    for column in continuous:
        standard_deviation = subset[column].std(ddof=1)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            raise ValueError(f"Cannot standardize {column}")
        subset[column] = (subset[column] - subset[column].mean()) / standard_deviation
    predictors = ["AD_vs_FTD", "male", *covariates]
    predictors = list(dict.fromkeys(predictors))
    design = sm.add_constant(subset[predictors], has_constant="add")
    fit = sm.OLS(subset[outcome], design).fit(cov_type="HC3")
    ci = fit.conf_int().loc["AD_vs_FTD"]
    return {
        "n": int(subset.shape[0]),
        "covariates": predictors[1:],
        "ad_vs_ftd_coefficient": float(fit.params["AD_vs_FTD"]),
        "hc3_standard_error": float(fit.bse["AD_vs_FTD"]),
        "ci95_low": float(ci.iloc[0]),
        "ci95_high": float(ci.iloc[1]),
        "t": float(fit.tvalues["AD_vs_FTD"]),
        "p_unadjusted": float(fit.pvalues["AD_vs_FTD"]),
        "adjusted_r_squared": float(fit.rsquared_adj),
        "condition_number": float(fit.condition_number),
    }


def run_mechanistic(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    n_surrogates: int,
    n_jobs: int,
) -> tuple[Path, Path, Path]:
    dataset_id = "ds004504"
    dataset_root = project_root / "data" / dataset_id
    settings = config["datasets"][dataset_id]
    participants = load_participants(dataset_root, config["study"]["labels"])
    nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_participant_features)(
            row,
            dataset_root=dataset_root,
            task=settings["task"],
            derivative_pipeline=settings.get("derivative_pipeline"),
            config=config,
            n_surrogates=n_surrogates,
            n_jobs=1,
        )
        for _, row in participants.iterrows()
    )
    records = [record for participant in nested for record in participant]
    channel_table = pd.DataFrame(records)
    regional_table = _regional_table(channel_table, config)
    reproduction = pd.read_csv(
        project_root / "outputs" / "reproduction" / "regional_complexity.csv"
    )
    box_count = reproduction.loc[
        reproduction["metric"] == "box_count_fd",
        ["participant_id", "rostrocaudal"],
    ].rename(columns={"rostrocaudal": "box_count_fd_rostrocaudal"})
    regional_table = regional_table.merge(box_count, on="participant_id", validate="one_to_one")

    spectral_covariates = [
        "Age",
        "aperiodic_exponent_rostrocaudal",
        "aperiodic_offset_rostrocaudal",
        "delta_relative_rostrocaudal",
        "theta_relative_rostrocaudal",
        "alpha_relative_rostrocaudal",
    ]
    iterations = int(config["statistics"]["bootstrap_iterations"])
    seed = int(config["study"]["seed"])
    summary = {
        "n_participants": int(regional_table["participant_id"].nunique()),
        "n_surrogates_per_channel": n_surrogates,
        "h2_spectral_independence": {
            "without_mmse": _adjusted_model(
                regional_table,
                "box_count_fd_rostrocaudal",
                spectral_covariates,
            ),
            "with_mmse": _adjusted_model(
                regional_table,
                "box_count_fd_rostrocaudal",
                [*spectral_covariates, "MMSE"],
            ),
        },
        "h3_nonlinear_excess_hfd": {
            "unadjusted": _contrast(
                regional_table,
                "surrogate_hfd_z_rostrocaudal",
                seed=seed + 3,
                bootstrap_iterations=iterations,
            ),
            "age_sex_mmse_adjusted": _adjusted_model(
                regional_table,
                "surrogate_hfd_z_rostrocaudal",
                ["Age", "MMSE"],
            ),
        },
    }

    output_dir = project_root / "outputs" / "mechanistic"
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_path = output_dir / "channel_features.csv"
    regional_path = output_dir / "regional_features.csv"
    summary_path = output_dir / "summary.json"
    channel_table.to_csv(channel_path, index=False)
    regional_table.to_csv(regional_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    inputs = [
        dataset_root / "participants.tsv",
        project_root / "outputs" / "reproduction" / "regional_complexity.csv",
        *iter_expected_set_files(dataset_root),
    ]
    write_manifest(
        project_root / "outputs" / "manifests" / "mechanistic.json",
        stage="mechanistic",
        project_root=project_root,
        config_path=config_path,
        inputs=inputs,
        outputs=[channel_path, regional_path, summary_path],
        extra={
            "n_participants": int(participants.shape[0]),
            "n_surrogates_per_channel": n_surrogates,
            "participant_workers": n_jobs,
        },
    )
    return channel_path, regional_path, summary_path
