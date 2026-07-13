from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from rcd.complexity import compute_complexity, regional_summary
from rcd.data import fixed_segment, load_participants, load_preprocessed_raw, set_path
from rcd.manifest import write_manifest


def _participant_features(
    row: pd.Series,
    *,
    dataset_root: Path,
    task: str,
    derivative_pipeline: str | None,
    config: dict[str, Any],
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
    records: list[dict[str, Any]] = []
    for channel, values in zip(raw.ch_names, segment, strict=True):
        metrics = asdict(
            compute_complexity(values, hfd_kmax=int(config["complexity"]["hfd_kmax_primary"]))
        )
        for metric, value in metrics.items():
            records.append(
                {
                    "participant_id": row["participant_id"],
                    "diagnosis": row["diagnosis"],
                    "Group": row["Group"],
                    "Age": float(row["Age"]),
                    "Gender": row["Gender"],
                    "MMSE": float(row["MMSE"]),
                    "channel": channel,
                    "metric": metric,
                    "value": value,
                }
            )
    return records


def _regional_table(channel_table: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    channel_groups = config["study"]["channels"]
    metadata = ["participant_id", "diagnosis", "Group", "Age", "Gender", "MMSE"]
    records: list[dict[str, Any]] = []
    for keys, frame in channel_table.groupby([*metadata, "metric"], dropna=False, sort=False):
        base = dict(zip([*metadata, "metric"], keys, strict=True))
        values = dict(zip(frame["channel"], frame["value"], strict=True))
        summary = regional_summary(values, channel_groups["rostral"], channel_groups["caudal"])
        records.append({**base, **summary})
    return pd.DataFrame(records)


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled = np.sqrt(((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2))
    d = (a.mean() - b.mean()) / pooled
    correction = 1 - 3 / (4 * (n_a + n_b) - 9)
    return float(correction * d)


def _summary(regional: pd.DataFrame) -> dict[str, Any]:
    results: dict[str, Any] = {"groups": {}, "ad_vs_ftd": {}}
    for metric in sorted(regional["metric"].unique()):
        subset = regional[regional["metric"] == metric]
        results["groups"][metric] = {}
        for group, frame in subset.groupby("diagnosis"):
            values = frame["rostrocaudal"].to_numpy(dtype=float)
            results["groups"][metric][group] = {
                "n": int(values.size),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "median": float(np.median(values)),
            }
        ad = subset.loc[subset["diagnosis"] == "AD", "rostrocaudal"].to_numpy(dtype=float)
        ftd = subset.loc[subset["diagnosis"] == "FTD", "rostrocaudal"].to_numpy(dtype=float)
        test = stats.ttest_ind(ad, ftd, equal_var=False)
        results["ad_vs_ftd"][metric] = {
            "mean_difference_ad_minus_ftd": float(ad.mean() - ftd.mean()),
            "welch_t": float(test.statistic),
            "welch_p_unadjusted": float(test.pvalue),
            "hedges_g": _hedges_g(ad, ftd),
        }
    return results


def run_reproduction(
    *,
    project_root: Path,
    config_path: Path,
    config: dict[str, Any],
    n_jobs: int = 1,
) -> tuple[Path, Path, Path]:
    dataset_id = "ds004504"
    dataset_root = project_root / "data" / dataset_id
    participants = load_participants(dataset_root, config["study"]["labels"])
    task = config["datasets"][dataset_id]["task"]
    derivative_pipeline = config["datasets"][dataset_id].get("derivative_pipeline")
    nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_participant_features)(
            row,
            dataset_root=dataset_root,
            task=task,
            derivative_pipeline=derivative_pipeline,
            config=config,
        )
        for _, row in participants.iterrows()
    )
    channel_table = pd.DataFrame([record for participant in nested for record in participant])
    regional_table = _regional_table(channel_table, config)
    summary = _summary(regional_table)

    output_dir = project_root / "outputs" / "reproduction"
    output_dir.mkdir(parents=True, exist_ok=True)
    channel_path = output_dir / "channel_complexity.csv"
    regional_path = output_dir / "regional_complexity.csv"
    summary_path = output_dir / "summary.json"
    channel_table.to_csv(channel_path, index=False)
    regional_table.to_csv(regional_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        project_root / "outputs" / "manifests" / "reproduction.json",
        stage="reproduction",
        project_root=project_root,
        config_path=config_path,
        inputs=[dataset_root / "participants.tsv"],
        outputs=[channel_path, regional_path, summary_path],
        extra={"n_participants": int(participants.shape[0]), "n_jobs": n_jobs},
    )
    return channel_path, regional_path, summary_path
