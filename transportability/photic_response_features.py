"""Extract a participant-level spectral response fingerprint during photic stimulation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import antropy as ant
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import welch

from rcd.data import (
    events_path,
    fixed_segment,
    load_participants,
    load_preprocessed_raw,
    photic_open_intervals,
    set_path,
)
from rcd.spectra import compute_spectral_features

CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
]
COMMON_FREQUENCIES = (5.0, 10.0, 15.0, 20.0)
LABELS = {"A": "AD", "F": "FTD", "C": "CN"}


def _spectral_entropy(power: np.ndarray) -> float:
    probability = power / np.maximum(power.sum(), np.finfo(float).tiny)
    entropy = -np.sum(probability * np.log(probability + np.finfo(float).tiny))
    return float(entropy / np.log(len(probability)))


def _peak_snr_db(
    frequencies: np.ndarray,
    power: np.ndarray,
    target_hz: float,
) -> float:
    if target_hz < frequencies[0] or target_hz > frequencies[-1]:
        return float("nan")
    peak = np.abs(frequencies - target_hz) <= 0.40
    neighborhood = (
        (np.abs(frequencies - target_hz) >= 0.75)
        & (np.abs(frequencies - target_hz) <= 2.0)
    )
    if not peak.any() or neighborhood.sum() < 2:
        return float("nan")
    peak_power = float(np.mean(power[peak]))
    noise_power = float(np.median(power[neighborhood]))
    return float(10.0 * np.log10(peak_power / max(noise_power, np.finfo(float).tiny)))


def interval_features(
    values: np.ndarray,
    *,
    sfreq: float,
    stimulus_hz: float,
) -> dict[str, float]:
    spectral = asdict(compute_spectral_features(values, sfreq))
    nperseg = min(values.size, int(round(3.0 * sfreq)))
    frequencies, power = welch(
        values,
        fs=sfreq,
        nperseg=nperseg,
        detrend="constant",
        scaling="density",
    )
    mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    frequencies = frequencies[mask]
    power = np.maximum(power[mask], np.finfo(float).tiny)
    harmonic_hz = 2.0 * stimulus_hz
    return {
        **spectral,
        "spectral_entropy": _spectral_entropy(power),
        "stimulus_snr_db": _peak_snr_db(frequencies, power, stimulus_hz),
        "harmonic_snr_db": _peak_snr_db(frequencies, power, harmonic_hz),
        "higuchi_fd": float(ant.higuchi_fd(values, kmax=32)),
    }


def participant_features(
    row: pd.Series,
    *,
    dataset_root: Path,
    interval_seconds: float = 2.0,
    transient_seconds: float = 0.25,
) -> list[dict[str, object]]:
    raw = load_preprocessed_raw(
        set_path(dataset_root, row["participant_id"], "photomark", "eeglab"),
        CHANNELS,
        l_freq=1.0,
        h_freq=45.0,
        resample_hz=250.0,
        rereference="average",
    )
    intervals = photic_open_intervals(
        events_path(dataset_root, row["participant_id"], "photomark")
    )
    intervals = intervals.loc[
        intervals["frequency_hz"].isin(COMMON_FREQUENCIES)
        & (intervals["duration"] >= interval_seconds + transient_seconds)
    ]
    records: list[dict[str, object]] = []
    for interval in intervals.itertuples(index=False):
        start = float(interval.onset) + transient_seconds
        if start + interval_seconds > raw.times[-1]:
            continue
        segment = fixed_segment(raw, interval_seconds, start)
        for channel, values in zip(raw.ch_names, segment, strict=True):
            records.append(
                {
                    "participant_id": row["participant_id"],
                    "diagnosis": row["diagnosis"],
                    "frequency_hz": float(interval.frequency_hz),
                    "channel": channel,
                    **interval_features(
                        values,
                        sfreq=float(raw.info["sfreq"]),
                        stimulus_hz=float(interval.frequency_hz),
                    ),
                }
            )
    if not records:
        return []
    table = pd.DataFrame(records)
    value_columns = [
        column
        for column in table.columns
        if column not in {"participant_id", "diagnosis", "frequency_hz", "channel"}
    ]
    aggregate = (
        table.groupby(
            ["participant_id", "diagnosis", "frequency_hz", "channel"], as_index=False
        )[value_columns]
        .mean()
    )
    counts = table.groupby(["frequency_hz", "channel"]).size()
    aggregate["n_intervals"] = [
        int(counts.loc[(frequency, channel)])
        for frequency, channel in aggregate[["frequency_hz", "channel"]].itertuples(
            index=False, name=None
        )
    ]
    return aggregate.to_dict("records")


def extract_photic_response_features(
    project_root: Path,
    *,
    n_jobs: int = 1,
    interval_seconds: float = 2.0,
    transient_seconds: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dataset_root = project_root / "data" / "ds006036"
    participants = load_participants(dataset_root, LABELS)
    nested = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(participant_features)(
            row,
            dataset_root=dataset_root,
            interval_seconds=interval_seconds,
            transient_seconds=transient_seconds,
        )
        for _, row in participants.iterrows()
    )
    table = pd.DataFrame([record for participant in nested for record in participant])
    table = table.sort_values(["participant_id", "frequency_hz", "channel"]).reset_index(
        drop=True
    )
    observed = (
        table.groupby("participant_id")["frequency_hz"].agg(lambda x: sorted(set(x))).to_dict()
    )
    complete = [
        participant_id
        for participant_id, frequencies in observed.items()
        if frequencies == list(COMMON_FREQUENCIES)
    ]
    summary: dict[str, object] = {
        "dataset": "OpenNeuro ds006036 snapshot 1.0.6",
        "interval_seconds": interval_seconds,
        "transient_seconds": transient_seconds,
        "common_frequencies_hz": list(COMMON_FREQUENCIES),
        "n_participants_total": int(len(participants)),
        "n_participants_with_features": int(table["participant_id"].nunique()),
        "n_participants_complete_four_frequency": len(complete),
        "participants_without_complete_four_frequency": sorted(
            set(participants["participant_id"]) - set(complete)
        ),
        "n_feature_rows": int(len(table)),
    }
    return table, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    root = args.project_root.resolve()
    table, summary = extract_photic_response_features(root, n_jobs=args.n_jobs)
    output_dir = root / "outputs" / "amendment_v1_2"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "photic_response_features.csv", index=False)
    (output_dir / "photic_response_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
