from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Participant:
    participant_id: str
    group: str
    age: float
    gender: str
    mmse: float


def load_participants(dataset_root: Path, label_map: dict[str, str]) -> pd.DataFrame:
    table = pd.read_csv(dataset_root / "participants.tsv", sep="\t", skipinitialspace=True)
    table.columns = table.columns.str.strip()
    for column in ("participant_id", "Gender", "Group"):
        table[column] = table[column].astype(str).str.strip()
    table["diagnosis"] = table["Group"].map(label_map)
    if table["diagnosis"].isna().any():
        unknown = sorted(table.loc[table["diagnosis"].isna(), "Group"].unique())
        raise ValueError(f"Unknown diagnostic labels: {unknown}")
    table["Age"] = pd.to_numeric(table["Age"], errors="raise")
    table["MMSE"] = pd.to_numeric(table["MMSE"], errors="coerce")
    return table


def set_path(
    dataset_root: Path,
    participant_id: str,
    task: str,
    derivative_pipeline: str | None = None,
) -> Path:
    derivatives = dataset_root / "derivatives"
    if derivative_pipeline:
        derivatives = derivatives / derivative_pipeline
    return derivatives / participant_id / "eeg" / f"{participant_id}_task-{task}_eeg.set"


def events_path(dataset_root: Path, participant_id: str, task: str) -> Path:
    return (
        dataset_root
        / participant_id
        / "eeg"
        / f"{participant_id}_task-{task}_events.tsv"
    )


def load_preprocessed_raw(
    path: Path,
    canonical_channels: list[str],
    *,
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    resample_hz: float = 250.0,
    rereference: str = "average",
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
    rename = {name: name.strip().replace("EEG ", "") for name in raw.ch_names}
    raw.rename_channels(rename)
    missing = [channel for channel in canonical_channels if channel not in raw.ch_names]
    if missing:
        raise ValueError(f"{path} is missing canonical channels: {missing}")
    raw.pick(canonical_channels)
    raw.set_channel_types({channel: "eeg" for channel in raw.ch_names})
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", phase="zero", verbose="ERROR")
    if rereference == "average":
        raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    elif rereference not in {"released", "mastoid"}:
        raise ValueError(f"Unsupported rereference setting: {rereference}")
    if not np.isclose(raw.info["sfreq"], resample_hz):
        raw.resample(resample_hz, npad="auto", verbose="ERROR")
    return raw


def fixed_segment(raw: mne.io.BaseRaw, seconds: float, start_seconds: float = 0.0) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    start = int(round(start_seconds * sfreq))
    stop = start + int(round(seconds * sfreq))
    if stop > raw.n_times:
        raise ValueError(
            f"Recording has {raw.n_times / sfreq:.2f}s, cannot extract "
            f"{seconds:.2f}s from {start_seconds:.2f}s"
        )
    return raw.get_data(start=start, stop=stop, units="uV")


def photic_open_intervals(events_path: Path) -> pd.DataFrame:
    events = pd.read_csv(events_path, sep="\t")
    events["value"] = events["value"].astype(str).str.strip()
    rows: list[dict[str, float]] = []
    current_frequency: float | None = None
    open_onset: float | None = None
    for event in events.itertuples(index=False):
        value = event.value.lower()
        if value.startswith("photo ") and "hz" in value:
            token = value.replace("photo", "").replace("hz", "").strip()
            try:
                current_frequency = float(token)
            except ValueError:
                current_frequency = None
        elif value == "open eyes":
            open_onset = float(event.onset)
        elif value == "closed eyes" and open_onset is not None:
            rows.append(
                {
                    "frequency_hz": (
                        float("nan") if current_frequency is None else current_frequency
                    ),
                    "onset": open_onset,
                    "duration": float(event.onset) - open_onset,
                }
            )
            open_onset = None
    return pd.DataFrame(rows, columns=["frequency_hz", "onset", "duration"])
