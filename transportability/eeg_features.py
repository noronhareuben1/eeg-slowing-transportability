"""Small, transparent EEG feature extraction for routine MATLAB recordings."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.signal import welch

CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
)
POSTERIOR = np.array([12, 13, 14, 15, 16, 17, 18])


def _largest_numeric_variable(path: pathlib.Path) -> str:
    entries = whosmat(path)
    candidates = [(int(np.prod(shape)), name) for name, shape, kind in entries
                  if kind in {"double", "single", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"}]
    if not candidates:
        raise ValueError(f"no numeric MATLAB variable found in {path}")
    return max(candidates)[1]


def load_recordings(path: str | pathlib.Path) -> np.ndarray:
    """Return recordings as (subjects, channels, samples).

    The Dryad release describes each group as a three-dimensional MATLAB
    matrix. This loader does not assume which of the three axes is first; it
    identifies the 19-channel axis and the subject axis from the dimensions.
    """
    path = pathlib.Path(path)
    name = _largest_numeric_variable(path)
    raw: Any = loadmat(path, variable_names=[name], squeeze_me=False)[name]
    array = np.asarray(raw, dtype=np.float64)
    array = np.squeeze(array)
    if array.ndim == 1:
        raise ValueError(f"unexpected one-dimensional recording array: {array.shape}")
    if array.ndim == 2:
        if 19 not in array.shape:
            raise ValueError(f"cannot identify 19-channel axis: {array.shape}")
        channel_axis = int(np.flatnonzero(np.array(array.shape) == 19)[0])
        array = np.moveaxis(array, channel_axis, 0)[None, :, :]
        return array
    if array.ndim != 3:
        raise ValueError(f"expected a 2D or 3D recording matrix, got {array.shape}")
    channel_axes = np.flatnonzero(np.array(array.shape) == 19)
    if len(channel_axes) != 1:
        raise ValueError(f"ambiguous channel axis in {array.shape}")
    channel_axis = int(channel_axes[0])
    remaining = [axis for axis in range(3) if axis != channel_axis]
    subject_axis = min(remaining, key=lambda axis: array.shape[axis])
    time_axis = next(axis for axis in remaining if axis != subject_axis)
    array = np.moveaxis(array, (subject_axis, channel_axis, time_axis), (0, 1, 2))
    return array


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return float("nan")
    return float(np.trapz(power[mask], freqs[mask]))


def _subject_features(recording: np.ndarray, sfreq: float) -> dict[str, float]:
    recording = np.asarray(recording, dtype=float)
    recording = recording - np.nanmedian(recording, axis=1, keepdims=True)
    recording = np.nan_to_num(recording, nan=0.0, posinf=0.0, neginf=0.0)
    n_samples = min(recording.shape[-1], int(120 * sfreq))
    recording = recording[:, :n_samples]
    nperseg = min(recording.shape[-1], int(4 * sfreq))
    freqs, spectra = welch(recording, fs=sfreq, nperseg=nperseg, axis=-1)
    spectra = np.maximum(spectra, np.finfo(float).tiny)
    global_power = spectra.mean(axis=0)
    posterior_power = spectra[POSTERIOR].mean(axis=0)
    delta_global = _band_power(freqs, global_power, 1.0, 4.0)
    alpha_global = _band_power(freqs, global_power, 8.0, 13.0)
    delta_post = _band_power(freqs, posterior_power, 1.0, 4.0)
    alpha_post = _band_power(freqs, posterior_power, 8.0, 13.0)
    fit = (freqs >= 2.0) & (freqs <= 30.0)
    slope = float(np.polyfit(np.log10(freqs[fit]), np.log10(global_power[fit]), 1)[0])
    return {
        "posterior_delta_alpha_ratio": delta_post / max(alpha_post, np.finfo(float).tiny),
        "posterior_alpha_relative": alpha_post / max(_band_power(freqs, posterior_power, 1.0, 30.0), np.finfo(float).tiny),
        "global_aperiodic_exponent": -slope,
        "global_delta_relative": delta_global / max(_band_power(freqs, global_power, 1.0, 30.0), np.finfo(float).tiny),
        "global_alpha_relative": alpha_global / max(_band_power(freqs, global_power, 1.0, 30.0), np.finfo(float).tiny),
    }


def extract_group(path: str | pathlib.Path, diagnosis: str, sfreq: float = 500.0) -> pd.DataFrame:
    recordings = load_recordings(path)
    rows = []
    for index, recording in enumerate(recordings):
        row = {"recording_index": index, "diagnosis": diagnosis}
        row.update(_subject_features(recording, sfreq))
        rows.append(row)
    return pd.DataFrame(rows)
