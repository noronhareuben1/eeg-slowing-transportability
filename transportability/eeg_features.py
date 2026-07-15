"""Small, transparent EEG feature extraction for routine MATLAB recordings."""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat
from scipy.io.matlab import MatReadError
from scipy.signal import welch

CHANNELS = (
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
    "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2",
)
POSTERIOR = np.array([12, 13, 14, 15, 16, 17, 18])


def _largest_numeric_variable(path: pathlib.Path) -> str:
    entries = whosmat(path)
    numeric_kinds = {
        "double",
        "single",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
    }
    candidates = [
        (int(np.prod(shape)), name)
        for name, shape, kind in entries
        if kind in numeric_kinds
    ]
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


def _hdf5_group_name(path: pathlib.Path) -> str | None:
    stem = path.stem.lower()
    if stem.startswith("alz"):
        return "alz_r"
    if stem.startswith("controls"):
        return "controls_r"
    return None


def _resolve_hdf5_recording(handle: Any, ref: Any) -> Any | None:
    if not ref:
        return None
    obj = handle[ref]
    shape = getattr(obj, "shape", None)
    dtype = getattr(obj, "dtype", None)
    if shape is not None and dtype is not None and dtype.kind != "O":
        if len(shape) == 2 and 19 in shape and max(shape) > 1000:
            return obj
        return None
    array = np.asarray(obj)
    if array.dtype == object:
        for child in array.flat:
            dataset = _resolve_hdf5_recording(handle, child)
            if dataset is not None:
                return dataset
    return None


def _decode_hdf5_value(handle: Any, ref: Any) -> object:
    if not ref:
        return np.nan
    obj = handle[ref]
    array = np.asarray(obj)
    if array.dtype == object:
        for child in array.flat:
            value = _decode_hdf5_value(handle, child)
            if value != "" and not (isinstance(value, float) and np.isnan(value)):
                return value
        return np.nan
    if array.dtype.kind in "ui" and array.size > 1:
        return "".join(chr(int(value)) for value in array.flat if int(value) != 0).strip()
    if array.size == 1:
        value = array.item()
        if isinstance(value, np.integer):
            code = int(value)
            if 32 <= code <= 126:
                return chr(code)
            return code
        return float(value) if isinstance(value, np.floating) else value
    if array.dtype.kind in "fiu" and array.size:
        return float(array.flat[0])
    return np.nan


def _slice_hdf5_recording(dataset: Any, sfreq: float) -> np.ndarray:
    n_samples = min(max(dataset.shape), int(120 * sfreq))
    if dataset.shape[1] == 19:
        return np.asarray(dataset[:n_samples, :], dtype=np.float64).T
    return np.asarray(dataset[:, :n_samples], dtype=np.float64)


def extract_hdf5_group(path: str | pathlib.Path, diagnosis: str) -> pd.DataFrame:
    """Extract features from P-ADIC MATLAB v7.3/HDF5 structs."""
    import h5py

    path = pathlib.Path(path)
    group_name = _hdf5_group_name(path)
    if group_name is None:
        raise ValueError(f"cannot infer P-ADIC group name from {path.name}")
    rows = []
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        recording_refs = np.asarray(group["G"])
        for index in np.ndindex(recording_refs.shape):
            dataset = _resolve_hdf5_recording(handle, recording_refs[index])
            if dataset is None:
                continue
            sfreq = _decode_hdf5_value(handle, np.asarray(group["g"])[index])
            sfreq = float(sfreq) if np.isfinite(sfreq) else 500.0
            recording = _slice_hdf5_recording(dataset, sfreq=sfreq)
            row = {
                "recording_index": f"{index[0]}_{index[1]}",
                "diagnosis": diagnosis,
                "age": _decode_hdf5_value(handle, np.asarray(group["age"])[index]),
                "sex": _decode_hdf5_value(handle, np.asarray(group["sex"])[index]),
                "sfreq": sfreq,
            }
            row.update(_subject_features(recording, sfreq=sfreq))
            rows.append(row)
    return pd.DataFrame(rows)


def _band_power(freqs: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
        return float("nan")
    return float(np.trapezoid(power[mask], freqs[mask]))


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
        "posterior_alpha_relative": alpha_post
        / max(_band_power(freqs, posterior_power, 1.0, 30.0), np.finfo(float).tiny),
        "global_aperiodic_exponent": -slope,
        "global_delta_relative": delta_global
        / max(_band_power(freqs, global_power, 1.0, 30.0), np.finfo(float).tiny),
        "global_alpha_relative": alpha_global
        / max(_band_power(freqs, global_power, 1.0, 30.0), np.finfo(float).tiny),
    }


def extract_group(path: str | pathlib.Path, diagnosis: str, sfreq: float = 500.0) -> pd.DataFrame:
    path = pathlib.Path(path)
    try:
        return extract_hdf5_group(path, diagnosis)
    except (ImportError, OSError, MatReadError, NotImplementedError, ValueError):
        pass
    recordings = load_recordings(path)
    rows = []
    for index, recording in enumerate(recordings):
        row = {"recording_index": index, "diagnosis": diagnosis}
        row.update(_subject_features(recording, sfreq))
        rows.append(row)
    return pd.DataFrame(rows)
