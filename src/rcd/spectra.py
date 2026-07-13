from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from specparam import SpectralModel


@dataclass(frozen=True)
class SpectralValues:
    delta_relative: float
    theta_relative: float
    alpha_relative: float
    beta_relative: float
    gamma_relative: float
    spectral_edge_95_hz: float
    median_frequency_hz: float
    loglog_slope: float
    aperiodic_offset: float
    aperiodic_exponent: float


def _band_power(frequencies: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequencies >= low) & (frequencies < high)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(power[mask], frequencies[mask]))


def compute_spectral_features(
    values: np.ndarray,
    sfreq: float,
    *,
    peak_width_limits: tuple[float, float] = (1.0, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.1,
) -> SpectralValues:
    signal = np.asarray(values, dtype=np.float64).squeeze()
    nperseg = min(signal.size, int(round(4.0 * sfreq)))
    frequencies, power = welch(
        signal,
        fs=sfreq,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )
    mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    frequencies, power = frequencies[mask], power[mask]
    total = float(np.trapezoid(power, frequencies))
    bands = {
        "delta": _band_power(frequencies, power, 1.0, 4.0) / total,
        "theta": _band_power(frequencies, power, 4.0, 8.0) / total,
        "alpha": _band_power(frequencies, power, 8.0, 13.0) / total,
        "beta": _band_power(frequencies, power, 13.0, 30.0) / total,
        "gamma": _band_power(frequencies, power, 30.0, 45.01) / total,
    }
    cumulative = np.cumsum(power)
    cumulative /= cumulative[-1]
    edge = float(frequencies[np.searchsorted(cumulative, 0.95)])
    median = float(frequencies[np.searchsorted(cumulative, 0.50)])
    slope_mask = (frequencies >= 2.0) & (frequencies <= 40.0) & (power > 0)
    slope = float(np.polyfit(np.log10(frequencies[slope_mask]), np.log10(power[slope_mask]), 1)[0])
    model = SpectralModel(
        aperiodic_mode="fixed",
        peak_width_limits=list(peak_width_limits),
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        verbose=False,
    )
    model.fit(frequencies, power, [1.0, 45.0])
    aperiodic = np.asarray(model.get_params("aperiodic"), dtype=float)
    if aperiodic.size != 2:
        raise ValueError(f"Expected fixed aperiodic parameters, received {aperiodic}")
    return SpectralValues(
        delta_relative=bands["delta"],
        theta_relative=bands["theta"],
        alpha_relative=bands["alpha"],
        beta_relative=bands["beta"],
        gamma_relative=bands["gamma"],
        spectral_edge_95_hz=edge,
        median_frequency_hz=median,
        loglog_slope=slope,
        aperiodic_offset=float(aperiodic[0]),
        aperiodic_exponent=float(aperiodic[1]),
    )
