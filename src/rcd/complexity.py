from __future__ import annotations

from dataclasses import dataclass

import antropy as ant
import numpy as np
from scipy.stats import iqr


@dataclass(frozen=True)
class ComplexityValues:
    box_count_fd: float
    higuchi_fd: float
    permutation_entropy: float
    sample_entropy: float
    lempel_ziv: float


def fractal_volatility(data: np.ndarray) -> tuple[float, float]:
    """Faithful Python translation of myFractal's ``fractalvol.m``.

    The original implementation embeds a one-dimensional series in the unit
    square, counts dyadic boxes, removes local-slope outliers, and fits an OLS
    line in log-log space. This function is retained specifically for direct
    reproduction; HFD and surrogate-normalized metrics are preferred for new
    inference.
    """

    values = np.asarray(data, dtype=np.float64).squeeze()
    if values.ndim != 1 or values.size < 16:
        raise ValueError("fractal_volatility expects a one-dimensional series with >=16 samples")
    if not np.isfinite(values).all() or np.ptp(values) == 0:
        return 0.5, np.nan

    x = np.arange(values.size, dtype=np.float64)
    x = (x - x.min()) / (x.max() - x.min())
    y = (values - values.min()) / (values.max() - values.min())

    min_width = int(abs(np.ceil(np.log2(np.min(np.diff(x))))) - 1)
    if min_width < 3:
        raise ValueError("Time series is too short for stable dyadic box counting")

    counts = np.zeros(min_width, dtype=np.float64)
    for j in range(1, min_width + 1):
        width = 2.0 ** (-j)
        bin_index = np.minimum(np.floor(x / width).astype(int), 2**j - 1)
        starts = np.flatnonzero(np.r_[True, np.diff(bin_index) != 0])
        sizes = np.diff(np.r_[starts, values.size])
        minima = np.minimum.reduceat(y, starts)
        maxima = np.maximum.reduceat(y, starts)
        column_counts = np.ones_like(minima)
        multiple = sizes > 1
        raw_counts = (maxima[multiple] - minima[multiple]) / width
        raw_counts += np.remainder(minima[multiple], width)
        column_counts[multiple] = np.ceil(raw_counts)
        counts[j - 1] = column_counts.sum()

    radii = 2.0 ** (-np.arange(1, min_width + 1, dtype=np.float64))
    local_slopes = -np.gradient(np.log(counts)) / np.gradient(np.log(radii))
    spread = iqr(local_slopes)
    keep = np.abs(local_slopes - np.median(local_slopes)) <= spread / 2.0
    if keep.sum() < 2:
        keep = np.ones_like(keep, dtype=bool)

    design = np.column_stack([np.ones(keep.sum()), np.log(radii[keep])])
    outcome = np.log(counts[keep])
    beta = np.linalg.pinv(design) @ outcome
    residual = outcome - design @ beta
    covariance = np.linalg.pinv(design.T @ design)
    sigma = np.sqrt(np.maximum((residual @ residual) * covariance, 0.0))
    return float(-beta[1]), float(sigma[1, 1])


def compute_complexity(values: np.ndarray, hfd_kmax: int = 32) -> ComplexityValues:
    signal = np.asarray(values, dtype=np.float64).squeeze()
    box_count_fd, _ = fractal_volatility(signal)
    binary = signal > np.median(signal)
    return ComplexityValues(
        box_count_fd=box_count_fd,
        higuchi_fd=float(ant.higuchi_fd(signal, kmax=hfd_kmax)),
        permutation_entropy=float(ant.perm_entropy(signal, order=3, delay=1, normalize=True)),
        sample_entropy=float(ant.sample_entropy(signal, order=2)),
        lempel_ziv=float(ant.lziv_complexity(binary.astype(int), normalize=True)),
    )


def regional_summary(
    channel_values: dict[str, float], rostral: list[str], caudal: list[str]
) -> dict[str, float]:
    rostral_mean = float(np.mean([channel_values[channel] for channel in rostral]))
    caudal_mean = float(np.mean([channel_values[channel] for channel in caudal]))
    return {
        "global": float(np.mean(list(channel_values.values()))),
        "rostral": rostral_mean,
        "caudal": caudal_mean,
        "rostrocaudal": rostral_mean - caudal_mean,
    }
