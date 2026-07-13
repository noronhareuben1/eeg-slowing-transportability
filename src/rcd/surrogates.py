from __future__ import annotations

import numpy as np


def iaaft(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier-transform surrogate."""

    original = np.asarray(values, dtype=np.float64).squeeze()
    if original.ndim != 1 or original.size < 8:
        raise ValueError("IAAFT expects a one-dimensional series with at least 8 samples")
    sorted_original = np.sort(original)
    target_magnitude = np.abs(np.fft.rfft(original))
    surrogate = rng.permutation(original)
    previous_error = np.inf
    denominator = np.linalg.norm(target_magnitude) + np.finfo(float).eps

    for _ in range(max_iterations):
        spectrum = np.fft.rfft(surrogate)
        phase = np.divide(
            spectrum,
            np.abs(spectrum),
            out=np.ones_like(spectrum),
            where=np.abs(spectrum) > 0,
        )
        spectral_match = np.fft.irfft(target_magnitude * phase, n=original.size)
        ranks = np.argsort(np.argsort(spectral_match, kind="mergesort"), kind="mergesort")
        surrogate = sorted_original[ranks]
        error = np.linalg.norm(np.abs(np.fft.rfft(surrogate)) - target_magnitude) / denominator
        if abs(previous_error - error) < tolerance:
            break
        previous_error = error
    return surrogate


def surrogate_zscore(observed: float, surrogates: np.ndarray) -> float:
    surrogate_values = np.asarray(surrogates, dtype=np.float64)
    standard_deviation = surrogate_values.std(ddof=1)
    if standard_deviation == 0 or not np.isfinite(standard_deviation):
        return np.nan
    return float((observed - surrogate_values.mean()) / standard_deviation)
