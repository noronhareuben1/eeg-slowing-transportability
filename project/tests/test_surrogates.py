import numpy as np

from rcd.surrogates import iaaft


def test_iaaft_preserves_distribution_and_approximately_preserves_spectrum() -> None:
    rng = np.random.default_rng(42)
    time = np.arange(2048) / 250
    signal = np.sin(2 * np.pi * 10 * time) + 0.4 * rng.normal(size=time.size)
    surrogate = iaaft(signal, rng=np.random.default_rng(7), max_iterations=500)
    np.testing.assert_allclose(np.sort(surrogate), np.sort(signal), rtol=0, atol=0)
    target = np.abs(np.fft.rfft(signal))
    actual = np.abs(np.fft.rfft(surrogate))
    relative_error = np.linalg.norm(actual - target) / np.linalg.norm(target)
    assert relative_error < 0.08
