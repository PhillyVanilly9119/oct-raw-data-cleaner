"""Signal processing — periodicity detection via autocorrelation & coherence."""

from __future__ import annotations

import numpy as np
from scipy.signal import coherence


def autocorrelation(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    r"""Compute the normalised autocorrelation of a real signal.

    .. math::
        R_{xx}(\tau) = \frac{\sum_{n} x[n]\, x^{*}[n - \tau]}
                            {\sum_{n} |x[n]|^2}

    Parameters
    ----------
    x : 1-D real array
    max_lag : optional cap on the number of lags to return

    Returns
    -------
    r_xx : 1-D array of length ``max_lag`` (or ``len(x)``)
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    if max_lag is None:
        max_lag = n

    # FFT-based autocorrelation (efficient for large n)
    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    xf = np.fft.rfft(x, n=fft_size)
    acf = np.fft.irfft(xf * np.conj(xf), n=fft_size)[:max_lag]
    if acf[0] != 0:
        acf /= acf[0]
    return acf


def acf_peak_strength(
    envelope: np.ndarray,
    candidate_lag: int,
    *,
    tolerance: int = 5,
    max_lag: int | None = None,
) -> float:
    """Return the ACF value near *candidate_lag* (±tolerance).

    Used to check whether a user-supplied A-scan length matches a real
    periodicity peak.  Returns the maximum ACF value in the window
    ``[candidate_lag - tolerance, candidate_lag + tolerance]``.
    """
    if max_lag is None:
        max_lag = min(len(envelope) // 2, candidate_lag * 3)
    acf = autocorrelation(envelope, max_lag=max_lag)
    lo = max(candidate_lag - tolerance, 0)
    hi = min(candidate_lag + tolerance + 1, len(acf))
    if lo >= hi:
        return 0.0
    return float(np.max(acf[lo:hi]))


def magnitude_squared_coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    nperseg: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Thin wrapper around :func:`scipy.signal.coherence`.

    Returns ``(freqs, Cxy)`` where ``Cxy`` is the magnitude-squared coherence.
    """
    freqs, cxy = coherence(x, y, fs=fs, nperseg=nperseg)
    return freqs, cxy


def detect_periodicity(
    envelope: np.ndarray,
    min_period: int = 64,
    max_period: int | None = None,
) -> int | None:
    """Estimate the dominant period (A-scan length) from the signal envelope.

    Strategy: find the first prominent peak in the autocorrelation function
    beyond ``min_period`` samples.

    Parameters
    ----------
    envelope : 1-D real array (e.g. ``|complex_signal|``)
    min_period : smallest plausible A-scan length
    max_period : largest plausible A-scan length (default: half the signal)

    Returns
    -------
    period : int or None if no clear periodicity is found
    """
    envelope = np.asarray(envelope, dtype=np.float64)
    n = len(envelope)
    if max_period is None:
        max_period = n // 2
    max_period = min(max_period, n // 2)

    if max_period <= min_period:
        return None

    acf = autocorrelation(envelope, max_lag=max_period)

    # Search for the first peak above a significance threshold
    search = acf[min_period:max_period]
    if len(search) == 0:
        return None

    threshold = 0.15
    peak_idx = None
    for i in range(1, len(search) - 1):
        if search[i] > search[i - 1] and search[i] > search[i + 1]:
            if search[i] >= threshold:
                peak_idx = i
                break

    if peak_idx is None:
        return None

    return int(min_period + peak_idx)
