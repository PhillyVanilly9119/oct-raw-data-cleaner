"""Core I/O — memory-mapped binary probing and entropy-based dtype detection."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np  # noqa: I001

# ── Supported candidate dtypes ──────────────────────────────────────────────

CANDIDATE_DTYPES: list[np.dtype] = [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.complex64),
    np.dtype(np.complex128),
]


@dataclass
class ProbeResult:
    """Container returned by :meth:`BinaryProber.probe`."""

    path: Path
    file_bytes: int
    best_dtype: np.dtype
    dtype_scores: dict[str, float] = field(default_factory=dict)
    a_scan_length: int | None = None
    b_scan_repeats: int | None = None


# ── Entropy helpers ─────────────────────────────────────────────────────────


def _byte_entropy(buf: np.ndarray) -> float:
    """Shannon entropy (bits) of a raw byte buffer."""
    counts = Counter(buf.tobytes())
    total = len(buf.tobytes())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _detect_element_width(mmap: np.memmap) -> int:
    """Infer the most-likely element width (bytes) from byte-level ACF.

    Computes the autocorrelation of the raw byte stream at candidate lags
    (1, 2, 4, 8, 16).  The lag at which the ACF first jumps significantly
    reveals the element width of the stored data.
    """
    raw = np.ndarray(min(len(mmap), 2**18), dtype=np.uint8, buffer=mmap)
    x = raw.astype(np.float64)
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy == 0:
        return 1

    candidate_lags = [1, 2, 4, 8, 16]
    acf_vals = {}
    for lag in candidate_lags:
        acf_vals[lag] = float(np.dot(x[lag:], x[:-lag])) / energy

    # Find the first lag where ACF jumps above a threshold, indicating
    # the true element boundary.
    threshold = 0.10
    for lag in candidate_lags:
        if acf_vals[lag] >= threshold:
            return lag

    return 1  # fall back to byte-level


def _dtype_entropy_score(
    mmap: np.memmap,
    dtype: np.dtype,
    inferred_width: int,
) -> float:
    """Score a candidate dtype using kurtosis and element-width agreement.

    Higher score → more plausible dtype.
    """
    n_elements = mmap.size // dtype.itemsize
    if n_elements < 64:
        return -math.inf

    view = np.ndarray(n_elements, dtype=dtype, buffer=mmap)
    sample = view[: min(n_elements, 2**18)].copy()

    if np.issubdtype(dtype, np.complexfloating):
        sample = np.abs(sample)

    with np.errstate(invalid="ignore"):
        sample = np.real(sample).astype(np.float64)

    if not np.all(np.isfinite(sample)):
        return -math.inf

    diff = np.diff(sample)
    std = diff.std()
    if std == 0:
        return -math.inf

    # Excess kurtosis of normalised first-difference.
    diff_norm = (diff - diff.mean()) / std
    kurt = float(np.mean(diff_norm**4)) - 3.0

    # Byte-width agreement bonus: dtypes whose itemsize matches the
    # inferred element width get a large boost.
    if dtype.itemsize == inferred_width:
        width_bonus = 10.0
    elif dtype.itemsize % inferred_width == 0 or inferred_width % dtype.itemsize == 0:
        width_bonus = 0.0
    else:
        width_bonus = -10.0

    # Light parsimony tiebreaker among same-width dtypes
    width_penalty = math.log2(max(dtype.itemsize, 1)) * 0.1
    return kurt + width_bonus - width_penalty

# ── BinaryProber ────────────────────────────────────────────────────────────


class BinaryProber:
    """Memory-mapped binary file prober with context-manager support.

    Usage::

        with BinaryProber("data.bin") as prober:
            result = prober.probe()
    """

    def __init__(
        self,
        path: str | Path,
        candidate_dtypes: Sequence[np.dtype] | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Binary file not found: {self.path}")
        self._candidates = list(candidate_dtypes or CANDIDATE_DTYPES)
        self._mmap: np.memmap | None = None

    # -- context-manager protocol -----------------------------------------

    def __enter__(self) -> "BinaryProber":
        self._mmap = np.memmap(self.path, dtype=np.uint8, mode="r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        if self._mmap is not None:
            del self._mmap
            self._mmap = None

    # -- public API -------------------------------------------------------

    @property
    def mmap(self) -> np.memmap:
        if self._mmap is None:
            raise RuntimeError(
                "BinaryProber must be used as a context manager "
                "(use `with BinaryProber(path) as p: …`)"
            )
        return self._mmap

    def detect_dtype(self) -> tuple[np.dtype, dict[str, float]]:
        """Return the most-likely dtype and a dict of per-dtype scores."""
        inferred_width = _detect_element_width(self.mmap)
        scores: dict[str, float] = {}
        for dt in self._candidates:
            scores[str(dt)] = _dtype_entropy_score(self.mmap, dt, inferred_width)
        best = max(scores, key=lambda k: scores[k])
        return np.dtype(best), scores

    def probe(self) -> ProbeResult:
        """Run full inference: dtype detection + periodicity analysis."""
        from oct_unscrambler.signal import detect_periodicity

        best_dtype, scores = self.detect_dtype()

        n_elements = self.mmap.size // best_dtype.itemsize
        view = np.ndarray(n_elements, dtype=best_dtype, buffer=self.mmap)
        if np.issubdtype(best_dtype, np.complexfloating):
            envelope = np.abs(view).astype(np.float64)
        else:
            envelope = np.real(view).astype(np.float64)

        a_scan_length = detect_periodicity(envelope)

        b_scan_repeats: int | None = None
        if a_scan_length and a_scan_length > 0:
            b_scan_repeats = n_elements // a_scan_length

        return ProbeResult(
            path=self.path,
            file_bytes=self.mmap.size,
            best_dtype=best_dtype,
            dtype_scores=scores,
            a_scan_length=a_scan_length,
            b_scan_repeats=b_scan_repeats,
        )
