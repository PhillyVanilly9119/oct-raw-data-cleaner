"""Core I/O — memory-mapped binary probing and entropy-based dtype detection."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import numpy.lib.format  # noqa: I001

# ── Supported candidate dtypes ──────────────────────────────────────────────

CANDIDATE_DTYPES: list[np.dtype] = [
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int16),
    np.dtype(np.float32),
    np.dtype(np.float64),
    np.dtype(np.complex64), # TODO: rethink if these types are realistic
    np.dtype(np.complex128),
]

# Minimum ACF strength to consider a user-supplied A-scan length plausible.
_ACF_PLAUSIBILITY_THRESHOLD = 0.10


@dataclass
class ProbeResult:
    """Container returned by :meth:`BinaryProber.probe`."""

    path: Path
    file_bytes: int
    best_dtype: np.dtype
    dtype_scores: dict[str, float] = field(default_factory=dict)
    a_scan_length: int | None = None
    b_scan_repeats: int | None = None
    bscan_length: int | None = None
    n_repeats: int | None = None

    @property
    def total_elements(self) -> int:
        return self.file_bytes // self.best_dtype.itemsize


@dataclass
class ValidationResult:
    """Outcome of :meth:`BinaryProber.validate_params`."""

    user_ascan: int | None
    detected_ascan: int | None
    acf_strength: float | None = None
    plausible: bool = True
    message: str = ""


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
    """Infer the most-likely element width (bytes) from byte-level ACF."""
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

    threshold = 0.10
    for lag in candidate_lags:
        if acf_vals[lag] >= threshold:
            return lag

    return 1


def _dtype_entropy_score(
    mmap: np.memmap,
    dtype: np.dtype,
    inferred_width: int,
) -> float:
    """Score a candidate dtype using kurtosis and element-width agreement."""
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

    diff_norm = (diff - diff.mean()) / std
    kurt = float(np.mean(diff_norm**4)) - 3.0

    if dtype.itemsize == inferred_width:
        width_bonus = 10.0
    elif (
        dtype.itemsize % inferred_width == 0
        or inferred_width % dtype.itemsize == 0
    ):
        width_bonus = 0.0
    else:
        width_bonus = -10.0

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
            scores[str(dt)] = _dtype_entropy_score(
                self.mmap, dt, inferred_width
            )
        best = max(scores, key=lambda k: scores[k])
        return np.dtype(best), scores

    def _envelope(self, dtype: np.dtype) -> np.ndarray:
        """Return the real-valued envelope for periodicity analysis."""
        n = self.mmap.size // dtype.itemsize
        view = np.ndarray(n, dtype=dtype, buffer=self.mmap)
        if np.issubdtype(dtype, np.complexfloating):
            return np.abs(view).astype(np.float64)
        return np.real(view).astype(np.float64)

    # ── 1. Metadata override & validation ───────────────────────────────

    def validate_params(
        self,
        envelope: np.ndarray,
        user_ascan: int | None = None,
    ) -> ValidationResult:
        """Compare a user-supplied A-scan length against the ACF peak.

        Returns a :class:`ValidationResult` with *plausible=False* when
        the ACF at the user's lag is below the threshold.
        """
        from oct_unscrambler.signal import (
            acf_peak_strength,
            detect_periodicity,
        )

        detected = detect_periodicity(envelope)

        if user_ascan is None:
            return ValidationResult(
                user_ascan=None,
                detected_ascan=detected,
                plausible=True,
                message="No user A-scan supplied; using auto-detection.",
            )

        strength = acf_peak_strength(envelope, user_ascan)
        if strength >= _ACF_PLAUSIBILITY_THRESHOLD:
            return ValidationResult(
                user_ascan=user_ascan,
                detected_ascan=detected,
                acf_strength=strength,
                plausible=True,
                message=(
                    f"User A-scan {user_ascan} is plausible "
                    f"(ACF = {strength:.3f})."
                ),
            )

        return ValidationResult(
            user_ascan=user_ascan,
            detected_ascan=detected,
            acf_strength=strength,
            plausible=False,
            message=(
                f"User A-scan {user_ascan} looks improbable "
                f"(ACF = {strength:.3f} < {_ACF_PLAUSIBILITY_THRESHOLD}). "
                f"Auto-detected period is {detected}."
            ),
        )

    # ── 2. Smart re-ordering ────────────────────────────────────────────

    @staticmethod
    def reshape_stream(
        flat: np.ndarray,
        a_scan_length: int,
        bscan_length: int | None = None,
        n_repeats: int | None = None,
    ) -> np.ndarray:
        """Reshape a flat element stream into ``[Volume, BScan, Repeat, AScan]``.

        Parameters
        ----------
        flat : 1-D typed array (or memmap view of the raw file).
        a_scan_length : number of samples per A-scan (depth).
        bscan_length : number of A-scans per B-scan.  If *None*,
            auto-detected from inter-B-scan correlation.
        n_repeats : number of repeated B-scans at each location.
            If *None*, detected by looking for a secondary
            periodicity in the B-scan–mean signal.

        Returns
        -------
        data : 4-D array ``[n_volumes, bscan_length, n_repeats, a_scan_length]``
        """
        total = len(flat)
        n_ascans_total = total // a_scan_length
        usable = n_ascans_total * a_scan_length
        flat = flat[:usable]

        if bscan_length is None:
            bscan_length = n_ascans_total  # single B-scan fallback

        if n_repeats is None:
            n_repeats = BinaryProber._detect_repeats(
                flat, a_scan_length, bscan_length
            )

        ascans_per_volume = bscan_length * n_repeats
        n_volumes = n_ascans_total // ascans_per_volume
        if n_volumes < 1:
            n_volumes = 1
            ascans_per_volume = n_ascans_total
            bscan_length = n_ascans_total // n_repeats

        usable = n_volumes * ascans_per_volume * a_scan_length
        return flat[:usable].reshape(
            n_volumes, bscan_length, n_repeats, a_scan_length
        )

    @staticmethod
    def _detect_repeats(
        flat: np.ndarray,
        a_scan_length: int,
        bscan_length: int,
    ) -> int:
        """Detect the number of repeated B-scans at a location.

        Compares consecutive B-scan means; a high correlation between
        adjacent B-scans indicates functional (repeated) scanning.
        """
        n_ascans = len(flat) // a_scan_length
        if n_ascans < 2 * bscan_length:
            return 1

        bscan_means: list[float] = []
        for i in range(0, n_ascans - bscan_length + 1, bscan_length):
            chunk = flat[
                i * a_scan_length : (i + bscan_length) * a_scan_length
            ]
            bscan_means.append(float(np.mean(np.abs(chunk))))

        if len(bscan_means) < 4:
            return 1

        means = np.array(bscan_means, dtype=np.float64)
        means = means - means.mean()
        if means.std() == 0:
            return 1

        from oct_unscrambler.signal import autocorrelation

        acf = autocorrelation(means, max_lag=len(means) // 2)
        # Find the first dip after lag-0 then the next peak as repeat count
        for i in range(2, len(acf) - 1):
            if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.3:
                return i
        return 1

    # ── 3. Memory-safe .npy export ──────────────────────────────────────

    def save_as_npy(
        self,
        output_path: str | Path,
        dtype: np.dtype,
        a_scan_length: int,
        bscan_length: int,
        n_repeats: int = 1,
    ) -> Path:
        """Write a structured 4-D ``.npy`` file using memory-mapped I/O.

        Shape: ``[n_volumes, bscan_length, n_repeats, a_scan_length]``.

        Only one B-scan chunk (~``a_scan_length * dtype.itemsize`` bytes)
        is resident in RAM at a time, so even 10 GB files stay lean.
        """
        output_path = Path(output_path)
        n_elements = self.mmap.size // dtype.itemsize
        ascans_total = n_elements // a_scan_length
        ascans_per_volume = bscan_length * n_repeats
        n_volumes = ascans_total // ascans_per_volume
        if n_volumes < 1:
            raise ValueError(
                f"File too small for the requested shape: "
                f"{ascans_total} A-scans available, "
                f"{ascans_per_volume} needed per volume."
            )

        shape = (n_volumes, bscan_length, n_repeats, a_scan_length)
        out = numpy.lib.format.open_memmap(
            str(output_path), mode="w+", dtype=dtype, shape=shape
        )

        src = np.ndarray(n_elements, dtype=dtype, buffer=self.mmap)
        chunk_size = n_repeats * a_scan_length  # one B-scan position

        for vol in range(n_volumes):
            for bscan in range(bscan_length):
                idx = (vol * ascans_per_volume + bscan * n_repeats)
                start = idx * a_scan_length
                end = start + chunk_size
                out[vol, bscan] = src[start:end].reshape(
                    n_repeats, a_scan_length
                )

        out.flush()
        del out
        return output_path

    # ── Full probe (updated) ────────────────────────────────────────────

    def probe(
        self,
        *,
        dtype: np.dtype | None = None,
        ascan_length: int | None = None,
        bscan_length: int | None = None,
        n_repeats: int | None = None,
        force: bool = False,
    ) -> ProbeResult:
        """Run full inference with optional user overrides.

        Parameters
        ----------
        dtype : override for data type (skip auto-detection).
        ascan_length : override for A-scan depth.
        bscan_length : number of A-scans per B-scan.
        n_repeats : repeated B-scans per spatial location.
        force : if *True*, skip validation of user-supplied parameters.
        """
        from oct_unscrambler.signal import detect_periodicity

        if dtype is not None:
            best_dtype = np.dtype(dtype)
            scores: dict[str, float] = {}
        else:
            best_dtype, scores = self.detect_dtype()

        envelope = self._envelope(best_dtype)
        n_elements = len(envelope)

        # A-scan length
        detected_ascan = detect_periodicity(envelope)
        if ascan_length is not None and not force:
            vr = self.validate_params(envelope, ascan_length)
            if not vr.plausible:
                warnings.warn(vr.message, stacklevel=2)
                ascan_length = detected_ascan
        effective_ascan = ascan_length or detected_ascan

        b_reps: int | None = None
        if effective_ascan and effective_ascan > 0:
            b_reps = n_elements // effective_ascan

        return ProbeResult(
            path=self.path,
            file_bytes=self.mmap.size,
            best_dtype=best_dtype,
            dtype_scores=scores,
            a_scan_length=effective_ascan,
            b_scan_repeats=b_reps,
            bscan_length=bscan_length,
            n_repeats=n_repeats,
        )
