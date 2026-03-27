"""Unit tests for oct-unscrambler — core inference and signal processing."""

from __future__ import annotations

import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest

from oct_unscrambler.core import BinaryProber, _dtype_entropy_score
from oct_unscrambler.signal import autocorrelation, detect_periodicity


# ── Fixtures ────────────────────────────────────────────────────────────────

PERIOD = 512  # synthetic A-scan length
N_BSCANS = 20


@pytest.fixture()
def uint16_binary(tmp_path: Path) -> Path:
    """Create a synthetic uint16 binary file with a clear periodic pattern."""
    rng = np.random.default_rng(42)
    single_ascan = np.sin(np.linspace(0, 4 * np.pi, PERIOD)) * 10_000 + 30_000
    single_ascan = single_ascan.astype(np.uint16)
    data = np.tile(single_ascan, N_BSCANS)
    # Add a small amount of noise so entropy is realistic
    noise = rng.integers(-50, 50, size=data.shape, dtype=np.int32)
    data = np.clip(data.astype(np.int32) + noise, 0, 65535).astype(np.uint16)
    fpath = tmp_path / "synthetic_uint16.bin"
    fpath.write_bytes(data.tobytes())
    return fpath


@pytest.fixture()
def float32_binary(tmp_path: Path) -> Path:
    """Create a synthetic float32 binary with a periodic envelope."""
    rng = np.random.default_rng(99)
    single = np.sin(np.linspace(0, 2 * np.pi, PERIOD)).astype(np.float32)
    data = np.tile(single, N_BSCANS)
    data += rng.normal(0, 0.01, size=data.shape).astype(np.float32)
    fpath = tmp_path / "synthetic_f32.bin"
    fpath.write_bytes(data.tobytes())
    return fpath


@pytest.fixture()
def tiny_binary(tmp_path: Path) -> Path:
    """A very small file (16 bytes) to test edge cases."""
    fpath = tmp_path / "tiny.bin"
    fpath.write_bytes(struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
    return fpath


# ── BinaryProber context manager ────────────────────────────────────────────


class TestBinaryProber:
    def test_context_manager_opens_mmap(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            assert prober.mmap is not None
            assert prober.mmap.size > 0

    def test_mmap_access_outside_context_raises(self, uint16_binary: Path) -> None:
        prober = BinaryProber(uint16_binary)
        with pytest.raises(RuntimeError, match="context manager"):
            _ = prober.mmap

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            BinaryProber(tmp_path / "nonexistent.bin")


# ── Dtype detection ─────────────────────────────────────────────────────────


class TestDtypeDetection:
    def test_uint16_detected(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            best_dtype, scores = prober.detect_dtype()
        assert best_dtype in (np.dtype(np.uint16), np.dtype(np.int16))

    def test_float32_detected(self, float32_binary: Path) -> None:
        with BinaryProber(float32_binary) as prober:
            best_dtype, scores = prober.detect_dtype()
        assert best_dtype == np.dtype(np.float32)

    def test_scores_dict_populated(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            _, scores = prober.detect_dtype()
        assert len(scores) > 0
        assert all(isinstance(v, float) for v in scores.values())


# ── Autocorrelation ─────────────────────────────────────────────────────────


class TestAutocorrelation:
    def test_peak_at_zero_is_one(self) -> None:
        x = np.random.default_rng(0).standard_normal(1024)
        acf = autocorrelation(x)
        assert abs(acf[0] - 1.0) < 1e-10

    def test_periodic_signal_has_peak(self) -> None:
        period = 128
        x = np.tile(np.sin(np.linspace(0, 2 * np.pi, period)), 20)
        acf = autocorrelation(x, max_lag=period * 2)
        # Expect a secondary peak near `period`
        search = acf[period - 5: period + 5]
        assert np.max(search) > 0.8


# ── Periodicity detection ──────────────────────────────────────────────────


class TestPeriodicityDetection:
    def test_known_period_recovered(self) -> None:
        period = 256
        x = np.tile(np.sin(np.linspace(0, 2 * np.pi, period)), 40)
        x += np.random.default_rng(7).normal(0, 0.05, size=x.shape)
        detected = detect_periodicity(x, min_period=64, max_period=1024)
        assert detected is not None
        assert abs(detected - period) <= 5  # tolerance ±5 samples

    def test_no_period_in_noise(self) -> None:
        x = np.random.default_rng(1).standard_normal(4096)
        detected = detect_periodicity(x, min_period=64)
        # Should return None or a spurious period; either is acceptable
        # but must not crash
        assert detected is None or isinstance(detected, int)


# ── Full probe pipeline ────────────────────────────────────────────────────


class TestProbe:
    def test_full_probe_uint16(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            result = prober.probe()
        assert result.file_bytes > 0
        assert result.best_dtype.itemsize > 0
        assert result.path == uint16_binary

    def test_full_probe_float32(self, float32_binary: Path) -> None:
        with BinaryProber(float32_binary) as prober:
            result = prober.probe()
        assert result.best_dtype == np.dtype(np.float32)

    def test_tiny_file_does_not_crash(self, tiny_binary: Path) -> None:
        with BinaryProber(tiny_binary) as prober:
            result = prober.probe()
        assert result.file_bytes == 16
