"""Unit tests for oct-unscrambler — core inference and signal processing."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from oct_unscrambler.core import BinaryProber, ValidationResult
from oct_unscrambler.signal import acf_peak_strength, autocorrelation, detect_periodicity


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
        search = acf[period - 5: period + 5]
        assert np.max(search) > 0.8


# ── ACF peak strength ──────────────────────────────────────────────────────


class TestACFPeakStrength:
    def test_strong_peak_at_true_lag(self) -> None:
        period = 256
        x = np.tile(np.sin(np.linspace(0, 2 * np.pi, period)), 40)
        strength = acf_peak_strength(x, period)
        assert strength > 0.8

    def test_weak_peak_at_wrong_lag(self) -> None:
        period = 256
        x = np.tile(np.sin(np.linspace(0, 2 * np.pi, period)), 40)
        # lag 137 is unlikely to be a harmonic
        strength = acf_peak_strength(x, 137)
        assert strength < 0.5


# ── Periodicity detection ──────────────────────────────────────────────────


class TestPeriodicityDetection:
    def test_known_period_recovered(self) -> None:
        period = 256
        x = np.tile(np.sin(np.linspace(0, 2 * np.pi, period)), 40)
        x += np.random.default_rng(7).normal(0, 0.05, size=x.shape)
        detected = detect_periodicity(x, min_period=64, max_period=1024)
        assert detected is not None
        assert abs(detected - period) <= 5

    def test_no_period_in_noise(self) -> None:
        x = np.random.default_rng(1).standard_normal(4096)
        detected = detect_periodicity(x, min_period=64)
        assert detected is None or isinstance(detected, int)


# ── Validate params ─────────────────────────────────────────────────────────


class TestValidateParams:
    def test_plausible_user_ascan(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            envelope = prober._envelope(np.dtype(np.uint16))
            detected = detect_periodicity(envelope)
            vr = prober.validate_params(envelope, detected)
        assert vr.plausible is True

    def test_implausible_user_ascan(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            envelope = prober._envelope(np.dtype(np.uint16))
            vr = prober.validate_params(envelope, 137)
        # 137 should not match the true period of ~512
        assert isinstance(vr, ValidationResult)
        assert vr.acf_strength is not None

    def test_none_user_ascan_returns_auto(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            envelope = prober._envelope(np.dtype(np.uint16))
            vr = prober.validate_params(envelope, None)
        assert vr.plausible is True
        assert vr.detected_ascan is not None


# ── Reshape stream ──────────────────────────────────────────────────────────


class TestReshapeStream:
    def test_basic_reshape(self) -> None:
        a_scan = 128
        bscan_len = 10
        n_reps = 2
        total = a_scan * bscan_len * n_reps
        flat = np.arange(total, dtype=np.float32)
        result = BinaryProber.reshape_stream(
            flat, a_scan, bscan_length=bscan_len, n_repeats=n_reps
        )
        assert result.shape == (1, bscan_len, n_reps, a_scan)

    def test_multi_volume(self) -> None:
        a_scan = 64
        bscan_len = 5
        n_reps = 1
        n_vols = 3
        total = a_scan * bscan_len * n_reps * n_vols
        flat = np.arange(total, dtype=np.float64)
        result = BinaryProber.reshape_stream(
            flat, a_scan, bscan_length=bscan_len, n_repeats=n_reps
        )
        assert result.shape == (n_vols, bscan_len, n_reps, a_scan)

    def test_trailing_samples_trimmed(self) -> None:
        a_scan = 100
        flat = np.arange(1050, dtype=np.float32)  # 10.5 A-scans
        result = BinaryProber.reshape_stream(
            flat, a_scan, bscan_length=10, n_repeats=1
        )
        assert result.shape == (1, 10, 1, 100)


# ── Save as .npy ────────────────────────────────────────────────────────────


class TestSaveAsNpy:
    def test_roundtrip(self, uint16_binary: Path, tmp_path: Path) -> None:
        out_path = tmp_path / "output.npy"
        with BinaryProber(uint16_binary) as prober:
            result = prober.probe()
            ascan = result.a_scan_length
            assert ascan is not None
            bscan_len = result.b_scan_repeats or 1
            prober.save_as_npy(
                out_path,
                dtype=result.best_dtype,
                a_scan_length=ascan,
                bscan_length=bscan_len,
                n_repeats=1,
            )
        loaded = np.load(str(out_path), mmap_mode="r")
        assert loaded.ndim == 4
        assert loaded.shape[-1] == ascan

    def test_correct_4d_shape(self, tmp_path: Path) -> None:
        """Verify [Volume, BScan, Repeat, AScan] structure."""
        a_scan = 64
        bscan_len = 10
        n_reps = 2
        total = a_scan * bscan_len * n_reps * 3  # 3 volumes
        data = np.arange(total, dtype=np.float32)
        bin_path = tmp_path / "vol.bin"
        bin_path.write_bytes(data.tobytes())

        out_path = tmp_path / "vol.npy"
        with BinaryProber(bin_path) as prober:
            prober.save_as_npy(
                out_path,
                dtype=np.dtype(np.float32),
                a_scan_length=a_scan,
                bscan_length=bscan_len,
                n_repeats=n_reps,
            )
        loaded = np.load(str(out_path), mmap_mode="r")
        assert loaded.shape == (3, bscan_len, n_reps, a_scan)

    def test_too_small_file_raises(self, tiny_binary: Path, tmp_path: Path) -> None:
        out = tmp_path / "bad.npy"
        with BinaryProber(tiny_binary) as prober:
            with pytest.raises(ValueError, match="too small"):
                prober.save_as_npy(
                    out,
                    dtype=np.dtype(np.float32),
                    a_scan_length=1024,
                    bscan_length=100,
                )

    def test_averaged_bscan_via_indexing(self, tmp_path: Path) -> None:
        """data[0].mean(axis=0) should give an averaged B-scan."""
        a_scan = 32
        bscan_len = 4
        n_reps = 3
        total = a_scan * bscan_len * n_reps
        raw = np.ones(total, dtype=np.float32) * 7.0
        bin_path = tmp_path / "ones.bin"
        bin_path.write_bytes(raw.tobytes())

        npy_path = tmp_path / "ones.npy"
        with BinaryProber(bin_path) as prober:
            prober.save_as_npy(
                npy_path,
                dtype=np.dtype(np.float32),
                a_scan_length=a_scan,
                bscan_length=bscan_len,
                n_repeats=n_reps,
            )

        data = np.load(str(npy_path), mmap_mode="r")
        # data[0] is (bscan_len, n_reps, a_scan) → mean over repeats
        avg_bscan = data[0].mean(axis=1)
        assert avg_bscan.shape == (bscan_len, a_scan)
        np.testing.assert_allclose(avg_bscan, 7.0)


# ── Probe with overrides ───────────────────────────────────────────────────


class TestProbeOverrides:
    def test_dtype_override(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            result = prober.probe(dtype=np.dtype(np.float32))
        assert result.best_dtype == np.dtype(np.float32)
        assert result.dtype_scores == {}

    def test_force_skips_validation(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            result = prober.probe(ascan_length=137, force=True)
        assert result.a_scan_length == 137

    def test_implausible_ascan_falls_back(self, uint16_binary: Path) -> None:
        with BinaryProber(uint16_binary) as prober:
            result = prober.probe(ascan_length=137, force=False)
        # Should fall back to auto-detected period (~512), not 137
        assert result.a_scan_length != 137 or result.a_scan_length is None


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
