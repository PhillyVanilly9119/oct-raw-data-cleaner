"""Tests for the view command and plot_projections utility."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from oct_unscrambler.cli import app
from oct_unscrambler.utils import plot_projections

runner = CliRunner()

# ── Fixtures ────────────────────────────────────────────────────────────────

SHAPE_3D = (10, 20, 30)
SHAPE_4D = (2, 10, 20, 30)


@pytest.fixture()
def volume_3d(tmp_path: Path) -> Path:
    """Create a 3-D .npy test volume."""
    rng = np.random.default_rng(42)
    vol = rng.integers(0, 255, size=SHAPE_3D, dtype=np.uint8)
    p = tmp_path / "vol3d.npy"
    np.save(str(p), vol)
    return p


@pytest.fixture()
def volume_4d(tmp_path: Path) -> Path:
    """Create a 4-D .npy test volume (multi-volume)."""
    rng = np.random.default_rng(99)
    vol = rng.standard_normal(SHAPE_4D).astype(np.float32)
    p = tmp_path / "vol4d.npy"
    np.save(str(p), vol)
    return p


@pytest.fixture()
def volume_complex(tmp_path: Path) -> Path:
    """Create a complex-valued 3-D .npy volume."""
    rng = np.random.default_rng(7)
    real = rng.standard_normal(SHAPE_3D).astype(np.float32)
    imag = rng.standard_normal(SHAPE_3D).astype(np.float32)
    vol = real + 1j * imag
    p = tmp_path / "vol_complex.npy"
    np.save(str(p), vol)
    return p


@pytest.fixture()
def volume_2d(tmp_path: Path) -> Path:
    """Create an invalid 2-D .npy file."""
    arr = np.zeros((10, 20), dtype=np.float32)
    p = tmp_path / "flat.npy"
    np.save(str(p), arr)
    return p


# ── plot_projections function tests ─────────────────────────────────────────


class TestPlotProjections:
    def test_creates_file_3d(self, tmp_path: Path) -> None:
        vol = np.random.default_rng(0).integers(0, 255, size=SHAPE_3D, dtype=np.uint8)
        out = tmp_path / "proj.png"
        plot_projections(vol, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_file_4d(self, tmp_path: Path) -> None:
        vol = np.random.default_rng(1).standard_normal(SHAPE_4D).astype(np.float32)
        out = tmp_path / "proj4d.png"
        plot_projections(vol, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_complex(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(2)
        vol = (rng.standard_normal(SHAPE_3D) + 1j * rng.standard_normal(SHAPE_3D)).astype(np.complex64)
        out = tmp_path / "proj_c.png"
        plot_projections(vol, save_path=out)
        assert out.exists()

    def test_rejects_2d(self) -> None:
        arr = np.zeros((10, 20))
        with pytest.raises(ValueError, match="3-D or 4-D"):
            plot_projections(arr)

    def test_rejects_1d(self) -> None:
        arr = np.zeros(100)
        with pytest.raises(ValueError, match="3-D or 4-D"):
            plot_projections(arr)

    def test_no_save_path_does_not_crash(self) -> None:
        vol = np.random.default_rng(3).integers(0, 255, size=SHAPE_3D, dtype=np.uint8)
        plot_projections(vol)  # should not raise

    def test_custom_title(self, tmp_path: Path) -> None:
        vol = np.random.default_rng(4).integers(0, 255, size=SHAPE_3D, dtype=np.uint8)
        out = tmp_path / "custom.png"
        plot_projections(vol, title="Custom Title", save_path=out)
        assert out.exists()


# ── CLI view command tests ──────────────────────────────────────────────────


class TestViewCLI:
    def test_view_3d_default_output(self, volume_3d: Path) -> None:
        result = runner.invoke(app, ["view", str(volume_3d)])
        assert result.exit_code == 0
        expected = volume_3d.with_name("vol3d_projections.png")
        assert expected.exists()
        assert expected.stat().st_size > 0

    def test_view_4d_default_output(self, volume_4d: Path) -> None:
        result = runner.invoke(app, ["view", str(volume_4d)])
        assert result.exit_code == 0
        expected = volume_4d.with_name("vol4d_projections.png")
        assert expected.exists()

    def test_view_custom_output(self, volume_3d: Path, tmp_path: Path) -> None:
        out = tmp_path / "subdir" / "my_proj.png"
        result = runner.invoke(app, ["view", str(volume_3d), "-o", str(out)])
        assert result.exit_code == 0
        assert out.exists()

    def test_view_complex_volume(self, volume_complex: Path) -> None:
        result = runner.invoke(app, ["view", str(volume_complex)])
        assert result.exit_code == 0
        expected = volume_complex.with_name("vol_complex_projections.png")
        assert expected.exists()

    def test_view_missing_file(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["view", str(tmp_path / "missing.npy")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_view_2d_rejects(self, volume_2d: Path) -> None:
        result = runner.invoke(app, ["view", str(volume_2d)])
        assert result.exit_code == 1
        assert "3-D" in result.output

    def test_view_output_message(self, volume_3d: Path) -> None:
        result = runner.invoke(app, ["view", str(volume_3d)])
        assert "saved" in result.output.lower() or "Saved" in result.output
