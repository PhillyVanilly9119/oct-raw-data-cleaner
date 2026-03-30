"""Plotting utilities — diagnostic visuals for OCT data inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# matplotlib is imported lazily so the CLI stays usable in headless environments.
_MPL_AVAILABLE = True
try:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe backend
    import matplotlib.pyplot as plt
except ImportError:
    _MPL_AVAILABLE = False


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError(
            "matplotlib is required for plotting.  Install it with: "
            "pip install matplotlib"
        )


def plot_autocorrelation(
    acf: np.ndarray,
    *,
    title: str = "Autocorrelation",
    save_path: str | Path | None = None,
) -> None:
    """Plot an autocorrelation curve and optionally save to disk."""
    _require_mpl()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acf, linewidth=0.6)
    ax.set_xlabel("Lag (samples)")
    ax.set_ylabel("R_xx")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_bscan_preview(
    data: np.ndarray,
    a_scan_length: int,
    *,
    n_bscans: int = 1,
    title: str = "B-Scan Preview",
    save_path: str | Path | None = None,
) -> None:
    """Render one or more B-scans as a 2-D image."""
    _require_mpl()
    total_samples = a_scan_length * n_bscans
    segment = data[:total_samples]
    if n_bscans > 1:
        image = segment.reshape(n_bscans, a_scan_length)
    else:
        image = segment.reshape(1, -1)

    if np.issubdtype(image.dtype, np.complexfloating):
        image = np.abs(image)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(
        np.real(image).astype(np.float64),
        aspect="auto",
        cmap="gray",
        interpolation="nearest",
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("B-Scan #")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_dtype_scores(
    scores: dict[str, float],
    *,
    title: str = "Dtype Entropy Scores",
    save_path: str | Path | None = None,
) -> None:
    """Bar chart of per-dtype entropy scores."""
    _require_mpl()
    labels = list(scores.keys())
    values = [scores[k] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, values)
    ax.set_xlabel("Score (higher = more likely)")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_alignment_comparison(
    envelope: np.ndarray,
    user_ascan: int,
    detected_ascan: int,
    *,
    n_bscans: int = 8,
    title: str = "Alignment Comparison",
    save_path: str | Path | None = None,
) -> None:
    """Side-by-side B-scan images using user vs. detected A-scan length.

    Displayed when validation fails so the user can visually compare
    which alignment is correct.
    """
    _require_mpl()
    fig, (ax_user, ax_det) = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ascan, label in [
        (ax_user, user_ascan, f"User ({user_ascan})"),
        (ax_det, detected_ascan, f"Detected ({detected_ascan})"),
    ]:
        n = min(n_bscans, len(envelope) // ascan) if ascan > 0 else 0
        if n < 1:
            ax.set_title(f"{label} — insufficient data")
            continue
        segment = envelope[: n * ascan].reshape(n, ascan)
        ax.imshow(segment, aspect="auto", cmap="gray", interpolation="nearest")
        ax.set_xlabel("Depth sample")
        ax.set_ylabel("B-Scan #")
        ax.set_title(label)

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def plot_projections(
    volume: np.ndarray,
    *,
    title: str = "Max-Intensity Projections",
    save_path: str | Path | None = None,
) -> None:
    """Save a 3-panel figure with max-intensity projections along each axis.

    Parameters
    ----------
    volume : np.ndarray
        A 3-D array (Z, Y, X).  If 4-D the first axis is squeezed
        (first volume taken).
    title : str
        Super-title for the figure.
    save_path : path, optional
        If given the figure is saved to disk.
    """
    _require_mpl()

    # Normalise to 3-D
    if volume.ndim == 4:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(
            f"Expected a 3-D or 4-D array, got shape {volume.shape}"
        )

    if np.issubdtype(volume.dtype, np.complexfloating):
        volume = np.abs(volume)
    volume = np.real(volume).astype(np.float64)

    axis_labels = [
        ("Z (depth)", "Y", "X"),
        ("Y (B-scan)", "Z", "X"),
        ("X (A-scan)", "Z", "Y"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax_idx, (ax, (proj_name, ylabel, xlabel)) in enumerate(
        zip(axes, axis_labels)
    ):
        projection = np.max(volume, axis=ax_idx)
        ax.imshow(projection, aspect="auto", cmap="gray", interpolation="nearest")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"MIP along {proj_name}")

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(str(save_path), dpi=150)
    plt.close(fig)
