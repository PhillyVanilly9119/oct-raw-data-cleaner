"""CLI entry-point — Typer application for oct-unscrambler."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from oct_unscrambler.core import BinaryProber

app = typer.Typer(
    name="oct-unscramble",
    help="Infer and unpack raw binary OCT data structures.",
    add_completion=False,
)
console = Console()


def _resolve_dtype(raw: str | None) -> np.dtype | None:
    """Convert a user-supplied dtype string to a numpy dtype, or None."""
    if raw is None:
        return None
    try:
        return np.dtype(raw)
    except TypeError:
        console.print(f"[red]Error:[/red] invalid dtype '{raw}'")
        raise typer.Exit(code=1) from None


# ── probe command ────────────────────────────────────────────────────────────


@app.command()
def probe(
    path: Path = typer.Argument(..., help="Path to a raw binary OCT file."),
    dtype: Optional[str] = typer.Option(
        None, "--dtype", "-d", help="Force a numpy dtype (e.g. uint16)."
    ),
    ascan: Optional[int] = typer.Option(
        None, "--ascan", "-a", help="A-scan depth (samples)."
    ),
    bscan: Optional[int] = typer.Option(
        None, "--bscan", "-b", help="B-scan width (# of A-scans)."
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Export structured .npy to this path.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip validation; use user values as-is.",
    ),
    plot: bool = typer.Option(
        False, "--plot", "-p", help="Save diagnostic plots."
    ),
    plot_dir: Path = typer.Option(
        Path("."), "--plot-dir", help="Directory for plot output."
    ),
) -> None:
    """Probe a binary file: detect dtype, A-scan length, and B-scan count.

    Optionally export to a structured .npy file with --output.
    """
    if not path.is_file():
        console.print(f"[red]Error:[/red] file not found: {path}")
        raise typer.Exit(code=1)

    resolved_dtype = _resolve_dtype(dtype)

    import warnings as _w

    with BinaryProber(path) as prober:
        with _w.catch_warnings():
            _w.simplefilter("ignore", UserWarning)
            result = prober.probe(
                dtype=resolved_dtype,
                ascan_length=ascan,
                bscan_length=bscan,
                force=force,
            )

        # ── Validation feedback ──────────────────────────────────────────
        if ascan is not None and not force:
            envelope = prober._envelope(result.best_dtype)
            vr = prober.validate_params(envelope, ascan)
            if not vr.plausible:
                console.print(
                    f"[yellow]Warning:[/yellow] {vr.message}"
                )
                if plot and vr.detected_ascan:
                    from oct_unscrambler.utils import (
                        plot_alignment_comparison,
                    )

                    plot_dir.mkdir(parents=True, exist_ok=True)
                    cmp_path = plot_dir / f"{path.stem}_alignment.png"
                    plot_alignment_comparison(
                        envelope,
                        ascan,
                        vr.detected_ascan,
                        save_path=cmp_path,
                    )
                    console.print(
                        f"[green]Comparison plot → {cmp_path}[/green]"
                    )

        # ── Summary table ────────────────────────────────────────────────
        table = Table(title=f"Probe Results — {result.path.name}")
        table.add_column("Property", style="bold")
        table.add_column("Value")

        table.add_row("File size", f"{result.file_bytes:,} bytes")
        table.add_row("Best dtype", str(result.best_dtype))
        table.add_row(
            "A-scan length",
            str(result.a_scan_length) if result.a_scan_length else "n/d",
        )
        table.add_row(
            "B-scan repeats",
            str(result.b_scan_repeats) if result.b_scan_repeats else "n/a",
        )
        if result.bscan_length:
            table.add_row("B-scan width", str(result.bscan_length))
        if result.n_repeats:
            table.add_row("Functional repeats", str(result.n_repeats))
        console.print(table)

        # ── Per-dtype scores ─────────────────────────────────────────────
        if result.dtype_scores:
            score_table = Table(title="Dtype Scores")
            score_table.add_column("dtype")
            score_table.add_column("Score", justify="right")
            for dt, sc in sorted(
                result.dtype_scores.items(), key=lambda t: -t[1]
            ):
                score_table.add_row(dt, f"{sc:.4f}")
            console.print(score_table)

        # ── Optional diagnostic plots ────────────────────────────────────
        if plot:
            from oct_unscrambler.signal import autocorrelation
            from oct_unscrambler.utils import (
                plot_autocorrelation,
                plot_bscan_preview,
                plot_dtype_scores,
            )

            plot_dir.mkdir(parents=True, exist_ok=True)
            stem = result.path.stem

            if result.dtype_scores:
                plot_dtype_scores(
                    result.dtype_scores,
                    save_path=plot_dir / f"{stem}_dtype_scores.png",
                )

            envelope = prober._envelope(result.best_dtype)
            acf = autocorrelation(
                envelope, max_lag=min(len(envelope), 8192)
            )
            plot_autocorrelation(
                acf,
                save_path=plot_dir / f"{stem}_autocorrelation.png",
            )

            if result.a_scan_length:
                plot_bscan_preview(
                    envelope,
                    result.a_scan_length,
                    n_bscans=min(result.b_scan_repeats or 1, 16),
                    save_path=plot_dir / f"{stem}_bscan_preview.png",
                )

            console.print(
                f"[green]Plots saved to {plot_dir.resolve()}[/green]"
            )

        # ── Optional .npy export ─────────────────────────────────────────
        if output is not None:
            a = result.a_scan_length
            b = result.bscan_length or result.b_scan_repeats or 1
            r = result.n_repeats or 1
            if a is None:
                console.print(
                    "[red]Error:[/red] cannot export — "
                    "A-scan length unknown."
                )
                raise typer.Exit(code=1)
            npy_path = prober.save_as_npy(
                output,
                dtype=result.best_dtype,
                a_scan_length=a,
                bscan_length=b,
                n_repeats=r,
            )
            shape_str = _npy_shape_str(npy_path)
            console.print(
                f"[green]Saved {npy_path} {shape_str}[/green]"
            )


def _npy_shape_str(npy_path: Path) -> str:
    """Read shape/dtype from .npy without loading the array."""
    loaded = np.load(str(npy_path), mmap_mode="r")
    return f"shape={loaded.shape} dtype={loaded.dtype}"


def main() -> None:
    """Package entry-point (called by ``oct-unscramble`` console script)."""
    app()


if __name__ == "__main__":
    main()
