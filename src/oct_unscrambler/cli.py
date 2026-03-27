"""CLI entry-point — Typer application for oct-unscrambler."""

from __future__ import annotations

from pathlib import Path

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


@app.command()
def probe(
    path: Path = typer.Argument(..., help="Path to a raw binary OCT file."),
    plot: bool = typer.Option(False, "--plot", "-p", help="Save diagnostic plots."),
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory for output plots.",
    ),
) -> None:
    """Probe a binary file: detect dtype, A-scan length, and B-scan count."""
    if not path.is_file():
        console.print(f"[red]Error:[/red] file not found: {path}")
        raise typer.Exit(code=1)

    with BinaryProber(path) as prober:
        result = prober.probe()

    # ── Summary table ────────────────────────────────────────────────────
    table = Table(title=f"Probe Results — {result.path.name}")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("File size", f"{result.file_bytes:,} bytes")
    table.add_row("Best dtype", str(result.best_dtype))
    table.add_row(
        "A-scan length",
        str(result.a_scan_length) if result.a_scan_length else "not detected",
    )
    table.add_row(
        "B-scan repeats",
        str(result.b_scan_repeats) if result.b_scan_repeats else "n/a",
    )
    console.print(table)

    # ── Per-dtype scores ─────────────────────────────────────────────────
    score_table = Table(title="Dtype Scores")
    score_table.add_column("dtype")
    score_table.add_column("Score", justify="right")
    for dt, score in sorted(result.dtype_scores.items(), key=lambda t: -t[1]):
        score_table.add_row(dt, f"{score:.4f}")
    console.print(score_table)

    # ── Optional plots ───────────────────────────────────────────────────
    if plot:
        import numpy as np

        from oct_unscrambler.signal import autocorrelation
        from oct_unscrambler.utils import (
            plot_autocorrelation,
            plot_bscan_preview,
            plot_dtype_scores,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        stem = result.path.stem

        plot_dtype_scores(
            result.dtype_scores,
            save_path=output_dir / f"{stem}_dtype_scores.png",
        )

        n_elements = result.file_bytes // result.best_dtype.itemsize
        with BinaryProber(path) as p2:
            view = np.ndarray(n_elements, dtype=result.best_dtype, buffer=p2.mmap)
            if np.issubdtype(result.best_dtype, np.complexfloating):
                envelope = np.abs(view).astype(np.float64)
            else:
                envelope = np.real(view).astype(np.float64)

            acf = autocorrelation(envelope, max_lag=min(len(envelope), 8192))
            plot_autocorrelation(
                acf,
                save_path=output_dir / f"{stem}_autocorrelation.png",
            )

            if result.a_scan_length:
                plot_bscan_preview(
                    envelope,
                    result.a_scan_length,
                    n_bscans=min(result.b_scan_repeats or 1, 16),
                    save_path=output_dir / f"{stem}_bscan_preview.png",
                )

        console.print(f"[green]Plots saved to {output_dir.resolve()}[/green]")


def main() -> None:
    """Package entry-point (called by ``oct-unscramble`` console script)."""
    app()


if __name__ == "__main__":
    main()
