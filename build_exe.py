"""Build a standalone executable using PyInstaller.

Run:
    python build_exe.py

The resulting binary will be in ``dist/oct-unscramble[.exe]``.
"""

from __future__ import annotations

import platform
import subprocess
import sys


def main() -> None:
    name = "oct-unscramble"
    entry = "src/oct_unscrambler/cli.py"

    cmd: list[str] = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--onefile",
        "--name",
        name,
        "--noconfirm",
        "--clean",
    ]

    # Collect heavy deps as hidden imports (numba/numpy may need help)
    for mod in ("numpy", "scipy", "matplotlib", "typer", "rich"):
        cmd += ["--hidden-import", mod]

    if platform.system() == "Windows":
        cmd.append("--console")

    cmd.append(entry)

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    print(f"\nDone — executable is in dist/{name}")


if __name__ == "__main__":
    main()
