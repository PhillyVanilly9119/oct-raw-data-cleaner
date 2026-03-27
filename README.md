# oct-unscrambler

A high-performance utility designed to infer the structure of raw, serialized binary data from Optical Coherence Tomography (OCT) systems.

## Overview

In many research environments, OCT data is saved as raw bitstreams with minimal metadata. `oct-unscrambler` provides a professional API and CLI to automatically detect data types, A-scan lengths, and B-scan repetitions, allowing researchers to convert "unusable" binary dumps into structured arrays for post-processing.

The core periodicity detector uses the autocorrelation of the signal envelope:

$$R_{xx}(\tau) = \sum_{n} x[n]\, x^{*}[n - \tau]$$

## Features

* **Memory-Mapped I/O:** Efficiently handles multi-gigabyte files without exhausting system RAM.
* **Entropy-Based Dtype Detection:** Automatically identifies the most plausible data type by minimising first-difference entropy.
* **Autocorrelation Periodicity:** Recovers A-scan lengths from the signal envelope's autocorrelation function.
* **Unified CLI:** A simple `oct-unscramble probe` command for rapid data exploration.
* **Diagnostic Visuals:** Generates alignment plots, autocorrelation curves, and sample B-scans.
* **Headless-Safe:** Matplotlib backend defaults to `Agg`; CLI works on servers without a display.

## Project Structure

```
oct-unscrambler/
├── .github/workflows/ci.yml
├── src/oct_unscrambler/
│   ├── __init__.py
│   ├── cli.py         # Typer entry-point
│   ├── core.py        # BinaryProber, mmap, dtype inference
│   ├── signal.py      # Autocorrelation & coherence
│   └── utils.py       # Plotting helpers
├── tests/
│   └── test_inference.py
├── Dockerfile
├── build_exe.py
├── pyproject.toml
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/PhillyVanilly9119/oct-raw-data-cleaner.git
cd oct-raw-data-cleaner

# Install in editable mode (with dev dependencies)
pip install -e ".[dev]"
```

## Usage

### CLI

```bash
# Basic probe
oct-unscramble probe data/raw_volume.bin

# With diagnostic plots saved to ./plots/
oct-unscramble probe data/raw_volume.bin --plot --output-dir plots
```

### Python API

```python
from oct_unscrambler.core import BinaryProber

with BinaryProber("data/raw_volume.bin") as prober:
    result = prober.probe()

print(result.best_dtype)     # e.g. uint16
print(result.a_scan_length)  # e.g. 1024
print(result.b_scan_repeats) # e.g. 500
```

## Docker

```bash
docker build -t oct-unscrambler .
docker run --rm -v "$PWD/data:/data" oct-unscrambler probe /data/raw_volume.bin
```

## Standalone Executable

```bash
pip install ".[dev]"   # includes PyInstaller
python build_exe.py    # produces dist/oct-unscramble[.exe]
```

## Testing

```bash
pytest --cov=oct_unscrambler tests/
```

## License

MIT
