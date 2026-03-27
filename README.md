# OCT Uncrambler

A high-performance utility designed to infer the structure of raw, serialized binary data from Optical Coherence Tomography (OCT) systems.

## Overview

In many research environments, OCT data is saved as raw bitstreams with minimal metadata. `oct-unscrambler` provides a professional API and CLI to automatically detect data types, A-scan lengths, and B-scan repetitions, allowing researchers to convert "unusable" binary dumps into structured arrays for post-processing.

## Features

* **Memory-Mapped I/O:** Efficiently handles multi-gigabyte files without exhausting system RAM.
* **Structural Inference:** Automatically identifies data types (dtype) and acquisition periodicities.
* **Unified CLI:** A simple command-line interface for rapid data probing and previewing.
* **Diagnostic Visuals:** Generates alignment plots and sample B-scans to verify data integrity.

## Installation

```bash
# Clone the repository
git clone [https://github.com/your-username/oct-unscrambler.git](https://github.com/your-username/oct-unscrambler.git)
cd oct-unscrambler

# Install in editable mode
pip install -e .