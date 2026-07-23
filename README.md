# PCWannier

[中文说明](README.zh.md)

PCWannier is a Python program for constructing photonic-crystal Wannier tight-binding models from numerical eigenmode data. It currently targets two-dimensional Bloch datasets and can generate localized Wannier functions, hopping matrices, interpolated bands, and optional topology results.

## Installation

PCWannier requires Python 3.10 or later. Install it from the project directory:

```bash
pip install -e .
```

Optional acceleration dependencies can be installed with:

```bash
pip install -e ".[numba,performance]"
```

## Usage

Prepare an `incar` file that describes the lattice, k-point grid, band window, projections, and paths to the mesh, field, material, and eigenvalue files. The numerical dataset is supplied by the user.

Run a calculation with:

```bash
pcwannier -i path/to/incar --out path/to/output
```

The module form is equivalent:

```bash
python -m pcwannier -i path/to/incar --out path/to/output
```

Common options:

```text
-t N                  use N worker threads
--backend auto         automatically select the compute backend
--cache                reuse cached calculation matrices
-b                     plot projection functions and exit
--interp points.txt    interpolate results on a supplied point mesh
```

Use `pcwannier --help` to view all command-line options. Typical outputs include Wannier functions, hopping matrices, band data and figures, topology figures, and a run log.
