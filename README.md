<p align="center">
<a href="https://github.com/nismod/snail/tree/main/tutorials">Tutorials</a> |
<a href="https://github.com/nismod/snail/issues">Issues</a>
</p>

<p align="center">
<img src="./images/snail.svg" alt="snail" />
</p>

[![PyPI version](https://img.shields.io/pypi/v/nismod-snail.svg)](https://pypi.org/project/nismod-snail/)
[![Build](https://github.com/nismod/snail/actions/workflows/build.yml/badge.svg)](https://github.com/nismod/snail/actions/workflows/build.yml)
[![License](https://img.shields.io/pypi/l/nismod-snail.svg)](https://opensource.org/licenses/MIT)

# 🤔 What is this?

This is a Python package to help with analysis of the potential impacts of
climate hazards and other perils on infrastructure networks.

## Installation

Install using pip:

    pip install nismod-snail

This should bring all essential dependencies with it.

If any of these cause difficulties, try using a
[conda](https://docs.conda.io/en/latest/miniconda.html) environment:

    conda env create -n snail_env \
        python=3.11 geopandas shapely rasterio python-igraph
    conda activate snail_env
    pip install nismod-snail

If all worked okay, you should be able to run python and import snail:

    $ python
    >>> import snail
    >>> help(snail)
    Help on package snail:

    NAME
        snail - snail - the spatial networks impact assessment library

## Using the `snail` command

Once installed, you can use `snail` directly from the command line.

Split features on a grid defined by its transform, width and height:

```bash
snail split \
    --features input.shp \
    --transform 1 0 -180 0 -1 90 \
    --width 360 \
    --height 180 \
    --output split.gpkg
```

Split features on a grid defined by a GeoTIFF, optionally adding the values from
each raster band to each split feature as a new attribute:

```bash
snail split \
    --features lines.geojson \
    --raster gridded_data.tif \
    --attribute \
    --lazy-rasters \
    --output split_lines_with_raster_values.geojson
```

Adding `--lazy-rasters` keeps large raster bands on disk and fetches values
lazily via `xarray`/`dask`.

Split multiple vector feature files along the grids defined by multiple raster
files, attributing all raster values:

```bash
snail process -fs features.csv -rs rasters.csv
```

Where at a minimum, each CSV has a column `path` with the path to each file.

The `--lazy-rasters` flag can be supplied to `snail process` when working with
large rasters.

### Transform

A note on `transform` - these six numbers define the transform from `i,j` cell
index (column/row) coordinates in the rectangular grid to `x,y` geographic
coordinates, in the coordinate reference system of the input and output files.
They effectively form the first two rows of a 3x3 matrix:

```
| x |   | a  b  c | | i |
| y | = | d  e  f | | j |
| 1 |   | 0  0  1 | | 1 |
```

In cases without shear or rotation, `a` and `e` define scaling or grid cell
size, while `c` and `f` define the offset or grid upper-left corner:

```
| x_scale 0       x_offset |
| 0       y_scale y_offset |
| 0       0       1        |
```

See [`rasterio/affine`](https://github.com/rasterio/affine#usage) and [GDAL Raster Data Model](https://gdal.org/user/raster_data_model.html#affine-geotransform) for more documentation.

## Damage curve library

`snail.damage_library` bundles a curated set of infrastructure damage curves
from [Nirandjan et al. (2023)](https://doi.org/10.5281/zenodo.10203846)
under the terms of the Creative Commons Attribution 4.0 license. The data are
installed alongside the package, so no extra download step is required.

```python
>>> from snail import damage_library
>>> damage_library.available_curves(hazard="flood").head()["curve_id"].tolist()
['F1.1', 'F1.2', 'F1.3', 'F1.4', 'F1.5']
>>> metadata = damage_library.get_metadata("F1.1")
>>> metadata.exposed_element
'Small power plants, capacity <100 MW'
>>> curve = damage_library.load_curve("F1.1")
>>> curve.damage_fraction([0.0, 0.5, 1.0])
array([0.     , 0.410105, 0.82021 ])
```

## Development

Clone this repository using [GitHub Desktop](https://desktop.github.com/) or on
the command line:

    git clone git@github.com:nismod/snail.git

Change directory into the root of the project:

    cd snail

To create and activate a conda environment with snail's dependencies installed:

    conda env create -f environment.yml
    conda activate snail-dev

Run this to install the source code as a package:

    pip install .

If you're working on snail itself, install it as "editable" along with test and
development packages:

    pip install -e .[dev,docs,tutorials]

Run tests using [pytest](https://docs.pytest.org/en/latest/) and
[pytest-cov](https://pytest-cov.readthedocs.io) to check coverage:

    pytest --cov=snail --cov-report=term-missing

Run a formatter ([black](https://github.com/psf/black)) to fix code
formatting:

    black src/snail

Build the docs:

    rm -r docs/build
    rm -r docs/source/tutorials
    cp -r tutorials docs/source/
    pushd docs
    sphinx-apidoc -M -o source/api ../src/snail/ --force
    make html
    popd

Serve the HTML docs locally:

    cd docs/build/html
    python -m http.server

### Benchmarks

The repository includes benchmarks for the polygon and linestring splitting
implementations. Run these commands from the repository root after installing
the development environment and building the C++ targets:

```bash
cmake -Bbuild ./extension
cmake --build build
./build/run_benchmarks
```

This measures the C++ splitting core without Python conversion overhead. For a
Python-level comparison, run:

```bash
python scripts/benchmark_split.py
```

The Python benchmark still compares `split_polygons` (the Shapely/GEOS overlay
implementation) with `split_polygons_experimental` (the C++ per-cell
implementation). It reports timings and piece counts, and checks that both
implementations conserve polygon area. Results are machine-dependent; use the
same environment and workload when comparing changes.

Recent run in the `snail-dev` environment:

| Workload | Overlay | Experimental | Speedup | Pieces |
| --- | ---: | ---: | ---: | ---: |
| 500 small buildings | 119.7 ms | 2.8 ms | 42.7× | 725 |
| 50 medium circles | 458.5 ms | 11.9 ms | 38.4× | 18,051 |
| 1 large circle | 215.8 ms | 4.5 ms | 47.5× | 6,528 |

### C++ library

The C++ library in `extension/src` contains the core routines to find intersections of
lines with raster grids.

Before working on the C++ library, fetch source code for Catch2 unit testing
library (this is included as a git submodule):

    git submodule update --init --recursive

Build the library and run tests:

    cmake -Bbuild ./extension
    cmake --build build/
    ./build/run_tests

Run code style auto-formatting:

    clang-format -i extension/src/*.{cpp,hpp}

Run lints and checks:

    clang-tidy --checks 'cppcoreguidelines-*' extension/src/*.{cpp,hpp}

This may need some includes for `pybind11` - which will vary depending on your
python installation. For example, with python via miniconda:

    clang-tidy --checks 'cppcoreguidelines-*' extension/src/* -- \
        -I/home/username/miniconda3/include/python3.12/ \
        -I./pybind11/include/

Or with C++ headers installed on a Linux machine:

    clang-tidy --checks 'cppcoreguidelines-*' extension/src/* -- \
        -std=c++14  \
        -I/usr/include/x86_64-linux-gnu/c++/11 \
        -I/usr/include/c++/11 \
        -I{$PWD}/extension/extern/pybind11/include \
        -I/usr/include/python3.12


### Integration of C++ and Python using pybind11

The `snail.core.intersections` module is built using `pybind11` with
`scikit-build-core` (see [docs](https://scikit-build-core.readthedocs.io/en/latest/))

- `extension/src/intersections.cpp` defines the module interface using the
  `PYBIND11_MODULE` macro
- `pyproject.toml` defines the build requirements for snail, which includes
  pybind11 and scikit-build-core

## Acknowledgments

> MIT License
>
> Copyright (c) 2020-23 Tom Russell and all [snail contributors](https://github.com/nismod/snail/graphs/contributors)

This library is developed by researchers in the [Oxford Programme for Sustainable
Infrastructure Systems](https://opsis.eci.ox.ac.uk/) at the University of Oxford,
funded by multiple research projects.

This research received funding from the FCDO Climate Compatible Growth Programme.
The views expressed here do not necessarily reflect the UK government's official
policies.
