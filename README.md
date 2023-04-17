<p align="center">
<a href="https://github.com/nismod/snail/tree/master/tutorials">Tutorials</a> |
<a href="https://github.com/nismod/snail/issues">Issues</a>
</p>

<p align="center">
<img src="./images/snail.svg" alt="snail" />
</p>

[![Build](https://github.com/nismod/snail/actions/workflows/build.yml/badge.svg)](https://github.com/nismod/snail/actions/workflows/build.yml)

> This code is under early development

# 🤔 What is this?

This is a Python package to help with analysis of the potential impacts of
climate hazards and other perils on infrastructure networks.

## Installation

Install using pip:

    pip install nismod-snail

This should bring all dependencies with it. If any of these cause difficulties,
try using a [conda](https://docs.conda.io/en/latest/miniconda.html) environment:

    conda env create -n snail_env \
        python=3.8 geopandas shapely rasterio python-igraph
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

Split features on a grid defined by a GeoTIFF, optionally adding the values from each raster band to each split feature as a new attribute:

```bash
snail split \
    --features lines.geojson \
    --raster gridded_data.tif \
    --attribute \
    --output split_lines_with_raster_values.geojson
```

Split multiple vector feature files along the grids defined by multiple raster files, attributing all raster values:

```bash
snail process -fs features.csv -rs rasters.csv
```

Where at a minimum, each CSV has a column `path` with the path to each file.

### Transform

A note on `transform` - these six numbers define the transform from `i,j` cell index (column/row) coordinates in the rectangular grid to `x,y` geographic coordinates, in the coordinate reference system of the input and output files. They effectively form the first two rows of a 3x3 matrix:

```
| x |   | a  b  c | | i |
| y | = | d  e  f | | j |
| 1 |   | 0  0  1 | | 1 |
```

In cases without shear or rotation, `a` and `e` define scaling or grid cell size, while `c` and `f` define the offset or grid upper-left corner:

```
| x_scale 0       x_offset |
| 0       y_scale y_offset |
| 0       0       1        |
```

See [`rasterio/affine`](https://github.com/rasterio/affine#usage) and [GDAL Raster Data Model](https://gdal.org/user/raster_data_model.html#affine-geotransform) for more documentation.

## Development

Clone this repository using [GitHub Desktop](https://desktop.github.com/) or on
the command line:

    git clone git@github.com:nismod/snail.git

Change directory into the root of the project:

    cd snail

To create and activate a conda environment with snail's dependencies installed:

    conda env create -f .environment.yml
    conda activate snail-dev

Run this to install the source code as a package:

    pip install .

If you're working on snail itself, install it as "editable" along with test and
development packages:

    pip install -e .[dev]

Run tests using [pytest](https://docs.pytest.org/en/latest/) and
[pytest-cov](https://pytest-cov.readthedocs.io) to check coverage:

    pytest --cov=snail --cov-report=term-missing

Run a formatter ([black](https://github.com/psf/black)) to fix code
formatting:

    black src/snail

When working on the tutorial notebooks, it is recommended to install and
configure [nbstripout](https://github.com/kynan/nbstripout) so data and outputs
are not committed in the notebook files:

    nbstripout --install

### C++ library

The C++ library in `src/cpp` contains the core routines to find intersections of
lines with raster grids.

Before working on the C++ library, fetch source code for Catch2 unit testing
library (this is included as a git submodule):

    git submodule update --init --recursive

Build the library and run tests:

    cmake -Bbuild .
    cmake --build build/
    ./build/run_tests

Run code style auto-formatting:

    clang-format -i src/cpp/*.hpp

Run lints and checks:

    clang-tidy --checks 'cppcoreguidelines-*' src/cpp/*.hpp

This may need some includes for `pybind11` - which will vary depending on your
python installation. For example, with python via miniconda:

    clang-tidy --checks 'cppcoreguidelines-*' src/cpp/* -- \
        -I/home/username/miniconda3/include/python3.7m/ \
        -I./pybind11/include/

### Integration of C++ and Python using pybind11

The `snail.core.intersections` module is built using `pybind11` with
`setuptools` (see [docs](https://pybind11.readthedocs.io/en/stable/compiling.html#building-with-setuptools))

- `src/cpp/intersections.cpp` defines the module interface using the
  `PYBIND11_MODULE` macro
- `pyproject.toml` defines the build requirements for snail, which includes
  pybind11, wheel and setuptools
- `setup.py` defines the `Pybind11Extension` module to build - both the C++
  files to compile, and the location of the built module within the python
  package
