==========================================
spatial networks impact assessment library
==========================================

.. raw:: html

    <div align="center">

.. image:: ../../images/snail.svg
    :alt: snail

.. raw:: html

    </div>


snail is a Python package to help with analysis of the potential impacts of
climate hazards on infrastructure networks.

.. image:: https://img.shields.io/badge/github-snail-brightgreen.svg
    :target: https://github.com/nismod/snail/
    :alt: snail on github

.. image:: https://img.shields.io/pypi/l/nismod-snail.svg
    :target: https://opensource.org/licenses/MIT
    :alt: License

.. image:: https://github.com/nismod/snail/actions/workflows/build.yml/badge.svg
    :target: https://github.com/nismod/snail/actions/workflows/build.yml
    :alt: Build

.. image:: https://img.shields.io/pypi/v/nismod-snail.svg
    :target: https://pypi.org/project/nismod-snail/
    :alt: PyPI version


Installation
------------

Install using pip::

    pip install nismod-snail

This should bring all dependencies with it. If any of these cause difficulties,
try using a `micromamba
<https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html>`_
environment::

    micromamba env create -n snail \
        python=3.13 geopandas shapely rasterio python-igraph
    micromamba activate snail
    pip install nismod-snail

If all worked okay, you should be able to run python and import snail::

    $ python
    >>> import snail
    >>> help(snail)
    Help on package snail:

    NAME
        snail - snail - the spatial networks impact assessment library


Using snail as a Python library
-------------------------------

The high-level :func:`snail.overlay_raster` and :func:`snail.overlay_rasters`
functions split vector features (points, lines or polygons) along the cells
of a raster grid and attribute the raster cell values to each split feature::

    >>> import geopandas
    >>> import snail

    >>> features = geopandas.read_file("lines.geojson")
    >>> splits = snail.overlay_raster(features, "gridded_data.tif")
    >>> splits.to_file("split_lines_with_raster_values.gpkg")

The result contains one row per split feature (each feature is split wherever
it crosses a raster cell boundary), with the input feature attributes, cell
indices in columns ``index_i`` and ``index_j``, and one column of raster
values per band. If the features and raster are in different coordinate
reference systems, the features are implicitly reprojected to the raster CRS
for splitting and value lookup, then returned in their original CRS.

:func:`snail.overlay_rasters` intersects all features with all rasters in one
call, splitting on each distinct grid and attributing one column per raster
band. Lower-level building blocks (grid definitions, splitting, cell
indexing, value lookup) are available in :mod:`snail.intersection`, and
helpers for reading and writing files in :mod:`snail.io`.


Using the `snail` command
-------------------------

Once installed, you can use `snail` directly from the command line.

Split features on a grid defined by its transform, width and height::

    snail split \
        --features input.shp \
        --transform 1 0 -180 0 -1 90 \
        --width 360 \
        --height 180 \
        --output split.gpkg


Split features on a grid defined by a GeoTIFF, optionally adding the values from each raster band to each split feature as a new attribute::

    snail split \
        --features lines.geojson \
        --raster gridded_data.tif \
        --attribute \
        --lazy-rasters \
        --output split_lines_with_raster_values.geojson

Add ``--lazy-rasters`` to keep raster bands on disk and fetch values lazily via
``xarray``/``dask``.


Split multiple vector feature files along the grids defined by multiple raster files, attributing all raster values::

    snail process -fs features.csv -rs rasters.csv

Where at a minimum, each CSV has a column `path` with the path to each file.

`snail process` calculates all features intersected with all rasters: each
row of the features CSV is split against every distinct raster grid and
produces one output file, with one column of attributed values per raster
file/band. See the project README for the full set of optional CSV columns
(``layer``, ``output_path``, ``bands``, ``key``).


.. toctree::
    :maxdepth: 2
    :caption: Contents:

Contents
--------

.. toctree::
    :maxdepth: 1

    setup

.. toctree::
    :maxdepth: 2

    Tutorials <tutorials>

.. toctree::
    :maxdepth: 3

    Reference <api/modules>

.. toctree::
    :maxdepth: 1

    License <license>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
