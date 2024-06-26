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
----------------

Install using pip
=================

    pip install nismod-snail

This should bring all dependencies with it. If any of these cause difficulties,
try using a `conda <https://docs.conda.io/en/latest/miniconda.html>`_ environment::

    conda env create -n snail_env \
        python=3.8 geopandas shapely rasterio python-igraph
    conda activate snail_env
    pip install nismod-snail

If all worked okay, you should be able to run python and import snail::

    $ python
    >>> import snail
    >>> help(snail)
    Help on package snail:

    NAME
        snail - snail - the spatial networks impact assessment library





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
      --output split_lines_with_raster_values.geojson


Split multiple vector feature files along the grids defined by multiple raster files, attributing all raster values::

   snail process -fs features.csv -rs rasters.csv

Where at a minimum, each CSV has a column `path` with the path to each file.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::
   :maxdepth: 1
   
   setup

.. toctree::
   :maxdepth: 1

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
