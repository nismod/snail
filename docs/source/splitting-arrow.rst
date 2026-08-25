Splitting large datasets
=========================

:func:`snail.intersection.split_linestrings` and
:func:`snail.intersection.split_polygons` (and the ``snail split``/``snail
process`` commands built on them) already do everything on this page for
you: they take a :class:`~geopandas.GeoDataFrame` and give one back. Read
on only if you want to split a source that is not already loaded as a
GeoDataFrame - a GeoParquet file bigger than memory, say - or you are
working directly against the compiled extension.

How a split runs
-----------------

Geometries cross into the C++ extension as `GeoArrow
<https://geoarrow.org/>`_, over the `Arrow C stream interface
<https://arrow.apache.org/docs/format/CStreamInterface.html>`_: the
extension pulls one batch of geometries at a time from its source, splits
it, and hands the pieces back the same way, as a stream of record batches.
Nothing is split until that stream is read, and only one batch of the
source is ever held at once - so a file far larger than memory can be
split by a consumer that takes the results as they come, without loading
the whole thing first.

:func:`snail.core.intersections.split_linestrings` and
:func:`~snail.core.intersections.split_polygons` accept the geometries from
*any* object implementing the Arrow PyCapsule stream interface
(``__arrow_c_stream__``) or the single-array interface
(``__arrow_c_array__``, read as a stream of one batch) - a
:class:`pyarrow.ChunkedArray`, :class:`~pyarrow.Table` or
:class:`~pyarrow.RecordBatchReader`, a GeoParquet or
:class:`pyarrow.dataset.Dataset` reader, or
:meth:`GeoSeries.to_arrow() <geopandas.GeoSeries.to_arrow>`.

The geometries themselves must be **GeoArrow-encoded** -
``geoarrow.linestring`` or ``geoarrow.polygon``. Coordinates may be
interleaved (as geopandas exports them) or separated into x and y arrays
(as GeoParquet stores them), in two or more dimensions; all are read
directly, with no per-feature Python object built on either side of the
interface. Well-known binary (``geoarrow.wkb``) is **not** read, and nor
are multi-part geometries - both are rejected with an explanatory error.

.. note::

   This is the catch when reading a file. :meth:`GeoDataFrame.to_parquet()
   <geopandas.GeoDataFrame.to_parquet>` writes ``geoarrow.wkb`` **by
   default**, and such a file cannot be split. Pass
   ``geometry_encoding="geoarrow"`` when writing, as the example below
   does. To split a WKB-encoded file, read it with
   :func:`geopandas.read_parquet` and use
   :func:`snail.intersection.split_linestrings` instead - that path goes
   through shapely and has no such restriction, at the cost of holding the
   geometries in memory.

Splitting a file directly
--------------------------

This reads a GeoParquet file of linestrings in batches and splits each
batch as it arrives, never holding more than one batch of geometries and
one batch of pieces at a time::

    import pyarrow
    import pyarrow.parquet
    from snail.core.intersections import split_linestrings
    from snail.intersection import GridDefinition

    grid = GridDefinition.from_raster("hazard.tif")

    parquet_file = pyarrow.parquet.ParquetFile("edges.geoparquet")
    geometry_batches = pyarrow.RecordBatchReader.from_batches(
        parquet_file.schema_arrow, parquet_file.iter_batches(batch_size=10_000)
    )

    stream = split_linestrings(
        geometry_batches, nrows=grid.height, ncols=grid.width, transform=grid.transform
    )

    reader = pyarrow.RecordBatchReader._import_from_c_capsule(stream.__arrow_c_stream__())
    for batch in reader:
        # batch has a "geometry" column of the pieces (GeoArrow-encoded)
        # and a "parent" column: the index, in the source, of the
        # geometry each piece was split from
        ...

``split_linestrings`` also takes a ``bounded`` argument, working the same
way as the ``bounded`` parameter of
:func:`snail.intersection.split_linestrings`: pass ``bounded=True`` to
leave geometries (or parts of geometries) outside the grid whole, rather
than split at every gridline they would otherwise cross.

To get the pieces as shapely geometries instead - trading the streaming
memory benefit for convenience - read the whole stream at once with
:func:`snail.intersection.read_split_stream`, or convert a batch with
:meth:`geopandas.GeoDataFrame.from_arrow`.

Reference
---------

* :class:`snail.core.intersections.SplitStream` - what a split returns
* :func:`snail.core.intersections.split_linestrings`,
  :func:`~snail.core.intersections.split_polygons` - the extension
  functions described above
* :func:`snail.intersection.to_geoarrow`,
  :func:`snail.intersection.read_split_stream` - the conversions
  :func:`snail.intersection.split_linestrings` and
  :func:`~snail.intersection.split_polygons_experimental` use to work with
  an in-memory :class:`~geopandas.GeoDataFrame`
