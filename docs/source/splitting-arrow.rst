Splitting large datasets
=========================

For most tasks, ``snail split`` and ``snail process`` commands should just work.
Or, using the Python functions :func:`snail.intersection.split_linestrings`,
:func:`~snail.intersection.split_polygons` and
:func:`~snail.intersection.split_geometries` you can pass in a
:class:`~geopandas.GeoDataFrame` of geometries and get a split one back.

This page goes into more of the implementation details, which might help if you
want to split a source that is not already loaded as a GeoDataFrame - for
example, a GeoParquet file bigger than available memory - or if you are working
directly against the compiled extension.

How a split runs
-----------------

The splitting logic is all implemented in a C++ extension, which is compiled and
made available to Python as :module:`snail.core.intersections`.

Geometries are passed into the C++ extension as `GeoArrow
<https://geoarrow.org/>`_, using the `Arrow C stream interface
<https://arrow.apache.org/docs/format/CStreamInterface.html>`_: the extension
pulls one batch of geometries at a time, splits it, and hands the pieces back
the same way, as a stream of record batches. Nothing is split until that stream
is read, and only one batch of the source is ever held at once - so it should be
possible for a file larger than memory to be split by a consumer that takes
the results as they come, without loading the whole thing first.

:func:`snail.core.intersections.split_linestrings`,
:func:`~snail.core.intersections.split_polygons` and
:func:`~snail.core.intersections.split_geometries` accept the geometries from
*any* object implementing the Arrow PyCapsule stream interface
(``__arrow_c_stream__``) or the single-array interface (``__arrow_c_array__``,
read as a stream of one batch) - a :class:`pyarrow.ChunkedArray`,
:class:`~pyarrow.Table` or :class:`~pyarrow.RecordBatchReader`, a GeoParquet or
:class:`pyarrow.dataset.Dataset` reader, or :meth:`GeoSeries.to_arrow()
<geopandas.GeoSeries.to_arrow>`.

Geometry encoding
-----------------

GeoArrow specifies two possible encodings, native and well-known binary. Either
can be read:

**Native** - ``geoarrow.linestring``, ``geoarrow.polygon`` - keeps the
coordinates in Arrow buffers. They may be interleaved (as geopandas exports
them) or separated into x and y arrays (as GeoParquet stores them), in two or
more dimensions; splitting works on x and y dimensions, so a z or m co-ordinates
are stepped over.

**Well-known binary** - ``geoarrow.wkb`` - holds each geometry as a
serialised blob, which has to be decoded on the way in. This is what
:meth:`GeoDataFrame.to_parquet() <geopandas.GeoDataFrame.to_parquet>`
writes unless told otherwise, so it is what a GeoParquet file usually
contains.

Geometries of more than one type
---------------------------------

:func:`snail.core.intersections.split_geometries` can handle a stream of mixed
geometry types:

- LineStrings and Polygons are split
- Points pass through
- multi-part geometries are split part by part
- GeometryCollections are split member by member

This returns ``geoarrow.wkb``, because an Arrow stream has one schema for all of
its batches and WKB is the only encoding that can carry every type in one. That
is also why a mixed layer cannot be written as native GeoArrow - geopandas
raises ``ValueError: Geometry type combination is not supported``.

There is some overhead to encoding and decoding `wkb`. Use the typed functions
where a column really does hold one type, and ``split_geometries`` where it does
not.

Splitting a file directly
--------------------------

This reads a GeoParquet file of linestrings in batches and splits each
batch as it arrives, never holding more than one batch of geometries and
one batch of pieces at a time - whatever the file's size::

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

Swap ``split_linestrings`` for
:func:`~snail.core.intersections.split_geometries` to split a file whose
geometries are not all one type; the ``"geometry"`` column of each batch is then
``geoarrow.wkb`` rather than native GeoArrow.

To get the pieces as shapely geometries instead, convert a batch with
:meth:`geopandas.GeoDataFrame.from_arrow`.

Or, to read the whole stream into memory, use
:func:`snail.intersection.read_split_stream`.
