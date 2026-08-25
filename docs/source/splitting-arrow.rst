Splitting large datasets
=========================

:func:`snail.intersection.split_linestrings`,
:func:`~snail.intersection.split_polygons` and
:func:`~snail.intersection.split_geometries` (and the ``snail split``/``snail
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

:func:`snail.core.intersections.split_linestrings`,
:func:`~snail.core.intersections.split_polygons` and
:func:`~snail.core.intersections.split_geometries` accept the geometries from
*any* object implementing the Arrow PyCapsule stream interface
(``__arrow_c_stream__``) or the single-array interface
(``__arrow_c_array__``, read as a stream of one batch) - a
:class:`pyarrow.ChunkedArray`, :class:`~pyarrow.Table` or
:class:`~pyarrow.RecordBatchReader`, a GeoParquet or
:class:`pyarrow.dataset.Dataset` reader, or
:meth:`GeoSeries.to_arrow() <geopandas.GeoSeries.to_arrow>`.

How the geometries are encoded
-------------------------------

Both of GeoArrow's encodings are read, and which one a source uses changes
what the split costs rather than whether it works.

**Native** - ``geoarrow.linestring``, ``geoarrow.polygon`` - keeps the
coordinates in Arrow buffers, and the extension reads them where they lie.
They may be interleaved (as geopandas exports them) or separated into x and
y arrays (as GeoParquet stores them), in two or more dimensions; splitting
is planar, so a z or m ordinate is stepped over rather than refused. No
per-feature Python object is built on either side of the interface.

**Well-known binary** - ``geoarrow.wkb`` - holds each geometry as a
serialised blob, which has to be decoded on the way in. This is what
:meth:`GeoDataFrame.to_parquet() <geopandas.GeoDataFrame.to_parquet>`
writes unless told otherwise, so it is what a GeoParquet file usually
contains.

A typed split still expects one geometry type throughout. A native column
declares its type in the schema, so a mismatch is caught before any data is
read; a WKB column does not - the type is inside each blob - so a feature
of the wrong type is reported as it is reached, naming the row::

    ValueError: Cannot split a Point (row 417) as a geoarrow.linestring

Geometries of more than one type
---------------------------------

:func:`snail.core.intersections.split_geometries` does not require a single
type. Each geometry is handled on its own terms: LineStrings and Polygons
are split, Points pass through unchanged, multi-part geometries are split
part by part, and a GeometryCollection is split member by member, however
deep it nests. Every piece is attributed to the row it came from, and an
empty geometry comes back as itself rather than dropping its row.

Its pieces come back as ``geoarrow.wkb``, because an Arrow stream has one
schema for all of its batches and WKB is the only encoding that can carry
every type in one. That is also why a mixed layer cannot be written as
native GeoArrow at all - geopandas raises ``ValueError: Geometry type
combination is not supported``.

What the encodings cost, splitting 20,000 linestrings into 300,000 pieces:

.. list-table::
   :header-rows: 1

   * - In
     - Out
     - Time
     - Size
   * - ``geoarrow.linestring``
     - ``geoarrow.linestring``
     - 19 ms
     - 13.5 MB
   * - ``geoarrow.wkb``
     - ``geoarrow.linestring``
     - 26 ms
     - 13.5 MB
   * - ``geoarrow.wkb``
     - ``geoarrow.wkb``
     - 24 ms
     - 16.2 MB

So most of the difference is decoding WKB on the way in rather than writing
it on the way out, and the blobs run about a fifth larger than the buffers.
Use the typed functions where a column really does hold one type, and
``split_geometries`` where it does not.

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

The file above needs no special treatment on the way out: written with
:meth:`GeoDataFrame.to_parquet() <geopandas.GeoDataFrame.to_parquet>`
defaults it is WKB-encoded, which is read as it stands.

Swapping ``split_linestrings`` for
:func:`~snail.core.intersections.split_geometries` is all it takes to split
a file whose geometries are not all one type; the ``"geometry"`` column of
each batch is then ``geoarrow.wkb`` rather than native GeoArrow.

``split_linestrings`` and ``split_geometries`` both take a ``bounded``
argument, working the same way as the ``bounded`` parameter of
:func:`snail.intersection.split_linestrings`: pass ``bounded=True`` to
leave geometries (or parts of geometries) outside the grid whole, rather
than split at every gridline they would otherwise cross.

To get the pieces as shapely geometries instead - trading the streaming
memory benefit for convenience - read the whole stream at once with
:func:`snail.intersection.read_split_stream`, or convert a batch with
:meth:`geopandas.GeoDataFrame.from_arrow`. Both work whichever encoding the
pieces came back in.

Reference
---------

* :class:`snail.core.intersections.SplitStream` - what a split returns
* :func:`snail.core.intersections.split_linestrings`,
  :func:`~snail.core.intersections.split_polygons` - the typed extension
  functions described above, giving back native GeoArrow
* :func:`snail.core.intersections.split_geometries` - the same for
  geometries of any type, giving back ``geoarrow.wkb``
* :func:`snail.intersection.to_geoarrow`,
  :func:`snail.intersection.read_split_stream` - the conversions
  :func:`snail.intersection.split_linestrings`,
  :func:`~snail.intersection.split_polygons_experimental` and
  :func:`~snail.intersection.split_geometries` use to work with an
  in-memory :class:`~geopandas.GeoDataFrame`
