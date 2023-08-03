import pytest
import snail.core.intersections

from shapely.geometry import LineString, Polygon


nrows = 2
ncols = 2
transform = [1, 0, 0, 0, 1, 0]


@pytest.mark.parametrize(
    "test_linestring,expected",
    [
        (
            LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
            [
                LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.5)]),
                LineString([(1.0, 0.5), (1.5, 0.5), (1.5, 1.0)]),
                LineString([(1.5, 1.0), (1.5, 1.5)]),
            ],
        ),
        (
            LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
            [
                LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.8333333)]),
                LineString([(1.0, 0.8333333), (1.125, 1.0)]),
                LineString([(1.125, 1.0), (1.5, 1.5)]),
            ],
        ),
    ],
)
def test_linestring_splitting(test_linestring, expected):
    splits = snail.core.intersections.split_linestring(
        test_linestring, nrows, ncols, transform
    )
    for split, expected_split in zip(splits, expected):
        assert split.equals_exact(expected_split, 1e-7)


@pytest.mark.parametrize(
    "test_linestring,expected",
    [
        (
            LineString([(0.25, 1.25), (0.5, 1.5), (0.5, 1.75)]),
            (0, 1),
        ),
        (
            LineString([(1.25, 1.25), (1.5, 1.5), (1.5, 1.75)]),
            (1, 1),
        ),
    ],
)
def test_get_cell_indices(test_linestring, expected):
    cell_indices = snail.core.intersections.get_cell_indices(
        test_linestring, nrows, ncols, transform
    )
    assert cell_indices == expected


@pytest.mark.xfail
def test_split_polygons():
    bad_poly = Polygon(
        (
            [-0.0062485600499826, 51.61041647955],
            [-0.0062485600499826, 51.602083146149994],
            [0.0020847733500204, 51.602083146149994],
            # [0.0020847733500204, 51.61041647955],
            # [-0.0062485600499826, 51.61041647955],
        )
    )

    # expect a RuntimeError: Expected even number of crossings on gridline.
    snail.core.intersections.split_polygon(
        bad_poly,
        36082,
        18000,
        (1000.0, 0.0, -18041000.0, 0.0, -1000.0, 9000000.0),
    )
