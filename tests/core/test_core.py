import pytest
import shapely.wkt
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


def test_linestring_splitting_issue_61():
    ncols = 120163
    nrows = 259542
    transform = (5.0, 0.0, 54675.0, 0.0, -5.0, 1217320.0)

    # reduced test case to single straight-line segment
    test_linestring = LineString(
        [(415805.57, 430046.95), (415800.0, 430015.0)]
    )
    expected = [
        LineString([(415805.57, 430046.95), (415805.23004, 430045.0)]),
        LineString([(415805.23004, 430045.0), (415805, 430043.68043)]),
        LineString([(415805.0, 430043.68043), (415804.35837, 430040.0)]),
        LineString([(415804.35837, 430040.0), (415803.48669, 430035.0)]),
        LineString([(415803.48669, 430035.0), (415802.61502, 430030.0)]),
        LineString([(415802.61502, 430030.0), (415801.74334, 430025.0)]),
        LineString([(415801.74334, 430025.0), (415800.87167, 430020.0)]),
        LineString([(415800.87167, 430020.0), (415800.0, 430015.0)]),
    ]
    actual = snail.core.intersections.split_linestring(
        test_linestring, nrows, ncols, transform
    )
    for split, expected_split in zip(actual, expected):
        if not split.equals_exact(expected_split, 1e-5):
            assert (
                False
            ), f"Expected split coordinates to match, got {split}, {expected_split}"


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


def test_split_polygons_issue_53():
    # reduced test case
    polygon = Polygon(
        (
            [-0.1, 2],
            [-0.1, 1],
            [0.1, 1],
        )
    )

    snail.core.intersections.split_polygon(
        polygon,
        2,
        2,
        (10.0, 0.0, -100.0, 0.0, -10.0, 100.0),
    )

    bad_poly = Polygon(
        (
            [-0.0062485600499826, 51.61041647955],
            [-0.0062485600499826, 51.602083146149994],
            [0.0020847733500204, 51.602083146149994],
            [0.0020847733500204, 51.61041647955],
            [-0.0062485600499826, 51.61041647955],
        )
    )

    snail.core.intersections.split_polygon(
        bad_poly,
        36082,
        18000,
        (1000.0, 0.0, -18041000.0, 0.0, -1000.0, 9000000.0),
    )
