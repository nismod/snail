import pytest
import snail.core.intersections
from shapely import wkt
from shapely.geometry import LineString, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

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
    actual = snail.core.intersections.split_linestring(
        test_linestring, nrows, ncols, transform
    )
    assert len(actual) == len(expected), "Expected the same number of splits"
    for split, expected_split in zip(actual, expected):
        assert split.equals_exact(expected_split, 1e-7)


def test_linestring_splitting_issue_61():
    ncols = 120163
    nrows = 259542
    transform = (5.0, 0.0, 54675.0, 0.0, -5.0, 1217320.0)

    # reduced test case to single straight-line segment
    test_linestring = LineString([(415805.57, 430046.95), (415800.0, 430015.0)])
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
    assert len(actual) == len(expected), "Expected the same number of splits"
    for split, expected_split in zip(actual, expected):
        if not split.equals_exact(expected_split, 1e-5):
            assert False, (
                f"Expected split coordinates to match, got {split}, {expected_split}"
            )


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


def assert_split_pieces_cover_polygon(polygon, splits):
    """The valid, non-overlapping splits should exactly cover the polygon."""
    assert len(splits) > 0
    for split in splits:
        assert isinstance(split, Polygon)
        assert split.is_valid

    merged = unary_union(splits)
    total = sum(split.area for split in splits)
    assert total == pytest.approx(polygon.area, rel=1e-6)
    assert total == pytest.approx(merged.area, rel=1e-6)
    assert merged.symmetric_difference(polygon).area == pytest.approx(
        0.0, abs=max(1e-12, polygon.area * 1e-6)
    )


def assert_splits_match_shapely_grid_intersections(
    polygon, splits, nrows, ncols, transform
):
    """Compare C++ split pieces with Shapely intersections for every cell."""
    a, _, c, _, e, f = transform
    expected = []
    for i in range(ncols):
        for j in range(nrows):
            x0 = i * a + c
            x1 = (i + 1) * a + c
            y0 = j * e + f
            y1 = (j + 1) * e + f
            cell = shapely_box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            clipped = polygon.intersection(cell)
            if clipped.geom_type == "Polygon" and not clipped.is_empty:
                expected.append(clipped)
            elif clipped.geom_type == "MultiPolygon":
                expected.extend(part for part in clipped.geoms if not part.is_empty)

    unmatched = list(splits)
    for expected_piece in expected:
        match = next(
            (
                index
                for index, actual_piece in enumerate(unmatched)
                if actual_piece.equals_exact(
                    expected_piece, tolerance=1e-12, normalize=True
                )
            ),
            None,
        )
        assert match is not None, f"No C++ split matched {expected_piece.wkt}"
        unmatched.pop(match)

    assert not unmatched


@pytest.mark.parametrize(
    "polygon",
    [
        Polygon(
            [
                (0.1, 0.1),
                (3.6, 0.1),
                (3.6, 3.8),
                (2.6, 3.8),
                (2.6, 1.6),
                (1.2, 1.6),
                (1.2, 3.8),
                (0.1, 3.8),
            ]
        ),
        Polygon(
            [(0.1, 0.1), (3.6, 0.1), (3.6, 3.8), (0.1, 3.8)],
            [[(0.9, 1.1), (2.8, 1.1), (2.8, 2.2), (0.9, 2.2)]],
        ),
    ],
    ids=["concave", "hole"],
)
@pytest.mark.parametrize(
    "transform",
    [
        (0.75, 0.0, -0.5, 0.0, 1.25, -0.5),
        (0.75, 0.0, -0.5, 0.0, -1.25, 4.5),
    ],
    ids=["positive-y", "negative-y"],
)
def test_split_polygon_matches_shapely_grid_intersections(polygon, transform):
    nrows = 4
    ncols = 6
    splits = snail.core.intersections.split_polygon(polygon, nrows, ncols, transform)

    assert_split_pieces_cover_polygon(polygon, splits)
    assert_splits_match_shapely_grid_intersections(
        polygon, splits, nrows, ncols, transform
    )
    """Each split should be a valid Polygon, and the split pieces should sum
    to the area of the polygon they came from"""
    for split in splits:
        assert isinstance(split, Polygon)
        assert split.is_valid
    total = sum(split.area for split in splits)
    assert total == pytest.approx(polygon.area, rel=1e-6)


def test_split_polygons_issue_53():
    # reduced test case
    polygon = Polygon(
        (
            [-0.1, 2],
            [-0.1, 1],
            [0.1, 1],
        )
    )

    splits = snail.core.intersections.split_polygon(
        polygon,
        2,
        2,
        (10.0, 0.0, -100.0, 0.0, -10.0, 100.0),
    )
    assert_split_pieces_cover_polygon(polygon, splits)

    bad_poly = Polygon(
        (
            [-0.0062485600499826, 51.61041647955],
            [-0.0062485600499826, 51.602083146149994],
            [0.0020847733500204, 51.602083146149994],
            [0.0020847733500204, 51.61041647955],
            [-0.0062485600499826, 51.61041647955],
        )
    )

    splits = snail.core.intersections.split_polygon(
        bad_poly,
        36082,
        18000,
        (1000.0, 0.0, -18041000.0, 0.0, -1000.0, 9000000.0),
    )
    assert_split_pieces_cover_polygon(bad_poly, splits)


@pytest.mark.parametrize(
    "building_wkt",
    [
        "POLYGON ((-77.280182457 17.97282044099908, -77.28015690000001 17.97282209999907, -77.28015000000001 17.9727272999991, -77.28013888 17.97272804099908, -77.28013888 17.97270079499907, -77.28014880000001 17.97270009999908, -77.2801487 17.9726981999991, -77.28018086100001 17.97269609699906, -77.280182457 17.97282044099908))",
        "POLYGON ((-78.02110589999999 18.45206629999907, -78.02105 18.45206859999908, -78.0210484 18.45203429999912, -78.0211044 18.45203199999911, -78.02110589999999 18.45206629999907))",
        "POLYGON ((-78.13875147900001 18.23190661899908, -78.13874999799999 18.23190337199908, -78.13874999799999 18.2318642849991, -78.1387642 18.23185839999909, -78.1387629 18.23183549999909, -78.1388044 18.23181769999908, -78.13882150000001 18.2318385999991, -78.13885000000001 18.23181769999908, -78.138889604 18.23190435499911, -78.13875147900001 18.23190661899908))",
        "POLYGON ((-76.2603323 17.91376489999907, -76.26028700000001 17.91380679999907, -76.260265 17.91378519999909, -76.26025 17.91379899999908, -76.260200149 17.91375000099906, -76.260317138 17.91375000099909, -76.2603323 17.91376489999907))",
        "POLYGON ((-77.34835 18.45626119999909, -77.348350719 18.45625000599908, -77.34840013100001 18.45625000599913, -77.3483981 18.4562824999991, -77.3483772 18.45628129999909, -77.3483755 18.45630829999907, -77.34833759999999 18.45630619999902, -77.34834050000002 18.45626069999907, -77.34835 18.45626119999909))",
        "POLYGON ((-77.2026698 18.43244629999913, -77.20265000000001 18.43244829999909, -77.2026425 18.43237909999909, -77.20270131100001 18.43237336299909, -77.20270311900001 18.43251529999906, -77.20267749999999 18.43251779999907, -77.2026698 18.43244629999913))",
        "POLYGON ((-78.13575 18.21203339999909, -78.135694442 18.21203276499908, -78.135694442 18.21189381799909, -78.1357517 18.2118943999991, -78.13575 18.21203339999909))",
        "POLYGON ((-77.252083324 18.46059157499909, -77.25213100000001 18.46047719999909, -77.2521508 18.46048469999907, -77.25217360000001 18.4604300999991, -77.25225 18.46042879999908, -77.2522822 18.46049999999909, -77.2522677 18.46053479999907, -77.25231309999999 18.46055189999906, -77.25233420000001 18.46050139999911, -77.25236110199999 18.46051149999909, -77.25236110199999 18.46062931199909, -77.252318524 18.46062981799905, -77.2522195 18.46059269999907, -77.2522115 18.46061189999907, -77.2521461 18.46058739999908, -77.25212748 18.46063208799907, -77.252083324 18.46063261299908, -77.252083324 18.46059157499909))",
        "POLYGON ((-77.36705790000001 18.25292739999907, -77.3670561 18.25300909999904, -77.3669494 18.25300699999909, -77.36695000000002 18.25298059999908, -77.36690040000001 18.25297959999905, -77.36690160000001 18.25292429999907, -77.36705790000001 18.25292739999907))",
        "POLYGON ((-76.80582605399999 17.99791666899907, -76.80585000000001 17.99786519999907, -76.8059296 17.99789869999908, -76.80592125100002 17.99791666899907, -76.80582605399999 17.99791666899907))",
        "POLYGON ((-77.92486289999999 18.44194119999911, -77.92496370000001 18.44195809999906, -77.92495 18.44203159999907, -77.924861107 18.44201669599908, -77.924861107 18.4419508179991, -77.92486289999999 18.44194119999911))",
    ],
)
def test_split_polygons_issue_45(building_wkt):
    # Building polygons with vertices within ~1e-14 of a gridline, which used
    # to fail with "Expected even number of crossings on gridline"
    polygon = wkt.loads(building_wkt)
    splits = snail.core.intersections.split_polygon(
        polygon,
        698,
        252,
        (
            0.0030999999999999925,
            0.0,
            -78.34655,
            0.0,
            -0.0030999999999999934,
            18.52365,
        ),
    )
    assert_split_pieces_cover_polygon(polygon, splits)


def test_split_polygon_with_hole():
    polygon = Polygon(
        [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)],
        [[(1.25, 1.25), (1.75, 1.25), (1.75, 1.75), (1.25, 1.75)]],
    )
    splits = snail.core.intersections.split_polygon(
        polygon, 3, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    assert_split_pieces_cover_polygon(polygon, splits)
    assert len(splits) == 9
    # the middle cell is fully covered, except for the hole
    (with_hole,) = [s for s in splits if s.interiors]
    assert with_hole.area == pytest.approx(0.75)


def test_split_polygon_with_hole_crossing_gridlines():
    polygon = Polygon(
        [(0.5, 0.5), (2.5, 0.5), (2.5, 2.5), (0.5, 2.5)],
        [[(0.9, 0.9), (2.1, 0.9), (2.1, 1.1), (0.9, 1.1)]],
    )
    splits = snail.core.intersections.split_polygon(
        polygon, 3, 3, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    assert_split_pieces_cover_polygon(polygon, splits)
    assert len(splits) == 9
