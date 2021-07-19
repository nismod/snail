import unittest

from shapely.geometry import LineString

from snail import intersections


class TestIntersections(unittest.TestCase):
    def test_linestring_splitting(self):
        test_linestrings = [
            LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 0.5), (1.5, 1.5)]),
            LineString([(0.5, 0.5), (0.75, 0.5), (1.5, 1.5)]),
        ]
        expected = [
            [
                LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.5)]),
                LineString([(1.0, 0.5), (1.5, 0.5), (1.5, 1.0)]),
                LineString([(1.5, 1.0), (1.5, 1.5)]),
            ],
            [
                LineString([(0.5, 0.5), (0.75, 0.5), (1.0, 0.8333)]),
                LineString([(1.0, 0.8333), (1.125, 1.0)]),
                LineString([(1.125, 1.0), (1.5, 1.5)]),
            ],
        ]

        nrows = 2
        ncols = 2
        transform = [1, 0, 0, 0, 1, 0]
        for i, test_data in enumerate(zip(test_linestrings, expected)):
            test_linestring, expected_splits = test_data
            with self.subTest(i=i):
                splits = intersections.split(
                    test_linestring, nrows, ncols, transform
                )
                self.assertTrue(
                    [
                        split.almost_equals(expected_split)
                        for split, expected_split in zip(
                            splits, expected_splits
                        )
                    ]
                )
