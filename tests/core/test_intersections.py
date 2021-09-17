import unittest

from shapely.geometry import LineString

import snail.core.intersections


class TestIntersections(unittest.TestCase):
    def setUp(self):
        self.nrows = 2
        self.ncols = 2
        self.transform = [1, 0, 0, 0, 1, 0]

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

        for i, test_data in enumerate(zip(test_linestrings, expected)):
            test_linestring, expected_splits = test_data
            with self.subTest(i=i):
                splits = snail.core.intersections.split_linestring(
                    test_linestring, self.nrows, self.ncols, self.transform
                )
                self.assertTrue(
                    [
                        split.almost_equals(expected_split)
                        for split, expected_split in zip(
                            splits, expected_splits
                        )
                    ]
                )

    def test_get_cell_indices(self):
        test_linestrings = [
            LineString([(0.25, 1.25), (0.5, 1.5), (0.5, 1.75)]),
            LineString([(1.25, 1.25), (1.5, 1.5), (1.5, 1.75)]),
        ]
        expected_cell_indices = [(0, 1), (1, 1)]

        for i, test_linestring in enumerate(test_linestrings):
            with self.subTest(i=i):
                cell_indices = snail.core.intersections.get_cell_indices(
                    test_linestring, self.nrows, self.ncols, self.transform
                )
                self.assertEqual(cell_indices, expected_cell_indices[i])
