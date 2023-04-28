import unittest

import numpy as np
from Recommender.models.pop import popular


class TestPopular(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([
            [0, 1, 2, 3],
            [0, 1, 3, 4],
            [0, 1, 4, 5],
            [1, 1, 3, 6],
            [1, 1, 4, 9],
            [2, 1, 4, 10]
        ])

        self.expected = np.array([
            [2, 1],
            [3, 2],
            [4, 3]
        ])

        self.expected_result = [self.expected]

    def test_popular(self):
        _, actual = popular(self.data)
        self.assertEqual(len(actual), len(self.expected_result))
        self.assertTrue(all([np.array_equal(actual[i], self.expected_result[i]) for i in range(len(actual))]))

