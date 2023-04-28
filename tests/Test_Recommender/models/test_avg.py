import numpy as np
from Recommender.models.avg import avg
import unittest


class TestAvg(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([
            [0, 1, 2, 1.3],
            [0, 1, 3, 1.7],
            [0, 1, 4, 2.0],
            [1, 1, 3, 2.3],
            [1, 1, 4, 3.3],
            [2, 1, 4, 3.7]
        ])

        self.expected_avg_2 = np.array([
            [3, 2.0],
            [4, 3.0]
        ])

        self.expected_avg_3 = np.array([
            [4, 3.0]
        ])

        self.expected_result_2 = [self.expected_avg_2]
        self.expected_result_2_3 = [self.expected_avg_2, self.expected_avg_3]

    def test_avg_2(self):
        _, actual = avg(self.data, [2])
        self.assertEqual(len(actual), len(self.expected_result_2))
        self.assertTrue(all([np.array_equal(actual[i], self.expected_result_2[i]) for i in range(len(actual))]))

    def test_avg_2_3(self):
        _, actual = avg(self.data, [2, 3])
        self.assertEqual(len(actual), len(self.expected_result_2_3))
        self.assertTrue(all([np.array_equal(actual[i], self.expected_result_2_3[i]) for i in range(len(actual))]))


