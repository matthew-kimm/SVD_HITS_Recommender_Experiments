import unittest
import pandas as pd
from Recommender.shared.neighbor import get_neighbor_table, filter_data_with_users


class TestNeighbors(unittest.TestCase):
    def setUp(self) -> None:
        self.columns = ['user', 'time', 'item', 'rating']
        self.attributes = ['attribute_1', 'attribute_2']
        self.columns += self.attributes
        data = [[i, 0, 5, 2.0, 'abc', 'def'] for i in range(11)]
        self.data = pd.DataFrame(data, columns=self.columns)
        self.test = set(range(0, 2))
        self.train = set(range(2, 11))
        self.target_time = 1
        extra_data = [[i, 0, 7, 3.0, 'abc', 'def'] for i in [0, 7, 9]]
        self.extra_data = pd.DataFrame(extra_data, columns=self.columns)

    def test_overlap_1(self):
        neighbors = get_neighbor_table(self.data, self.train, self.test,
                                       attributes=self.attributes, overlap=1,
                                       target_time=self.target_time,
                                       round_rating=False)
        expected = {k: set(range(2, 11)) for k in range(0, 2)}
        self.assertEqual(expected, neighbors)

    def test_overlap_1_missing_attr(self):
        data = self.data.copy()
        data.iloc[3, :] = [3, 0, 5, 2.0, 'no', 'def']
        neighbors = get_neighbor_table(data, self.train, self.test,
                                       attributes=self.attributes, overlap=1,
                                       target_time=self.target_time,
                                       round_rating=False)
        expected = {k: set(range(2, 11)).difference({3}) for k in range(0, 2)}
        self.assertEqual(expected, neighbors)

    def test_overlap_1_missing_grade(self):
        data = self.data.copy()
        data.iloc[3, :] = [3, 0, 5, 2.3, 'abc', 'def']
        neighbors = get_neighbor_table(data, self.train, self.test,
                                       attributes=self.attributes, overlap=1,
                                       target_time=self.target_time,
                                       round_rating=False)
        expected = {k: set(range(2, 11)).difference({3}) for k in range(0, 2)}
        self.assertEqual(expected, neighbors)

    def test_overlap_1_round_grade(self):
        data = self.data.copy()
        data.iloc[3, :] = [3, 0, 5, 2.3, 'abc', 'def']
        neighbors = get_neighbor_table(data, self.train, self.test,
                                       attributes=self.attributes, overlap=1,
                                       target_time=self.target_time,
                                       round_rating=True)
        expected = {k: set(range(2, 11)) for k in range(0, 2)}
        self.assertEqual(expected, neighbors)

    def test_overlap_2(self):
        data = self.data.copy()
        data = pd.concat([data, self.extra_data])
        neighbors = get_neighbor_table(data, self.train, self.test,
                                       attributes=self.attributes, overlap=2,
                                       target_time=self.target_time,
                                       round_rating=False)
        expected = {0: {7, 9}} | {1: set()}
        self.assertEqual(expected, neighbors)

    def test_filter_with_users(self):
        data = self.data.copy()
        users_to_filter = {3, 5}
        expected = data.iloc[[3, 5]].reset_index(drop=True)
        actual = filter_data_with_users(data, users_to_filter)
        self.assertTrue(expected.equals(actual))
