import unittest
import pandas as pd
from Recommender.shared.item_filter import get_item_filter_allowed_table, filter_data_allowed_items


class TestItemFilter(unittest.TestCase):
    def setUp(self) -> None:
        self.neighbors = {0: set([1, 2, 3, 4])}
        self.data = pd.DataFrame(
            [[0, 0, 1, 2.0],
             [1, 0, 1, 2.0],
             [1, 1, 2, 1.0],
             [1, 1, 3, 2.0],
             [2, 0, 1, 2.0],
             [2, 1, 2, 0.0],
             [2, 1, 4, 4.0],
             [3, 0, 1, 2.0],
             [3, 1, 5, 1.0],
             [4, 0, 1, 2.0],
             [4, 1, 13, 3.0]], columns=['user', 'time', 'item', 'rating']
        )
        self.filtered_data = pd.DataFrame(
            [[1, 1, 3, 2.0],
             [2, 1, 4, 4.0],
             [3, 1, 5, 1.0]], columns=['user', 'time', 'item', 'rating'],
            index=[3, 6, 8]
        )
        self.filter_data = {0: {1, 2, 13}}
        self.allowed_items = {3, 4, 5}
        self.expected_filter_table = {0: self.allowed_items}
        self.target_time = 1

    def test_get_item_filter_table(self):
        actual = get_item_filter_allowed_table(self.data, self.filter_data, self.neighbors,
                                               self.target_time)
        self.assertEqual(self.expected_filter_table, actual)

    def test_item_filter_data(self):
        actual = filter_data_allowed_items(self.data, self.allowed_items)
        self.assertTrue(self.filtered_data.equals(actual))
