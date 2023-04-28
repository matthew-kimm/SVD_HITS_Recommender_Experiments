import unittest
import numpy as np
import itertools
from Recommender.data.data import cartesian_product, item_item_data,\
    fast_pivot_value,\
    fast_agg_sum, co_occurrence_matrix,\
    filter_item_item_data_items, filter_item_item_data_users


class TestItemItemData(unittest.TestCase):
    def setUp(self) -> None:
        self.x = np.arange(0, 6).reshape((6, 1))
        self.y = np.arange(0, 8).reshape((8, 1))

        # user, item, time, rating (int), user avg rating (int)
        # ordinal data like grades
        self.data = np.array([
            [0, 0, 1, 5, 3],
            [0, 0, 2, 3, 3],
            [0, 1, 11, 4, 3],
            [1, 0, 6, 3, 2],
            [1, 1, 16, 2, 2],
            [1, 1, 17, 4, 2]
        ])

        self.expected_item_item_data = np.array([
            [0, 1, 5, 11, 4, 3],
            [0, 2, 3, 11, 4, 3],
            [1, 6, 3, 16, 2, 2],
            [1, 6, 3, 17, 4, 2]
        ])

        self.allowed_users = np.array([1])
        self.allowed_items = np.array(list({16, 17}))
        self.allowed_items_set = {16, 17}
        self.expected_filtered_item_item_data = np.array([
            [1, 6, 3, 16, 2, 2],
            [1, 6, 3, 17, 4, 2]
        ])

        self.cartesian_product = list(map(list, list(sorted(itertools.product(self.x.ravel(), self.y.ravel())))))

    def test_cartesian_product(self):
        actual = cartesian_product(self.x, self.y)
        self.assertEqual(sorted(actual.tolist()), self.cartesian_product)

    def test_item_item_data(self):
        actual = item_item_data(self.data)
        self.assertTrue(np.array_equal(actual, self.expected_item_item_data))

    def test_item_item_data_filter_item(self):
        actual = filter_item_item_data_items(self.expected_item_item_data, self.allowed_items)
        self.assertTrue(np.array_equal(actual, self.expected_filtered_item_item_data))

    def test_item_item_data_filter_users(self):
        actual = filter_item_item_data_users(self.expected_item_item_data, self.allowed_users)
        self.assertTrue(np.array_equal(actual, self.expected_filtered_item_item_data))



class TestPivot(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([
            [4, 2, 2.0],
            [5, 1, 1.0],
            [5, 3, 3.0],
            [6, 0, 0.5]
        ])

        self.expected = np.array([
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.5, 0.0, 0.0, 0.0]
        ])

        self.expected_key1 = np.array([4, 5, 6])
        self.expected_key2 = np.array([0, 1, 2, 3])

        self.key1, self.key2, self.actual = fast_pivot_value(self.data)

    def test_pivot_matrix(self):
        self.assertTrue(np.array_equal(self.actual, self.expected))

    def test_pivot_key1(self):
        self.assertTrue(np.array_equal(self.key1, self.expected_key1))

    def test_pivot_key2(self):
        self.assertTrue(np.array_equal(self.key2, self.expected_key2))


class CoOccurrenceMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.data = np.array([
            [4, 1, 2.5],
            [4, 1, 1.5],
            [4, 1, 1.0],
            [4, 3, 1.0],
            [7, 2, 1.0],
            [7, 4, 1.0]
        ])

        self.expected = np.array([
            [5.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0, 1.0]
        ])
        self.e_idx = np.array([4, 7])
        self.e_col = np.array([1, 2, 3, 4])

        self.e_unique_keys = np.array([
            [4, 1],
            [4, 3],
            [7, 2],
            [7, 4]
        ])

        self.e_key_sums = np.array([[5.0], [1.0], [1.0], [1.0]])

        self.a_unique_keys, self.a_key_sums = fast_agg_sum(self.data)
        self.a_idx, self.a_col, self.a_matrix = co_occurrence_matrix(self.data)

    def test_fast_agg_sum_unique_keys(self):
        self.assertTrue(np.array_equal(self.a_unique_keys, self.e_unique_keys))

    def test_fast_agg_sum_key_sums(self):
        self.assertTrue(np.array_equal(self.a_key_sums, self.e_key_sums))

    def test_co_occurrence_idx(self):
        self.assertTrue(np.array_equal(self.a_idx, self.e_idx))

    def test_co_occurrence_col(self):
        self.assertTrue(np.array_equal(self.a_col, self.e_col))

    def test_co_occurrence_matrix(self):
        self.assertTrue(np.array_equal(self.expected, self.a_matrix))
