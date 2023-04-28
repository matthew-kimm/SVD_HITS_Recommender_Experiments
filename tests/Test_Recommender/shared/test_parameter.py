import unittest
import pandas as pd
import numpy as np
from Recommender.shared.parameter import descriptive_parameter_expansion_model, descriptive_parameter_expansion_models,\
    descriptive_parameter_columns


class TestParameter(unittest.TestCase):
    def setUp(self) -> None:
        self.avg = {'model': 'AVG', 'parameters': {'min_count': [10, 20]}}
        self.avg_result = [{'model': 'AVG', 'min_count': 10}, {'model': 'AVG', 'min_count': 20}]

        self.pop = {'model': 'POP', 'parameters': {}}
        self.pop_result = [{'model': 'POP'}]

        self.hits_pf = {'model': 'HITSW-PF', 'parameters': {'variation': ['+', '+-'], 'power': [1], 'xi': [0.1, 0.2], 'req_rating': [0.0, 4.0]}}
        self.hits_pf_result = [{'model': 'HITSW-PF', 'req_rating': 0.0, 'power': 1, 'xi': 0.1, 'variation': '+'},
                               {'model': 'HITSW-PF', 'req_rating': 0.0, 'power': 1, 'xi': 0.1, 'variation': '+-'},
                               {'model': 'HITSW-PF', 'req_rating': 0.0, 'power': 1, 'xi': 0.2, 'variation': '+'},
                               {'model': 'HITSW-PF', 'req_rating': 0.0, 'power': 1, 'xi': 0.2, 'variation': '+-'},
                               {'model': 'HITSW-PF', 'req_rating': 4.0, 'power': 1, 'xi': 0.1, 'variation': '+'},
                               {'model': 'HITSW-PF', 'req_rating': 4.0, 'power': 1, 'xi': 0.1, 'variation': '+-'},
                               {'model': 'HITSW-PF', 'req_rating': 4.0, 'power': 1, 'xi': 0.2, 'variation': '+'},
                               {'model': 'HITSW-PF', 'req_rating': 4.0, 'power': 1, 'xi': 0.2, 'variation': '+-'}]

        self.svd_gb = {'model': 'SVD-GB', 'parameters': {'variation': ['+', '+-'], 'd': [1, 2]}}
        self.svd_gb_result = [{'model': 'SVD-GB', 'variation': '+', 'd': 1},
                              {'model': 'SVD-GB', 'variation': '+', 'd': 2},
                              {'model': 'SVD-GB', 'variation': '+-', 'd': 1},
                              {'model': 'SVD-GB', 'variation': '+-', 'd': 2}]

        self.svd_pf = {'model': 'SVD-PF', 'parameters': {'variation': ['+', '+-'], 'd': [1, 2], 'req_rating': [0.0, 4.0]}}
        self.svd_pf_result = [{'model': 'SVD-PF', 'req_rating': 0.0, 'variation': '+', 'd': 1},
                              {'model': 'SVD-PF', 'req_rating': 0.0, 'variation': '+', 'd': 2},
                              {'model': 'SVD-PF', 'req_rating': 0.0, 'variation': '+-', 'd': 1},
                              {'model': 'SVD-PF', 'req_rating': 0.0, 'variation': '+-', 'd': 2},
                              {'model': 'SVD-PF', 'req_rating': 4.0, 'variation': '+', 'd': 1},
                              {'model': 'SVD-PF', 'req_rating': 4.0, 'variation': '+', 'd': 2},
                              {'model': 'SVD-PF', 'req_rating': 4.0, 'variation': '+-', 'd': 1},
                              {'model': 'SVD-PF', 'req_rating': 4.0, 'variation': '+-', 'd': 2}]

    def test_pop_model(self):
        actual = descriptive_parameter_expansion_model(self.pop)
        self.assertEqual(actual, self.pop_result)

    def test_avg_model(self):
        actual = descriptive_parameter_expansion_model(self.avg)
        self.assertEqual(actual, self.avg_result)

    def test_hits_pf_model(self):
        actual = descriptive_parameter_expansion_model(self.hits_pf)
        self.assertEqual(actual, self.hits_pf_result)

    def test_svd_gb_model(self):
        actual = descriptive_parameter_expansion_model(self.svd_gb)
        self.assertEqual(actual, self.svd_gb_result)

    def test_svd_pf_model(self):
        actual = descriptive_parameter_expansion_model(self.svd_pf)
        self.assertEqual(actual, self.svd_pf_result)

    def test_parameter_expansion_models(self):
        counts, actual = descriptive_parameter_expansion_models([self.pop, self.avg, self.svd_gb])
        self.assertEqual(self.pop_result + self.avg_result + self.svd_gb_result, actual)
        self.assertEqual(counts, [1, 2, 4])

    def test_parameter_columns(self):
        actual = descriptive_parameter_columns(self.avg_result + self.svd_gb_result)
        expected = pd.DataFrame([
              ['AVG', 10, np.nan, np.nan],
              ['AVG', 20, np.nan, np.nan],
              ['SVD-GB', np.nan, '+', 1],
              ['SVD-GB', np.nan, '+', 2],
              ['SVD-GB', np.nan, '+-', 1],
              ['SVD-GB', np.nan, '+-', 2]
            ], columns=['model', 'min_count', 'variation', 'd']
        )
        self.assertTrue(expected.equals(actual))

