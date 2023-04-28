import unittest
import pandas as pd
import numpy as np
from Recommender.shared.metric import get_metric_base, recommended_with_metric_base, compute_metrics


class TestMetric(unittest.TestCase):
    def setUp(self) -> None:
        self.metric_base = pd.DataFrame(
            [[{1, 2, 3}, {1, 2, 3}, set(), {1, 2}, {3}],
             [{4, 6}, {4}, {6}, set(), {4, 6}],
             [{12}, set(), {12}, {12}, set()]],
            columns=['rated', 'passed', 'failed', 'good', 'bad']
        )
        self.recommended = np.array([{1, 5, 7, 8, 10}, {4, 6, 12, 13, 14}, set()])
        self.data = pd.DataFrame(
            [[1, 1, 1, 4.0, 3.5],
             [1, 1, 2, 4.0, 3.5],
             [1, 1, 3, 3.0, 3.5],
             [2, 0, 12, 3.0, 2.5],
             [2, 1, 4, 2.0, 2.5],
             [2, 1, 6, 1.0, 2.5],
             [3, 1, 12, 1.0, 0.0],
             [4, 1, 7, 4.0, 1.75]],
            columns=['user', 'time', 'item', 'rating', 'user_avg_rating']
        )
        self.test_set = {1, 2, 3}
        self.pass_rating = 2.0
        self.target_time = 1

        self.base_intersect_recommended = pd.DataFrame([
            [{1}, {1}, set(), {1}, set(), {1, 5, 7, 8, 10}],
            [{4, 6}, {4}, {6}, set(), {4, 6}, {4, 6, 12, 13, 14}],
            [set(), set(), set(), set(), set(), set()]
        ], columns=['recommended_rated', 'recommended_passed', 'recommended_failed',
                    'recommended_good', 'recommended_bad', 'recommended'])

        self.base_with_recommended = pd.concat([self.metric_base, self.base_intersect_recommended], axis=1)

        self.computed_metrics = pd.DataFrame([
            [3, 3, 0, 2, 1, 1, 1, 0, 1, 0, 5, 1/3, np.nan, 1/2, 0, 1/5, 0, 1/5, 0],
            [2, 1, 1, 0, 2, 2, 1, 1, 0, 2, 5, 1, 1, np.nan, 1, 1/5, 1/5, 0/5, 2/5],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, np.nan, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan]
        ], columns=[
            'count_rated',
            'count_passed',
            'count_failed',
            'count_good',
            'count_bad',
            'count_recommended_rated',
            'count_recommended_passed',
            'count_recommended_failed',
            'count_recommended_good',
            'count_recommended_bad',
            'count_recommended',
            'recall_passed',
            'recall_failed',
            'recall_good',
            'recall_bad',
            'precision_passed',
            'precision_failed',
            'precision_good',
            'precision_bad'
        ])

    def test_get_metric_base(self):
        actual = get_metric_base(self.data, self.test_set, self.pass_rating, self.target_time)
        self.assertTrue(self.metric_base.equals(actual))

    def test_metric_base_with_recommended(self):
        actual = recommended_with_metric_base(self.metric_base, self.recommended)
        self.assertTrue(self.base_with_recommended.equals(actual))

    def test_metric_base_with_recommended_2_models(self):
        recommended_2_models = self.recommended.repeat(2)
        base_with_rec_2_models = pd.DataFrame(np.repeat(np.array(self.base_with_recommended), 2, axis=0), columns=self.base_with_recommended.columns)
        actual = recommended_with_metric_base(self.metric_base, recommended_2_models)
        self.assertTrue(base_with_rec_2_models.equals(actual))

    def test_compute_metric(self):
        actual = compute_metrics(self.base_with_recommended)
        self.assertTrue(self.computed_metrics.equals(actual))
