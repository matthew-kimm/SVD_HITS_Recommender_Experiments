import pandas as pd
import unittest
from Recommender.shared.splitter import train_test_split


class TestSplitter(unittest.TestCase):
    def setUp(self) -> None:
        self.partial_data = pd.DataFrame([[1], [1], [2], [3], [4], [5]], columns=['user'])
        self.unique_users = set(self.partial_data['user'].unique())
        self.train, self.test = train_test_split(self.partial_data, train=0.8)

    def test_splitter_train_length(self):
        self.assertTrue(len(self.train) == 4)

    def test_splitter_test_length(self):
        self.assertTrue(len(self.test) == 1)

    def test_splitter_disjoint(self):
        self.assertTrue(self.train.union(self.test) == self.unique_users)
