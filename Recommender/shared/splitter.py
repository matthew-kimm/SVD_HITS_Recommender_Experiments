import pandas as pd
import numpy as np
from typing import Tuple


def train_test_split(data: pd.DataFrame, train: float = 0.9) -> Tuple[set, set]:
    users = list(data['user'].unique())
    train = round(len(users) * train)
    test = len(users) - train
    test_users = set(np.random.choice(users, test, replace=False))
    train_users = set(users).difference(test_users)
    return train_users, test_users
