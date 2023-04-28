import pandas as pd
from itertools import combinations
from typing import Dict


def get_neighbor_table(data: pd.DataFrame, train: set, test: set,
                       attributes: list, overlap: int, target_time: int,
                       round_rating: bool, global_neighborhood_override: bool = False) -> Dict[int, set]:
    neighbors = {}
    if global_neighborhood_override:
        for user in test:
            user_neighbors = train
            user_neighbors.discard(user)
            neighbors[user] = user_neighbors
        return neighbors

    df = data.copy()
    df = df[df['time'] < target_time]

    if round_rating:
        df['rating'] = df['rating'].apply(round)

    groupby_columns = ['item', 'rating'] + attributes
    single_item_rating_matches = list(df.groupby(groupby_columns)['user'].agg(set).values)
    partitions = [set.intersection(*matches)
                  for matches in
                  combinations(single_item_rating_matches, overlap)]

    for user in test:
        user_in_partitions = list(filter(lambda x: user in x, partitions))
        if len(user_in_partitions):
            user_neighbors = set.union(*user_in_partitions).intersection(train)
            user_neighbors.discard(user)
        else:
            user_neighbors = set()
        neighbors[user] = user_neighbors

    return neighbors


def filter_data_with_users(data: pd.DataFrame, users: set) -> pd.DataFrame:
    df = data.copy()
    neighbors_idx = pd.DataFrame(users, columns=['user'])
    df = pd.merge(df, neighbors_idx, how='inner', on='user')
    return df
