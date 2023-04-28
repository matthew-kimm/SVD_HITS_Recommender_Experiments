import pandas as pd
from Recommender.shared.neighbor import filter_data_with_users
from typing import Dict


def get_item_filter_allowed_table(data: pd.DataFrame, item_filter_not_allowed: Dict[int, set],
                                  neighbors: Dict[int, set], target_time: int) -> Dict[int, set]:
    df = data.copy()
    df = df[df['time'] == target_time]

    item_filter_allowed = {}
    for user, neighbors in neighbors.items():
        tdf = filter_data_with_users(df, neighbors)
        possible_items = set(tdf['item'].unique())
        if user in item_filter_not_allowed:
            item_filter_allowed[user] = possible_items.difference(item_filter_not_allowed[user])
        else:
            item_filter_allowed[user] = possible_items

    return item_filter_allowed


def filter_data_allowed_items(data: pd.DataFrame, allowed_items: set) -> pd.DataFrame:
    df = data.copy()
    df = df[df['item'].isin(allowed_items)]
    return df
