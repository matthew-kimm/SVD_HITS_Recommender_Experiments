import numpy as np
import pandas as pd
from typing import Tuple, Literal
from Recommender.shared.item_filter import filter_data_allowed_items


def cartesian_product(x: np.array, y: np.array):
    m = x.shape[0]
    n = y.shape[0]
    left = np.tile(x, (n, 1))
    right = np.repeat(y, m, axis=0)
    product = np.hstack((left, right))
    return product


def item_item_data(data: np.array):
    # assumes each user in data has a history (time = 0) and target (time = 1)
    arr = data.copy()
    arr = arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0])), :]
    times = arr[:, 1]
    user_target_history_split = np.where(times[:-1] != times[1:])[0] + 1
    item_rating = arr[:, [0, 2, 3, 4]]
    uth = np.vsplit(item_rating, user_target_history_split)
    history_target = zip(uth[::2], uth[1::2])
    result = np.vstack([cartesian_product(x, y) for x, y in history_target])
    result = result[:, [0, 1, 2, 5, 6, 7]]
    # format of result = user, item1, irating1, item2, irating2, user_avg_irating
    return result


def fast_pivot_value(data: np.array):
    # key 1, key 2, value
    arr = data.copy()
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0])), :]
    # assumes unique key1, key2 in data passed
    # if duplicates, last value for key1, key2 is stored
    key1, key1_idx = np.unique(arr[:, 0], return_inverse=True)
    key1 = key1.astype(int)
    m = len(key1)
    key2, key2_idx = np.unique(arr[:, 1], return_inverse=True)
    key2 = key2.astype(int)
    n = len(key2)

    matrix = np.zeros((m, n), dtype=float)
    matrix[key1_idx, key2_idx] = arr[:, 2]

    return key1, key2, matrix


def fast_agg_sum(data: np.array):
    # key1, key2, value
    # sums value by key1, key2
    arr = data.copy()
    arr = arr[np.lexsort((arr[:, 1], arr[:, 0])), :]
    key_change = np.where(np.any(arr[:-1, :2] != arr[1:, :2], axis=1))[0] + 1
    key_sums = np.vstack([np.sum(ka) for ka in np.vsplit(arr[:, [2]], key_change)])
    unique_keys = np.vstack((arr[0, :2], arr[key_change, :2]))
    return unique_keys, key_sums


def co_occurrence_matrix(data: np.array):
    # key1, key2, value
    # sums value by key1, key2
    # then pivots the single value for key1, key2
    unique_keys, key_sums = fast_agg_sum(data)
    co_occurrence_data = np.hstack((unique_keys, key_sums.reshape((unique_keys.shape[0], 1))))
    key1, key2, matrix = fast_pivot_value(co_occurrence_data)
    return key1, key2, matrix


def get_integer_ratings(ratings: np.array):
    # integer_rating -> irating
    # if only comparisons needed on ordinal data, convert float to int for speed
    rs = ratings.copy()
    rs_val, rs_idx = np.unique(rs, return_inverse=True)
    iratings = rs_idx + 1
    return iratings


def get_next_closest_irating(ratings: np.array, possible_ratings: np.array):
    m = len(ratings)
    n = len(possible_ratings)
    diffs = np.repeat(ratings.reshape((m, 1)), n, axis=1) - possible_ratings.reshape((1, n))
    diffs = np.where(diffs >= 0, diffs, np.inf)
    next_closest_irating = np.argmin(diffs, axis=1) + 1
    return next_closest_irating


def get_user_history(user: int, data: pd.DataFrame, target_time: int) -> set:
    df = data.copy()
    user_history = set(df[(df['time'] < target_time) & (df['user'] == user)]['item'].unique())
    return user_history


def history_target_data_split(data: pd.DataFrame, target_time: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = data.copy()
    history = df[df['time'] < target_time]
    target = df[df['time'] == target_time]
    return history, target


def filter_history_target_data_item_filter(history: pd.DataFrame, target: pd.DataFrame,
                                           allowed_items: set) -> Tuple[pd.DataFrame, pd.DataFrame]:
    filtered_history_data = history.copy()
    pre_filter_target_data_users = set(target['user'].unique())
    filtered_target_data = filter_data_allowed_items(target, allowed_items)
    post_filter_target_data_users = set(target['user'].unique())
    # Users who no longer have target data must be removed from history
    users_to_remove = pre_filter_target_data_users.difference(post_filter_target_data_users)
    if len(users_to_remove):
        filtered_history_data = filtered_history_data[~filtered_history_data['user'].isin(users_to_remove)]
    remaining_users = np.array(filtered_history_data['user'].unique())
    return filtered_history_data, filtered_target_data, remaining_users


def arr_filter_users(data: np.array, neighbors: np.array) -> np.array:
    return data[np.isin(data[:, 0], neighbors), :]


def filter_item_item_data_users(data: np.array, remaining_users: np.array):
    arr_item_item_filtered_data = arr_filter_users(data, remaining_users)
    return arr_item_item_filtered_data


def filter_item_item_data_items(data: np.array, allowed_items: np.array) -> np.array:
    return data[np.isin(data[:, 3], allowed_items), :]


def get_recommendations(data: np.array, max_recommendations: int):
    if data is None:
        return set()
    arr = data.copy()
    arr = arr[np.argsort(arr[:, 1]), :]
    arr = arr[::-1, :]
    recommendations = set(arr[:max_recommendations, 0].ravel().astype(int))
    return recommendations


def get_operator_map():
    operator_map = {'>': np.greater,
                    '>=': np.greater_equal,
                    '<': np.less,
                    '<=': np.less_equal,
                    '==': np.equal,
                    '!=': np.not_equal}
    return operator_map


def proportion_rating(data: np.array, how: Literal['>', '>=', '<', '<=', '==', '!='],
                      rating: int):
    operator_map = get_operator_map()
    operator = operator_map[how]
    items, ratings = np.hsplit(data[np.lexsort((data[:, 2],)), 2:4], 2)
    items_change_idx = np.where(items[:-1] != items[1:])[0] + 1
    met_condition = np.where(operator(ratings, rating), 1, 0)
    items, items_count = np.unique(items, return_counts=True)
    proportions = np.array(list(map(np.sum, np.split(met_condition, items_change_idx)))) / items_count
    return proportions
