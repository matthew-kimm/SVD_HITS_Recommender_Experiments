import numpy as np
from typing import List, Literal, Tuple
from Recommender.data.data import co_occurrence_matrix, proportion_rating
import warnings


def svd_recommend(matrix: np.array, users_item_history_indicator: np.array, d_iter: List[int],
                  items: np.array) -> Tuple[List[dict], List[np.array]]:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        F = matrix / np.linalg.norm(matrix, ord=1, axis=1, keepdims=True)
    F = np.nan_to_num(F, nan=0.0)
    u, s, vh = np.linalg.svd(F)
    m, n = u.shape[0], vh.shape[0]
    results = []
    extras = []
    for d in d_iter:
        k = min(d, m, n)
        ue = u[:, :k]*np.sqrt(s[:k]).reshape((1, k))
        ve = vh.T[:, :k]*np.sqrt(s[:k]).reshape((1, k))
        p = users_item_history_indicator.reshape((1, m))@ue
        r = ve@p.T
        result = np.hstack((items.reshape((n, 1)), r))
        results.append(result)
        extras.append({'history_items': m, 'target_items': n})
    return extras, results


def item_item_matrix_gb(data: np.array):
    # u, i1, ir1, i2, ir2, aur
    arr = data.copy()
    good = np.zeros((arr.shape[0], 3))
    good[:, :2] = arr[:, [1, 3]]
    bad = np.zeros((arr.shape[0], 3))
    bad[:, :2] = arr[:, [1, 3]]
    good[:, 2] = np.where(arr[:, 4] >= arr[:, 5], 1, 0)
    bad[:, 2] = np.where(arr[:, 4] < arr[:, 5], 1, 0)
    key1, key2, good = co_occurrence_matrix(good)
    _, _, bad = co_occurrence_matrix(bad)
    return key1, key2, good, bad


def item_item_matrix_pf(data: np.array, req_rating: int):
    # u, i1, ir1, i2, ir2, aur
    arr = data.copy()
    passed = np.zeros((arr.shape[0], 3))
    passed[:, :2] = arr[:, [1, 3]]
    failed = np.zeros((arr.shape[0], 3))
    failed[:, :2] = arr[:, [1, 3]]
    passed[:, 2] = np.where(arr[:, 4] >= req_rating, 1, 0)
    failed[:, 2] = np.where(arr[:, 4] < req_rating, 1, 0)
    key1, key2, passed = co_occurrence_matrix(passed)
    _, _, failed = co_occurrence_matrix(failed)
    return key1, key2, passed, failed


def svd_recommend_gb(data: np.array, user_history: set, d: List[int],
                     variation: List[Literal['+-', '+', '++']]) -> Tuple[list, List[np.array]]:
    idx, col, good, bad = item_item_matrix_gb(data)
    co_occur_types = {'+-': good - bad, '+': good, '++': good + bad}
    uh = np.isin(idx, np.array(list(user_history)).astype(int))
    extras = []
    results = []
    for var in variation:
        extra, result = svd_recommend(co_occur_types[var], uh, d, col)
        extras += extra
        results += result
    return extras, results


def svd_recommend_pf(data: np.array, req_rating: List[int], user_history: set, d: List[int],
                     variation: List[Literal['+-', '+', '++']]) -> Tuple[list, List[np.array]]:
    extras = []
    results = []
    for rr in req_rating:
        idx, col, passed, failed = item_item_matrix_pf(data, rr)
        co_occur_types = {'+-': passed - failed, '+': passed, '++': passed + failed}
        uh = np.isin(idx, np.array(list(user_history)).astype(int))
        for var in variation:
            extra, result = svd_recommend(co_occur_types[var], uh, d, col)
            extras += extra
            results += result
    return extras, results


def svd_recommend_spf(data: np.array, req_rating: List[int], power: List[int], user_history: set, d: List[int],
                     variation: List[Literal['+-', '+', '++']]) -> Tuple[list, List[np.array]]:
    extras = []
    results = []
    for rr in req_rating:
        idx, col, passed, failed = item_item_matrix_pf(data, rr)
        scale_passed = proportion_rating(data[:, [0, 1, 3, 4]], how='>=', rating=rr)
        scale_failed = proportion_rating(data[:, [0, 1, 3, 4]], how='<', rating=rr)
        for pw in power:
            passed_pw = passed@np.diag(np.power(scale_passed, pw))
            failed_pw = failed@np.diag(np.power(scale_failed, pw))
            co_occur_types = {'+-': passed_pw - failed_pw, '+': passed_pw, '++': passed_pw + failed_pw}
            uh = np.isin(idx, np.array(list(user_history)).astype(int))
            for var in variation:
                extra, result = svd_recommend(co_occur_types[var], uh, d, col)
                extras += extra
                results += result
    return extras, results
