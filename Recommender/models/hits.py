import numpy as np
from typing import Literal, Tuple, List
from Recommender.data.data import fast_pivot_value, proportion_rating, get_operator_map


def hits_power_iteration(matrix: np.array, xi: float, tol: float = 10e-5, max_iter: int = 100) -> np.array:
    n = matrix.shape[0]
    matrix_xi = xi * matrix
    x = np.ones((n, 1))
    t = (x - xi) / n
    for i in range(max_iter):
        xnew = matrix_xi@x + t
        xnew = xnew / np.linalg.norm(xnew, 1)
        if np.linalg.norm(xnew - x, 1) < tol:
            break
        x = xnew
    return xnew


def gram_matrix(matrix: np.array):
    return matrix.T@matrix


def scale_matrix(matrix: np.array, left: np.array = 1, right: np.array = 1) -> np.array:
    return left * matrix * right


def get_hits_base_matrix(data: np.array) -> Tuple[np.array, np.array, np.array]:
    key_1, key_2, matrix = fast_pivot_value(data[:, [0, 2, 3]])
    return key_1, key_2, matrix


def get_hits_filter_matrix(data: np.array, how: Literal['>', '>=', '<', '<=', '==', '!='],
                           rating: int) -> Tuple[np.array, np.array, np.array]:
    arr = data.copy()
    operator_map = get_operator_map()
    operator = operator_map[how]
    arr[:, 3] = np.where(operator(data[:, 3], rating), 1, 0)
    key_1, key_2, matrix = get_hits_base_matrix(arr)
    return key_1, key_2, matrix


def hits(data: np.array, req_rating: List[float], xi: List[float]):
    users, items, matrix = get_hits_base_matrix(data)
    m, n = matrix.shape
    results = []
    extras = []
    for rating in req_rating:
        users, items, pos_matrix = get_hits_filter_matrix(data, how='>=', rating=rating)
        pos_matrix = gram_matrix(pos_matrix)

        for x in xi:
            ratings = hits_power_iteration(pos_matrix, x)
            result = np.hstack((items.reshape((n, 1)), ratings.reshape((n, 1))))
            results.append(result)
            extras.append({'neighbors': m, 'target_items': n})
    return extras, results


def hitsw_pf(data: np.array, xi: List[float], req_rating: List[int], power: List[int],
             variation: List[Literal['+', '+-']]):
    users, items, matrix = get_hits_base_matrix(data)
    m, n = matrix.shape
    results = []
    extras = []
    for rating in req_rating:
        pos_scale = proportion_rating(data, how='>=', rating=rating)
        neg_scale = proportion_rating(data, how='<', rating=rating)
        for pw in power:
            pos_matrix = gram_matrix(
                scale_matrix(matrix, right=np.power(pos_scale, pw).reshape((1, n))))
            neg_matrix = gram_matrix(
                scale_matrix(matrix, right=np.power(neg_scale, pw).reshape((1, n))))
            for x in xi:
                pos_result = hits_power_iteration(pos_matrix, x)
                neg_result = hits_power_iteration(neg_matrix, x)
                var_results = {'+': pos_result, '+-': pos_result - neg_result}
                for var in variation:
                    result = var_results[var]
                    result = np.hstack((items.reshape((n, 1)), result.reshape((n,1))))
                    results.append(result)
                    extras.append({'neighbors': m, 'target_items': n})
    return extras, results
