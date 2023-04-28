import numpy as np


def avg(data: np.array, min_count: int):
    results = []
    extras = []
    for mc in min_count:
        unique_items, counts = np.unique(data[:, 2], return_counts=True)
        keep_items = unique_items[counts >= mc]
        items, ratings = np.hsplit(data[np.argsort(data[:, 2]), 2:4], 2)
        item_splits = np.where(items[:-1] != items[1:])[0] + 1
        arrs = np.split(ratings, item_splits)
        avg_ratings = np.vstack([np.mean(a) for a in arrs])
        if len(unique_items):
            arr = np.hstack((unique_items.reshape(unique_items.shape[0], 1), avg_ratings))
            arr = arr[np.isin(unique_items, keep_items, assume_unique=True), :]
            results.append(arr)
        else:
            results.append(None)
        extras.append({})
    return extras, results
