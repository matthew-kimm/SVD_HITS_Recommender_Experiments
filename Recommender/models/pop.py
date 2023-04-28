import numpy as np


def popular(data: np.array):
    items, counts = np.unique(data[:, 2], return_counts=True)
    arr = np.vstack((items, counts)).T
    extras = [{}]
    results = [arr]
    return extras, results
