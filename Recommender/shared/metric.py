import pandas as pd
import numpy as np
from Recommender.shared.neighbor import filter_data_with_users
from functools import reduce


def get_metric_base(data: pd.DataFrame, test: set, pass_rating: float, target_time: int) -> pd.DataFrame:
    df = data.copy()
    df = df[df['time'] == target_time]
    df = filter_data_with_users(df, test)
    df['passed'] = df['rating'] >= pass_rating
    df['failed'] = df['rating'] < pass_rating
    df['good'] = df['rating'] >= df['user_avg_rating']
    df['bad'] = df['rating'] < df['user_avg_rating']

    rated = df.groupby('user').agg(rated=pd.NamedAgg('item', set))
    passed = df[df['passed']]
    passed = passed.groupby('user').agg(passed=pd.NamedAgg('item', set))
    failed = df[df['failed']]
    failed = failed.groupby('user').agg(failed=pd.NamedAgg('item', set))
    good = df[df['good']]
    good = good.groupby('user').agg(good=pd.NamedAgg('item', set))
    bad = df[df['bad']]
    bad = bad.groupby('user').agg(bad=pd.NamedAgg('item', set))

    metric = reduce(lambda df1, df2: pd.merge(df1, df2, how='left', on='user'), [rated, passed, failed, good, bad])
    metric = pd.DataFrame(np.where(metric.isnull(), set(), metric), columns=metric.columns, index=metric.index)
    metric = metric.reset_index(drop=True)
    return metric


def recommended_with_metric_base(metric_base: pd.DataFrame, recommended: np.array):
    # computes metric base intersect recommended
    result = metric_base.copy()
    recommended_arr = recommended.copy()
    columns = list(result.columns)
    m, n = result.shape
    n_recommend_models_and_users = len(recommended)
    if n_recommend_models_and_users % m != 0:
        raise ValueError('Total Users*Models is not divisible by Models')
    total_models = n_recommend_models_and_users // m
    recommended_form = np.repeat(recommended_arr.reshape((n_recommend_models_and_users, 1)), n, axis=1)
    result_form = np.repeat(np.array(result), total_models, axis=0)
    result_recommended = result_form - (result_form - recommended_form)
    result = pd.DataFrame(np.hstack((result_form, result_recommended)),
                          columns=columns + [f'recommended_{col}' for col in columns])
    result['recommended'] = recommended_arr
    return result


def compute_metrics(data: pd.DataFrame):
    metrics = ['passed', 'failed', 'good', 'bad']
    df = data.copy()
    df[[f'count_{col}' for col in df.columns]] = df.applymap(len)
    df = df.drop(columns=[col for col in df.columns if not col.startswith('count')])
    for metric in metrics:
        df[f'recall_{metric}'] = df[f'count_recommended_{metric}'] / df[f'count_{metric}']
    for metric in metrics:
        df[f'precision_{metric}'] = df[f'count_recommended_{metric}'] / df[f'count_recommended']
    return df
