import numpy as np
import pandas as pd
import glob
import argparse
from os import mkdir
from os.path import exists


def get_results(experiment_dir: str, experiment_name: str, is_parameter_analysis: bool):
    dfs_avg = []
    dfs_total = []
    for file in glob.glob(f'{experiment_dir}{experiment_name}__n-*__s-*.csv'):
        extra_params = file.split('/')[-1].removeprefix(f'{experiment_name}__').removesuffix('.csv').split('__')
        extra_params = {k: v for k, v in map(lambda kv: kv.split('-'), extra_params)}
        tdf = pd.read_csv(file)
        for key, value in extra_params.items():
            if key != 's':
                tdf.insert(1, key, value)
        tdf = tdf.drop(columns=['user'])
        tgroupby_columns = list(tdf.columns)[0:min([i for i, col in enumerate(tdf.columns) if col.startswith('count')])]
        tcount_columns = [col for col in tdf.columns if col.startswith('count')]
        tdf[tgroupby_columns] = tdf[tgroupby_columns].fillna('NA')
        tavg = tdf.groupby(tgroupby_columns).mean().reset_index()
        ttotal = tdf.groupby(tgroupby_columns)[tcount_columns].sum().reset_index()
        ttotal = ttotal.rename(columns={k: 'total_' + k for k in tcount_columns})
        tavg['recall_diff_gb'] = tavg['recall_good'] - tavg['recall_bad']
        tavg['recall_diff_pf'] = tavg['recall_passed'] - tavg['recall_failed']
        dfs_avg.append(tavg)
        dfs_total.append(ttotal)
    df_avg = pd.concat(dfs_avg) if len(dfs_avg) > 1 else dfs_avg[0]
    df_total = pd.concat(dfs_total) if len(dfs_total) > 1 else dfs_total[0]

    groupby_columns = list(df_avg.columns)[0:min([i for i, col in enumerate(df_avg.columns) if col.startswith('count')])]
    avg = df_avg.groupby(groupby_columns).mean().reset_index()
    total = df_total.groupby(groupby_columns).mean().reset_index()
    final = pd.merge(avg, total, how='inner', on=groupby_columns)

    results_dir = f'{experiment_dir}results/'
    if not exists(results_dir):
        mkdir(results_dir)

    final.replace('NA', np.nan).to_csv(f'{results_dir}final_result.csv', index=False)

    if is_parameter_analysis:
        gb_table = final.loc[final.groupby(['model', 'variation'])['recall_diff_gb'].idxmax()][groupby_columns + ['recall_good', 'recall_bad', 'recall_diff_gb', 'recall_diff_pf', 'total_count_recommended_rated']]
        gb_table.replace('NA', np.nan).to_csv(f'{results_dir}gb_table.csv', index=False)
        pf_table = final.loc[final.groupby(['model', 'variation'])['recall_diff_pf'].idxmax()][groupby_columns + ['recall_passed', 'recall_failed', 'recall_diff_pf', 'recall_diff_gb', 'total_count_recommended_rated']]
        pf_table.replace('NA', np.nan).to_csv(f'{results_dir}pf_table.csv', index=False)

        best_gb_table = final.loc[final.groupby('model')['recall_diff_gb'].idxmax()][groupby_columns + ['recall_good', 'recall_bad', 'recall_diff_gb', 'recall_diff_pf', 'total_count_recommended_rated']]
        best_gb_table.replace('NA', np.nan).to_csv(f'{results_dir}best_gb_table.csv', index=False)
        best_pf_table = final.loc[final.groupby('model')['recall_diff_pf'].idxmax()][groupby_columns + ['recall_passed', 'recall_failed', 'recall_diff_pf', 'recall_diff_gb', 'total_count_recommended_rated']]
        best_pf_table.replace('NA', np.nan).to_csv(f'{results_dir}best_pf_table.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Results",
        description="Aggregates Results for Experiments",
        epilog="----"
    )
    parser.add_argument('experiment_dir', type=str, help="path to experiment directory with trailing /")
    parser.add_argument('experiment_name', type=str, help='name of experiment in config file (prefix of result files), use quotes if contain space')
    parser.add_argument('--is_parameter_analysis', action=argparse.BooleanOptionalAction, help='use if collecting results for parameter analysis')
    args = parser.parse_args()
    get_results(args.experiment_dir, args.experiment_name, args.is_parameter_analysis)
