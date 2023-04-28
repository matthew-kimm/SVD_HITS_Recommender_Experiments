import json
import pandas as pd
import numpy as np
import argparse


def setup_global_experiment(experiment_dir: str):
    parameter_type_map = {'req_rating': float, 'xi': float, 'power': int,
                          'variation': str, 'min_count': int, 'd': int}

    with open('configs/templates/global-template.json', 'r') as f:
        global_experiment_config = json.load(f)

    results_dir = f'{experiment_dir}results'
    pf_analysis = f'{results_dir}/pf_table.csv'
    df = pd.read_csv(pf_analysis)
    for i, row in df.iterrows():
        model = row['model']
        parameters = {}
        parameter_columns = list(df.columns)[2:min([j for j, col in enumerate(df.columns) if col.startswith('recall_')])]
        for col in parameter_columns:
            if type(row[col]) == str or not np.isnan(row[col]):
                parameters[col] = [parameter_type_map[col](row[col])]
        global_experiment_config['models'].append({'model': model, 'parameters': parameters})

    with open('configs/global.json', 'w') as f:
        json.dump(global_experiment_config, f, indent=2)


def setup_compare_neighbor_experiments(experiment_dir: str, setup_experiment: str):
    parameter_type_map = {'req_rating': float, 'xi': float, 'power': int,
                          'variation': str, 'min_count': int, 'd': int}

    with open(f'configs/templates/{setup_experiment}-template.json', 'r') as f:
        setup_experiment_config = json.load(f)

    results_dir = f'{experiment_dir}results'
    pf_analysis = f'{results_dir}/best_pf_table.csv'
    df = pd.read_csv(pf_analysis)
    for i, row in df.iterrows():
        model = row['model']
        parameters = {}
        parameter_columns = list(df.columns)[2:min([j for j, col in enumerate(df.columns) if col.startswith('recall_')])]
        for col in parameter_columns:
            if type(row[col]) == str or not np.isnan(row[col]):
                parameters[col] = [parameter_type_map[col](row[col])]
        setup_experiment_config['models'].append({'model': model, 'parameters': parameters})

    with open(f'configs/neighbor/{setup_experiment}.json', 'w') as f:
        json.dump(setup_experiment_config, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Setup Experiments using Parameter Analysis Results (best models wrt recall-diff-pf)",
        description="Experiments for Global Neighborhood, Exact Grade Neighborhood," +
        "Letter Grade Neighborhood, Exact Grade Neighborhood with Attributes," +
        "Exact Grade Neighborhood with Filter",
        epilog="----"
    )
    parser.add_argument('experiment_dir', type=str, help="path to parameter analysis experiment directory with trailing /")
    args = parser.parse_args()
    setup_global_experiment(args.experiment_dir)
    for experiment in ['exact', 'letter', 'attribute', 'filtered']:
        setup_compare_neighbor_experiments(args.experiment_dir, experiment)
