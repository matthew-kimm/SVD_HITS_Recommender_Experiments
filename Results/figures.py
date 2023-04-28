import pandas as pd
import argparse
import matplotlib.pyplot as plt
from os.path import exists
from os import mkdir


def model_name_from_variation(model_name: str, variation: str):
    if type(variation) == str:
        return f'{model_name}({variation})'
    else:
        return model_name


def global_neighborhood_analysis_figures(experiment_dir: str, experiment_name: str):
    figures_dir = f'{experiment_dir}figures'
    if not exists(figures_dir):
        mkdir(figures_dir)

    results_dir = f'{experiment_dir}results'
    final_result_file = f'{results_dir}/final_result.csv'
    df = pd.read_csv(final_result_file)

    metric_title_map = {'recall_diff_pf': 'Recall-Diff-PF', 'total_count_recommended_rated': 'Matched'}
    for metric in ['recall_diff_pf', 'total_count_recommended_rated']:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        x = df.sort_values(by=['model', 'variation'])[['model', 'variation']].apply(lambda z: model_name_from_variation(z[0], z[1]), axis=1).values
        y = df.sort_values(by=['model', 'variation'])[metric].values
        plt.bar(x, y)

        ax.set_xlabel('Model')
        ax.set_ylabel(metric_title_map[metric])
        plt.xticks(x, x, rotation=80)
        plt.title(f'{metric_title_map[metric]} for {experiment_name} with Varying Neighbors')
        plt.savefig(f'{figures_dir}/{metric}.pdf', bbox_inches='tight')


def vary_neighborhood_analysis_figures(experiment_dir: str, experiment_name: str):
    figures_dir = f'{experiment_dir}figures'
    if not exists(figures_dir):
        mkdir(figures_dir)

    results_dir = f'{experiment_dir}results'
    final_result_file = f'{results_dir}/final_result.csv'
    df = pd.read_csv(final_result_file)

    x = list(sorted(df['n'].unique()))
    models = df['model'].unique()
    markers = ['x', 'D', '|', '2', '^', 'o', 'v']
    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
    metric_title_map = {'recall_diff_pf': 'Recall-Diff-PF', 'total_count_recommended_rated': 'Matched'}

    for metric in ['recall_diff_pf', 'total_count_recommended_rated']:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i, model in enumerate(models):
            y = df[df['model'] == model].sort_values(by=['n'])[metric].values
            ax.plot(x, y, f'{markers[i % 7]}-{colors[i % 7]}', label=model)

        ax.legend()
        ax.set_xlabel('Min Neighbor Count $n$')
        ax.set_ylabel(metric_title_map[metric])
        plt.xticks(x, x, rotation=45)
        plt.title(f'{metric_title_map[metric]} for {experiment_name} with Varying Neighbors')
        plt.savefig(f'{figures_dir}/{metric}.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Figures",
        description="Draws figures for neighborhood analysis",
        epilog="----"
    )
    parser.add_argument('experiment_dir', type=str, help="path to experiment directory with trailing /")
    parser.add_argument('experiment_name', type=str, help='name of experiment in config file (prefix of result files), use quotes if contain space')
    parser.add_argument('--is_global_neighborhood_analysis', action=argparse.BooleanOptionalAction, help='use if drawing figure for global_neighborhood_analysis')
    args = parser.parse_args()

    if args.is_global_neighborhood_analysis:
        global_neighborhood_analysis_figures(args.experiment_dir, args.experiment_name)
    else:
        vary_neighborhood_analysis_figures(args.experiment_dir, args.experiment_name)

