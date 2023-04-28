import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from os import mkdir


def svd_spf_heatmap(experiment_dir: str):
    figures_dir = f'{experiment_dir}figures'
    if not exists(figures_dir):
        mkdir(figures_dir)

    results_dir = f'{experiment_dir}results'
    final_result_file = f'{results_dir}/final_result.csv'
    df = pd.read_csv(final_result_file)

    df = df[(df['power'] == 1) & (df['variation'] == '+-')]
    df = df.groupby(['req_rating', 'd'])['recall_diff_pf'].agg('mean').unstack()
    df = df.apply(lambda x: round(x, 3))
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, ax=ax, fmt='.3f')
    ax.set(xlabel="$d$", ylabel="$z$")
    plt.title(f'Recall-Diff-PF for SVD-SPF(+-) by $d$ and $z$')
    plt.savefig(f'{figures_dir}/svdspf_heatmap.pdf', bbox_inches='tight')


def hitsw_heatmap(experiment_dir: str):
    figures_dir = f'{experiment_dir}figures'
    if not exists(figures_dir):
        mkdir(figures_dir)

    results_dir = f'{experiment_dir}results'
    final_result_file = f'{results_dir}/final_result.csv'
    df = pd.read_csv(final_result_file)

    df = df[(df['power'] == 1) & (df['variation'] == '+-')]
    df = df.groupby(['req_rating', 'xi'])['recall_diff_pf'].agg('mean').unstack()
    df = df.apply(lambda x: round(x, 3))
    f, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df, annot=True, ax=ax, fmt='.3f')
    ax.set(xlabel="$\\xi$", ylabel="$z$")
    plt.title(f'Recall-Diff-PF for HITSW(+-) by $\\xi$ and $z$')
    plt.savefig(f'{figures_dir}/hitws_heatmap.pdf', bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Heatmap for Parameter Analysis",
        description="Draws heatmaps for Parameter Analysis",
        epilog="----"
    )
    parser.add_argument('experiment_dir', type=str, help="path to experiment directory with trailing /")
    parser.add_argument('--draw_svd_spf', action=argparse.BooleanOptionalAction, help='use if drawing heatmap for svd-spf(+-)')
    parser.add_argument('--draw_hitsw', action=argparse.BooleanOptionalAction, help='use if drawing heatmap for hitsw(+-)')
    args = parser.parse_args()

    if args.draw_svd_spf:
        svd_spf_heatmap(args.experiment_dir)
    if args.draw_hitsw:
        hitsw_heatmap(args.experiment_dir)
