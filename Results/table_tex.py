import argparse
import numpy as np
import pandas as pd


def underline(value: float, idx: int, column: pd.Series, maximum: bool, rounding: int):
    if maximum:
        ul = bool(idx == column.idxmax())
    else:
        ul = bool(idx == column.idxmin())
    if ul:
        return f'\\underline{{{round(value, rounding)}}}'
    else:
        return f'{round(value, rounding)}'


def model_name_from_variation(model_name: str, variation: str):
    if type(variation) == str:
        return f'{model_name}({variation})'
    else:
        return model_name


def make_latex_table(experiment_dir: str, table_type: str):
    parameter_type_map = {'req_rating': float, 'xi': float, 'power': int,
                          'variation': str, 'min_count': int, 'd': int}
    parameter_symbol_map = {'req_rating': '$z$', 'xi': '$\\xi$', 'power': '$q$',
                            'variation': '$v$', 'min_count': '$c$', 'd': '$d$'}

    results_dir = f'{experiment_dir}results'
    table = pd.read_csv(f'{results_dir}/{table_type}_table.csv')
    parameter_columns = list(table.columns)[2:min([i for i, col in enumerate(list(table.columns)) if col.startswith('recall_')])]

    primary = table_type
    secondary = 'pf' if table_type == 'gb' else 'gb'
    positive = 'good' if table_type == 'gb' else 'passed'
    negative = 'bad' if table_type == 'gb' else 'failed'
    txt_head = f"\\begin{{table}}[h!]\n\scriptsize\n\centering\n"+\
    f"\caption{{Best models with respect to \\textit{{Recall(diff)}}$_{{{primary}}}$ from the parameter analysis.}} \n" +\
    f"\\begin{{tabular}}{{|c|c|c|c|c|c|c|}} \n"+\
    "\hline \n"+\
    f"\\textbf{{Model}} & \\textbf{{Parameters}} & \\textbf{{\\textit{{Recall({positive})}}}} & \\textbf{{\\textit{{Recall({negative})}}}} & \\textbf{{\\textit{{Recall(diff)}}}}$_{{{primary}}}$ & \\textbf{{\\textit{{Recall(diff)}}}}$_{{{secondary}}}$ & \\textbf{{\\textit{{Matched}}}}  \\\\ \n" + \
    "\hline \n"

    txt_content = ""
    for i, row in table.iterrows():
        parameters = ""
        for col in parameter_columns:
            if type(row[col]) == str or not np.isnan(row[col]):
                parameters += f'{parameter_symbol_map[col]} = {parameter_type_map[col](row[col])}\\\\ '
        txt_content += f"{model_name_from_variation(row['model'], row['variation'])} & \makecell{{{parameters}}} &" +\
            f" {underline(row[f'recall_{positive}'], i, table[f'recall_{positive}'], True, 3)} &" +\
            f" {underline(row[f'recall_{negative}'], i, table[f'recall_{negative}'], False, 3)} &" + \
            f" {underline(row[f'recall_diff_{primary}'], i, table[f'recall_diff_{primary}'], True, 3)} &" + \
            f" {underline(row[f'recall_diff_{secondary}'], i, table[f'recall_diff_{secondary}'], True, 3)} &" + \
            f" {underline(row['total_count_recommended_rated'], i, table['total_count_recommended_rated'], True, 1)} \\\\ \n \hline \n"

    txt_tail = f"\end{{tabular}}\n\label{{tab:pa:{primary}}}\n\end{{table}}"

    txt = txt_head + txt_content + txt_tail

    with open(f'{results_dir}/{primary}_table.tex.txt', 'w') as f:
        f.write(txt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="Make LaTeX Table",
        description="Make LaTeX table for recall-diff-pf or recall-diff-gb",
        epilog="----"
    )
    parser.add_argument('experiment_dir', type=str, help="path to experiment directory with trailing /")
    parser.add_argument('table_type', type=str, help='table to construct either pf for pass/fail or gb for good/bad')
    args = parser.parse_args()
    make_latex_table(args.experiment_dir, args.table_type)
