from typing import List
from functools import reduce
import pandas as pd


def descriptive_parameter_expansion_model(model: dict) -> List[dict]:
    # format for adding columns for each parameter (will populate nan if not applicable)
    model_type = model['model']
    if model_type == 'POP':
        return [{'model': 'POP'}]
    elif model_type == 'AVG':
        return [{'model': 'AVG', 'min_count': count}
                for count in model['parameters']['min_count']]
    elif model_type == 'HITS':
        return [{'model': 'HITS', 'req_rating': rr, 'xi': xi}
                for rr in model['parameters']['req_rating']
                for xi in model['parameters']['xi']]
    elif model_type == 'HITSW-PF':
        return [{'model': 'HITSW-PF', 'req_rating': rr, 'power': pw, 'xi': xi, 'variation': var}
                for rr in model['parameters']['req_rating']
                for pw in model['parameters']['power']
                for xi in model['parameters']['xi']
                for var in model['parameters']['variation']]
    elif model_type == 'SVD-GB':
        return [{'model': 'SVD-GB', 'variation': var, 'd': d}
                for var in model['parameters']['variation']
                for d in model['parameters']['d']]
    elif model_type == 'SVD-PF':
        return [{'model': 'SVD-PF', 'req_rating': rr, 'variation': var, 'd': d}
                for rr in model['parameters']['req_rating']
                for var in model['parameters']['variation']
                for d in model['parameters']['d']]
    elif model_type == 'SVD-SPF':
        return [{'model': 'SVD-SPF', 'req_rating': rr, 'power': pw, 'variation': var, 'd': d}
                for rr in model['parameters']['req_rating']
                for pw in model['parameters']['power']
                for var in model['parameters']['variation']
                for d in model['parameters']['d']]
    else:
        raise ValueError(f'{model_type} is not a valid model.')


def descriptive_parameter_expansion_models(models: List[dict]) -> List[dict]:
    parameter_expansions = [descriptive_parameter_expansion_model(model) for model in models]
    counts = [len(pe) for pe in parameter_expansions]
    return counts, reduce(lambda x, y: x+y, parameter_expansions)


def descriptive_parameter_columns(parameters: List[dict]) -> pd.DataFrame:
    return pd.json_normalize(parameters)
