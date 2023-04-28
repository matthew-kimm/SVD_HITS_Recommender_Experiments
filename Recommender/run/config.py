import json
import copy
import numpy as np
import pandas as pd
from os.path import exists
from os import mkdir
from Recommender.models.avg import avg
from Recommender.models.pop import popular
from Recommender.models.svd import svd_recommend_gb, svd_recommend_pf, svd_recommend_spf
from Recommender.models.hits import hits, hitsw_pf
from Recommender.data.data import get_integer_ratings, get_next_closest_irating,\
    item_item_data, filter_item_item_data_items,\
    get_user_history, history_target_data_split, filter_history_target_data_item_filter, filter_item_item_data_users,\
    get_recommendations
from Recommender.shared.splitter import train_test_split
from Recommender.shared.neighbor import get_neighbor_table, filter_data_with_users
from Recommender.shared.item_filter import get_item_filter_allowed_table
from Recommender.shared.metric import get_metric_base, recommended_with_metric_base, compute_metrics
from Recommender.shared.parameter import descriptive_parameter_expansion_models, descriptive_parameter_columns


class Config:
    def __init__(self, config_path: str, min_neighbors: int, seed: int):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.min_neighbors = min_neighbors
        self.seed = seed

        self.name = self.config['name']
        self.data = pd.read_csv(self.config['data'])
        self.target_time = self.config['target_time']
        self.irating_map = {}
        self.data_validation()
        self.data_additions()

        if self.config['item_filter']:
            with open(self.config['item_filter'], 'r') as f:
                self.item_filter_out = json.load(f)
                self.item_filter_out = {int(k): set(v) for k, v in self.item_filter_out.items()}
        else:
            self.item_filter_out = {}

        self.train_proportion = self.config['train']
        self.output = self.config['output']

        if not exists(self.output):
            mkdir(self.output)

        self.file_output = f'{self.output}/{self.name}__n-{self.min_neighbors}__s-{self.seed}.csv'

        self.neighbor_attributes = self.config['neighbor']['attributes']
        self.neighbor_overlap = self.config['neighbor']['overlap']
        self.neighbor_round_rating = self.config['neighbor']['round_rating']
        try:
            self.neighbor_global = self.config['neighbor']['global']
        except KeyError:
            self.neighbor_global = False

        self.max_recommendations = self.config['max_recommendations']

        self.metric_pass_rating = self.config['metric']['pass']

        self.model_functions = {
            'AVG': avg,
            'POP': popular,
            'HITS': hits,
            'HITSW-PF': hitsw_pf,
            'SVD-GB': svd_recommend_gb,
            'SVD-PF': svd_recommend_pf,
            'SVD-SPF': svd_recommend_spf
        }

        # Initialize
        self.train, self.test = train_test_split(self.data, self.train_proportion)
        self.neighbors = get_neighbor_table(self.data, self.train, self.test,
                                       self.neighbor_attributes,
                                       self.neighbor_overlap,
                                       self.target_time,
                                       self.neighbor_round_rating,
                                       self.neighbor_global)
        self.item_filter = get_item_filter_allowed_table(self.data,
                                                    self.item_filter_out,
                                                    self.neighbors, self.target_time)
        self.metric_base = get_metric_base(self.data, self.test, pass_rating=self.metric_pass_rating,
                                           target_time=self.target_time)

        self.all_item_item_data = item_item_data(self.data[['user', 'time', 'item', 'irating', 'user_avg_irating']].to_numpy())

        self.models = self.config['models']
        self.original_models = copy.deepcopy(self.models)
        self.convert_model_rating_parameter_to_irating()

    def data_validation(self):
        df = self.data.copy()
        df['check'] = df['time'] < self.target_time
        if sum(df.groupby('user')['check'].nunique() < 2):
            raise ValueError('All users must have items before and for the target time.')

    def data_additions(self):
        self.get_irating()
        self.get_user_avg_irating()

    def get_irating(self):
        self.data['irating'] = get_integer_ratings(self.data['rating'].to_numpy())
        self.irating_map = dict(zip(self.data['rating'].astype(str), self.data['irating']))

    def get_user_avg_irating(self):
        df = self.data.copy()
        df = df[df['time'] < self.target_time]
        user_avg_rating = df.groupby('user').agg(user_avg_rating=pd.NamedAgg('rating', 'mean'))
        self.data = pd.merge(self.data, user_avg_rating, how='inner', on='user')
        possible_ratings = np.sort(np.array(self.data['rating'].unique()))
        self.data['user_avg_irating'] = get_next_closest_irating(self.data['user_avg_rating'].to_numpy(), possible_ratings)

    def get_model_parameters(self, model_type: str, model_parameters: dict, user_history: np.array):
        adjusted_model_parameters = model_parameters.copy()
        if model_type.startswith('SVD'):
            adjusted_model_parameters['user_history'] = user_history
        return adjusted_model_parameters

    def convert_model_rating_parameter_to_irating(self):
        for i, model in enumerate(self.models):
            for parameter, value in model['parameters'].items():
                if 'rating' in parameter:
                    self.models[i]['parameters'][parameter] = [self.irating_map[str(val)] for val in value]

    def run(self):
        extras = []
        results = []
        count_users_in_test = len(self.test)
        # expand parameters to match results
        counts, parameter_expansion = descriptive_parameter_expansion_models(self.original_models)
        parameter_expansion = parameter_expansion * len(self.test)
        parameters = descriptive_parameter_columns(parameter_expansion)

        for i, user in enumerate(sorted(list(self.test))):
            print(f'{i} / {count_users_in_test}')
            user_history = get_user_history(user, self.data, self.target_time)
            user_neighbors = self.neighbors[user]
            if len(user_neighbors) < self.min_neighbors:
                results += [set()] * sum(counts)
                extras += [{}] * sum(counts)
                continue
            user_data_filter_neighbors = filter_data_with_users(self.data, user_neighbors)
            history_data, target_data = history_target_data_split(user_data_filter_neighbors, self.target_time)
            user_item_filter = self.item_filter[user]
            history_data, target_data, remaining_users = filter_history_target_data_item_filter(history_data, target_data, user_item_filter)

            arr_target_rating = target_data[['user', 'time', 'item', 'rating']].to_numpy()
            arr_target_irating = target_data[['user', 'time', 'item', 'irating', 'user_avg_irating']].to_numpy()
            arr_item_item_data = filter_item_item_data_users(self.all_item_item_data, remaining_users)
            user_item_filter_arr = np.array(list(user_item_filter))
            arr_item_item_data = filter_item_item_data_items(arr_item_item_data, user_item_filter_arr)

            model_data_map = {'AVG': arr_target_rating,
                              'POP': arr_target_irating,
                              'HITS': arr_target_irating,
                              'HITSW-PF': arr_target_irating,
                              'SVD-GB': arr_item_item_data,
                              'SVD-PF': arr_item_item_data,
                              'SVD-SPF': arr_item_item_data}

            for j, model in enumerate(self.models):
                model_type = model['model']
                model_parameters = model['parameters']
                model_parameters = self.get_model_parameters(model_type, model_parameters, user_history)
                model_data = model_data_map[model_type]
                if not model_data.shape[0]:
                    results += [set()] * counts[j]
                    extras += [{}] * counts[j]
                    continue
                model_function = self.model_functions[model_type]
                extra, result = model_function(model_data, **model_parameters)
                result = [get_recommendations(res, self.max_recommendations) for res in result]
                results += result
                extras += extra

        extras = pd.json_normalize(extras)
        recommended = np.array(results)
        users_col = pd.DataFrame(np.repeat(np.array(list(self.test)), sum(counts)), columns=['user'])
        metric_recommended = recommended_with_metric_base(self.metric_base, recommended)
        metrics = compute_metrics(metric_recommended)
        output_data = pd.concat([users_col, parameters, metrics, extras], axis=1)
        output_data.to_csv(self.file_output, index=False)
