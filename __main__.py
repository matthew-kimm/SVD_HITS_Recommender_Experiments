import argparse
import numpy as np
from Recommender.run.config import Config
import json
from os.path import exists

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog="",
        description="",
        epilog=""
    )
    parser.add_argument('config', type=str, help="json config file")
    parser.add_argument('-n', '--min_neighbors', type=int,
                        help="minimum number of neighbors for a recommendation", default=0)
    parser.add_argument('-s', '--seed', type=int, help="integer seed", default=0)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        json_config = json.load(f)
        resume = json_config.get('resume', False)
        experiment_name = json_config['name']
        experiment_dir = json_config['output']
        experiment_file = f'{experiment_dir}{experiment_name}__n-{args.min_neighbors}__s-{args.seed}.csv'
        file_exists = exists(experiment_file)
    if resume and file_exists:
        print(f'***\n Skipping ... Result Already Exists (n-{args.min_neighbors}, s-{args.seed}): {experiment_file}\n***')
    else:
        np.random.seed(args.seed)
        config = Config(config_path=args.config, min_neighbors=args.min_neighbors, seed=args.seed)
        config.run()
