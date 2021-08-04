import os
import copy
from time import time

import optuna

from utils import Logger, ResultTable, set_random_seed
from experiment import train_model

class BayesianSearch:
    def __init__(self, model_base, dataset, evaluator, early_stop, logger, config, device, seed=2020, num_trials=10, num_parallel=1):
        self.model_base = model_base
        self.dataset = dataset
        self.evaluator = evaluator
        self.early_stop = early_stop
        self.base_logger = logger
        self.base_dir = logger.log_dir
        self.metric_to_optimize = early_stop.early_stop_measure
        self.config = config
        self.seed = seed
        
        self.num_trials = num_trials
        self.num_parallel = num_parallel

        self.exp_logger = []

        self.search_params = config['BayesSearch']

        # (score, [list of params])
        self.result = None
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_param = None

        self.device = device

    def generate_search_space(self, trial):
        # For integer: ('int', [low, high])
        # For float: ('float', 'domai', [low, high])
        # For categorical: ('categorical', [list of choices])
        search_spaces = {}

        for param_name in self.search_params:
            space = self.search_params[param_name]
            space_type = space[0]

            if space_type == 'categorical':
                search_spaces[param_name] = trial.suggest_categorical(param_name, space[1])
            elif space_type == 'int':
                [low, high] = space[1]
                search_spaces[param_name] = trial.suggest_int(param_name, low, high)
            elif space_type == 'float':
                domain = space[1]
                [low, high] = space[2]
                if domain == 'uniform':
                    search_spaces[param_name] = trial.suggest_uniform(param_name, low, high)
                elif domain == 'loguniform':
                    search_spaces[param_name] = trial.suggest_loguniform(param_name, low, high)
                else:
                    raise ValueError('Unsupported float search domain: %s' % domain)
            else:
                raise ValueError('Search parameter type error: %s' % space_type)
        
        return search_spaces

    def optimize(self, model, early_stop, logger, config):
        valid_score, train_time = train_model(model, self.dataset, self.evaluator, early_stop, logger, config)
        self.valid_score.append(valid_score)

        score = valid_score[self.metric_to_optimize]
        
        valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]
        logger.info(', '.join(valid_score_str))

        return score

    def objective_function(self, trial):
        # update config
        config_copy = copy.deepcopy(self.config)

        # config_to_update = dict(zip(self.search_param_names, cur_space))
        search_params = self.generate_search_space(trial)
        config_copy.update_params(search_params)
        
        set_random_seed(self.seed)
        
        # create model
        model = self.model_base(self.dataset, config_copy['Model'], self.device)

        exp_logger = self.init_search_logger()

        score = self.optimize(model, self.early_stop, exp_logger, config_copy)
        exp_logger.close()

        self.exp_num += 1

        del model

        return score

    def init_search_logger(self):
        exp_dir = os.path.join(self.base_dir, 'exp_%d' % self.exp_num)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        logger = Logger(exp_dir)
        self.exp_logger.append(logger)
        return logger

    def search(self):
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(self.objective_function, n_trials=self.num_trials, n_jobs=self.num_parallel)
        all_trials = sorted(self.study.trials, key=lambda x: x.value, reverse=True)
        best_trial = all_trials[0]

        self.best_exp_num = best_trial.number
        self.best_score = best_trial.value
        self.best_params = best_trial.params

        search_result_table = ResultTable(table_name='Param Search Result', header=list(self.best_params.keys()) + [self.metric_to_optimize], float_formatter='%.6f')
        for trial in all_trials:
            row_dict = {}
            row_dict[self.metric_to_optimize] = trial.value
            for k, v in trial.params.items():
                row_dict[k] = v
            search_result_table.add_row('Exp %d' % trial.number, row_dict)
        
        if optuna.visualization.is_available():
            optuna.visualization.plot_optimization_history(self.study)

        return search_result_table

    def best_result(self):
        best_dir = self.exp_logger[self.best_exp_num].log_dir
        best_valid_score = self.valid_score[self.best_exp_num]

        best_config = copy.deepcopy(self.config)
        best_config.update_params(self.best_params)

        return best_dir, best_valid_score, best_config