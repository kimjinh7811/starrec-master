import os
import copy
from joblib import Parallel, delayed

from experiment import train_model
from utils import Logger, ResultTable, set_random_seed

class GridSearch:
    def __init__(self, model_base, dataset, evaluator, early_stop, logger, config, device, seed=2020, num_parallel=1):
        self.model_base = model_base
        self.dataset = dataset
        self.evaluator = evaluator
        self.early_stop = early_stop
        self.base_logger = logger
        self.base_dir = logger.log_dir
        self.metric_to_optimize = early_stop.early_stop_measure
        self.config = config
        self.seed = seed
        # self.num_parallel = num_parallel
        self.num_parallel = 1

        self.exp_logger = []

        self.search_params = config['GridSearch']
        self.search_space = self.generate_search_space()
        self.search_param_names = list(self.search_params.keys())

        # (score, [list of params])
        self.result = []
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_params = None

        self.device = device

    def generate_search_space(self):
        search_params = list(self.search_params.keys())
        search_space = []
        for i, param_name in enumerate(search_params):
            new_space = []
            
            if not isinstance(self.search_params[param_name], list):
                self.config[param_name] = self.search_params[param_name]

            for param_value in self.search_params[param_name]:
                if i == 0:
                    new_space.append({param_name: param_value})
                else:
                    tmp_list = copy.deepcopy(search_space)
                    for param_setting in tmp_list:
                        param_setting[param_name] = param_value
                    new_space += tmp_list
            search_space = new_space

        return search_space

    def optimize(self, model, early_stop, logger, config):
        valid_score, train_time = train_model(model, self.dataset, self.evaluator, early_stop, logger, config)
        
        score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]
        logger.info('Exp %d best valid score: ' % self.exp_num + ', '.join(score_str))
        self.valid_score.append(valid_score)

        score = valid_score[self.metric_to_optimize]

        return score

    def objective_function(self, cur_space):
        # update config
        config_copy = copy.deepcopy(self.config)
        config_copy.update_params(cur_space)

        set_random_seed(self.seed)

        # create model
        model = self.model_base(self.dataset, config_copy['Model'], self.device)

        exp_logger = self.init_search_logger()

        score = self.optimize(model, self.early_stop, exp_logger, config_copy)
        if score > self.best_score:
            self.best_score = score
            self.best_params = cur_space

        self.base_logger.info('Exp %d value=%.4f, current best value=%.4f with parameters %s\n' % (self.exp_num, score, self.best_score, str(self.best_params)))

        exp_logger.close()
        del model

        self.exp_num += 1

        return score

    def init_search_logger(self):
        exp_dir = os.path.join(self.base_dir, 'exp_%d' % self.exp_num)
        if not os.path.exists(exp_dir):
            os.mkdir(exp_dir)

        logger = Logger(exp_dir)
        self.exp_logger.append(logger)
        return logger

    def search(self):
        self.result = []
        
        if self.num_parallel == 1:
            scores = []
            for cur_space in self.search_space:
                score = self.objective_function(cur_space)
                scores.append(score)
        else:
            with Parallel(n_jobs=self.num_parallel, prefer="threads") as parallel:
                scores = parallel(
                    delayed(self.objective_function)(cur_space)
                    for cur_space in self.search_space)
        results = [
            {'number': i, 'value': scores[i], 'params': cur_space}
            for i, cur_space in enumerate(self.search_space)]
        
        all_trials = sorted(results, key=lambda x: x['value'], reverse=True)
        best_trial = all_trials[0]

        self.best_exp_num = best_trial['number']
        self.best_score = best_trial['value']
        self.best_params = best_trial['params']

        search_result_table = ResultTable(table_name='Param Search Result', header=list(self.best_params.keys()) + [self.metric_to_optimize], float_formatter='%.6f')
        for trial in all_trials:
            row_dict = {}
            row_dict[self.metric_to_optimize] = trial['value']
            for k, v in trial['params'].items():
                row_dict[k] = v
            search_result_table.add_row('Exp %d' % trial['number'], row_dict)

        return search_result_table

    def best_result(self):
        best_dir = self.exp_logger[self.best_exp_num].log_dir
        best_valid_score = self.valid_score[self.best_exp_num]

        best_config = copy.deepcopy(self.config)
        best_config.update_params(self.best_params)

        return best_dir, best_valid_score, best_config
