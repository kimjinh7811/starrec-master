import os
import copy

from experiment import train_model
from utils import Logger, set_random_seed

class GridSearch:
    def __init__(self, model_base, dataset, evaluator, early_stop, logger, config, device):
        self.model_base = model_base
        self.dataset = dataset
        self.evaluator = evaluator
        self.early_stop = early_stop
        self.base_logger = logger
        self.base_dir = logger.log_dir
        self.metric_to_optimize = early_stop.early_stop_measure
        self.config = config
        self.seed = self.config.get_param('Experiment', 'seed')

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
        self.best_param = None

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
        self.valid_score.append(valid_score)

        score = valid_score[self.metric_to_optimize]

        valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]
        logger.info(', '.join(valid_score_str))

        return score

    def objective_function(self, cur_space):
        # update config
        config_copy = copy.deepcopy(self.config)
        config_copy.update_params(cur_space)

        set_random_seed(self.seed)

        # create model
        model = self.model_base(self.dataset, config_copy['Model'], self.device)

        exp_logger = self.init_search_logger()

        # score = -NDCG
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
        # check validity of search space

        # initialize attributes

        # search over space
        self.result = []

        for cur_space in self.search_space:
            score = self.objective_function(cur_space)
            self.result.append((score, list(cur_space.values())))

        # find best parameters
        for i, (score, param) in enumerate(self.result):
            if score > self.best_score:
                self.best_exp_num = i
                self.best_score = score
                self.best_param = param

    def best_result(self):
        best_dir = self.exp_logger[self.best_exp_num].log_dir
        best_valid_score = self.valid_score[self.best_exp_num]

        best_config = copy.deepcopy(self.config)
        best_config.update_params(self.search_space[self.best_exp_num])

        return best_dir, best_valid_score, best_config
