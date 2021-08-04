import copy
from time import time

from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

# https://colab.research.google.com/github/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb#scrollTo=6Pi9ebpTB9em

def train_model(model, dataset, evaluator, early_stop, logger, config):
    # train model with the given experimental setting
    # input: sess, initialized model, dataset, evaluator, early_stop, logger, config
    # output: optimized model, best_valid_score

    early_stop.initialize()
    valid_score, train_time = model.train_model(dataset, evaluator, early_stop, logger, config)

    return valid_score, train_time

def grid_search_space(grid_conf):
    search_params = list(grid_conf.keys())
    search_space = []
    for i, param_name in enumerate(search_params):
        new_space = []
        for param_value in grid_conf[param_name]:
            if i == 0:
                new_space.append({param_name: param_value})
            else:
                tmp_list = copy.deepcopy(search_space)
                for param_setting in tmp_list:
                    param_setting[param_name] = param_value
                new_space += tmp_list
        search_space = new_space

    return search_space

def gp_search_space(grid_conf):
    search_space = []

    for param_name in grid_conf:
        space = grid_conf[param_name]

        if isinstance(space[0], int):
            if len(space) != 2:
                raise ValueError
            param_dim = Integer(low=space[0], high=space[1], name=param_name)
        elif isinstance(space[0], float):
            if len(space) != 2:
                raise ValueError
            param_dim = Real(low=space[0], high=space[1], name=param_name)
        elif isinstance(space[0], str):
            param_dim = Categorical(categories=space, name=param_name)
        else:
            raise TypeError
        search_space.append(param_dim)
    return search_space