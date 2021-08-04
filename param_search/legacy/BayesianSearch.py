import os
import copy
from time import time

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from utils import Logger, set_random_seed
from experiment import train_model

class BayesianSearch:
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

        self.search_params = config['BayesSearch']
        self.search_space = self.generate_search_space()
        self.search_param_names = [p.name for p in self.search_space]

        self._set_skopt_params(random_state=self.seed)

        # (score, [list of params])
        self.result = None
        self.valid_score = []
        self.exp_num = 0

        self.best_exp_num = 0
        self.best_score = -1
        self.best_param = None

        self.device = device

    def _set_skopt_params(self, n_calls = 10,
                          n_random_starts = 5,
                          n_points = 10000,
                          n_jobs = 1,
                          noise = 1e-5,
                          acq_func = 'gp_hedge',
                          acq_optimizer = 'auto',
                          random_state = None,
                          verbose = False,
                          n_restarts_optimizer = 10,
                          xi = 0.01,
                          kappa = 1.96,
                          x0 = None,
                          y0 = None):
        """
        wrapper to change the params of the bayesian optimizator.
        for further details:
        https://scikit-optimize.github.io/#skopt.gp_minimize

        """
        self.n_point = n_points
        self.n_calls = n_calls
        self.n_random_starts = n_random_starts
        self.n_jobs = n_jobs
        self.acq_func = acq_func
        self.acq_optimizer = acq_optimizer
        self.random_state = random_state
        self.n_restarts_optimizer = n_restarts_optimizer
        self.verbose = verbose
        self.xi = xi
        self.kappa = kappa
        self.noise = noise
        self.x0 = x0
        self.y0 = y0

    def generate_search_space(self):
        search_space = []

        for param_name in self.search_params:
            space = self.search_params[param_name]

            if not isinstance(self.search_params[param_name], list):
                self.config[param_name] = self.search_params[param_name]

            if isinstance(space[0], int):
                if len(space) != 2:
                    raise ValueError
                param_dim = Integer(low=min(space), high=max(space), name=param_name)
            elif isinstance(space[0], float):
                if len(space) != 2:
                    raise ValueError
                param_dim = Real(low=min(space), high=max(space), name=param_name)
            elif isinstance(space[0], str):
                param_dim = Categorical(categories=space, name=param_name)
            else:
                raise TypeError
            search_space.append(param_dim)
        return search_space

    def optimize(self, model, early_stop, logger, config):
        valid_score, train_time = train_model(model, self.dataset, self.evaluator, early_stop, logger, config)
        self.valid_score.append(valid_score)

        score = valid_score[self.metric_to_optimize]
        
        valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]
        logger.info(', '.join(valid_score_str))

        return -score

    def objective_function(self, cur_space):
        # update config
        config_copy = copy.deepcopy(self.config)

        config_to_update = dict(zip(self.search_param_names, cur_space))
        config_copy.update_params(config_to_update)
        
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

        # Callback
        _callback = _Callback(self.n_calls)

        # search over space
        res = gp_minimize(self.objective_function,
                                  self.search_space,
                                  base_estimator=None,
                                  n_calls=self.n_calls,
                                  n_random_starts=self.n_random_starts,
                                  acq_func=self.acq_func,
                                  acq_optimizer=self.acq_optimizer,
                                  x0=self.x0,
                                  y0=self.y0,
                                  random_state=self.random_state,
                                  verbose=self.verbose,
                                  callback=_callback,
                                  n_points=self.n_point,
                                  n_restarts_optimizer=self.n_restarts_optimizer,
                                  xi=self.xi,
                                  kappa=self.kappa,
                                  noise=self.noise,
                                  n_jobs=self.n_jobs)

        res_list = list(zip(res.func_vals, res.x_iters))
        self.result = [(-s, x) for s, x in res_list]
        for i, (score, param) in enumerate(self.result):
            if score > self.best_score:
                self.best_exp_num = i
                self.best_score = score
                self.best_param = param

    def best_result(self):
        best_dir = self.exp_logger[self.best_exp_num].log_dir
        best_valid_score = self.valid_score[self.best_exp_num]

        best_config = copy.deepcopy(self.config)
        best_params_to_update = {k: v for k, v in zip(self.search_param_names, self.best_param)}
        best_config.update_params(best_params_to_update)

        return best_dir, best_valid_score, best_config

# almost copy from ksopt
class _Callback:
    """
    Callback to control the verbosity.

    Parameters
    ----------
    * `n_init` [int, optional]:
        Number of points provided by the user which are yet to be
        evaluated. This is equal to `len(x0)` when `y0` is None

    * `n_random` [int, optional]:
        Number of points randomly chosen.

    * `n_total` [int]:
        Total number of func calls.

    Attributes
    ----------
    * `iter_no`: [int]:
        Number of iterations of the optimization routine.
    """

    def __init__(self, n_total):
        self.n_total = n_total
        self.iter_no = 1

        self._start_time = time()
        self._cur_start_time = None

    def __call__(self, res):
        """
        Parameters
        ----------
        * `res` [`OptimizeResult`, scipy object]:
            The optimization as a OptimizeResult object.
        """
        time_taken = time() - self._start_time

        curr_param = res.x_iters[-1]
        curr_y = res.func_vals[-1]
        curr_min = res.fun

        print('[SEARCH %2d] Param: %s, cur value: %.4f, cur min: %.4f [%.4f sec]' % (self.iter_no, curr_param, curr_y, curr_min, time_taken))
        # print("Time taken: %0.4f" % time_taken)
        # print("Function value obtained: %0.4f" % curr_y)
        # print("Current minimum: %0.4f" % curr_min)

        self.iter_no += 1
        # if self.iter_no <= self.n_total:
        self._start_time = time()