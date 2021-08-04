# Import packages
import os
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp

import utils.Constant as CONSTANT
from dataloader import UIRTDatset
from evaluation import Evaluator
from utils import Config, Logger, ResultTable, make_log_dir, set_random_seed, seconds_to_hms
from experiment import EarlyStop, train_model
from param_search import GridSearch, BayesianSearch

import pickle
from utils.Statistics import Statistics # jhkim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# read configs
config = Config('main_config.cfg')

task = config.get_param('Experiment', 'task')
model_name = config.get_param('Experiment', 'model_name')

# logger
log_dir = make_log_dir(os.path.join('saves', model_name))
logger = Logger(log_dir)

# dataset
dataset_name = config.get_param('Dataset', 'dataset')
dataset_type = CONSTANT.DATASET_TO_TYPE[dataset_name]
if dataset_type == 'UIRT':
    dataset = UIRTDatset(**config['Dataset']) ###
else:
    raise ValueError('Dataset type error!')

# early stop
early_stop = EarlyStop(**config['EarlyStop'])

# Save log & dataset config.
logger.info(config)
logger.info(dataset)

if task == 'Ranking':
    import model.Ranking as ranking
    MODEL_CLASS = getattr(ranking, model_name)
# elif task == 'Rating':
#     import model.Rating as rating
#     MODEL_CLASS = getattr(rating, model_name)

# elif task == 'Session':
#     import model.Session as session
#     MODEL_CLASS = getattr(session, model_name)
else:
    raise ValueError('error')

search_config = config['ParamSearch']
param_search = search_config['param_search']
seed = config.get_param('Experiment', 'seed')




valid_matrix = dataset.valid_pos_matrix + dataset.valid_target_matrix
test_matrix = dataset.test_pos_matrix[0] + dataset.test_target_matrix[0]

train_matrix = dataset.train_matrix.toarray()
valid_matrix = valid_matrix.toarray()
test_matrix = test_matrix.toarray()

train_sh = train_matrix.shape
val_sh = valid_matrix.shape
test_sh = test_matrix.shape

data_matrix = np.concatenate((train_matrix, valid_matrix), axis=0)

item_pop = np.sum(data_matrix, axis=0, dtype=np.int32)
num_users = data_matrix.shape[0]


if __name__ == '__main__':
    # =============================================== Parameter Search
    if param_search:
        val_eval_pos, val_eval_target, eval_neg_candidates = dataset.valid_data()
        valid_evaluator = Evaluator(val_eval_pos, val_eval_target, eval_neg_candidates, dataset.split_type, dataset.item_popularity, dataset.num_users,  **config['Evaluator'], novelty=False)

        search_method = search_config['search_method']
        num_parallel = search_config['num_parallel']
        if search_method == 'grid':
            param_opt = GridSearch(MODEL_CLASS, dataset, valid_evaluator, early_stop, logger, config, device, seed=seed, num_parallel=num_parallel)
        elif search_method == 'bayesian':
            num_trials = search_config['num_trials']
            param_opt = BayesianSearch(MODEL_CLASS, dataset, valid_evaluator, early_stop, logger, config, device, seed=seed, num_trials=num_trials, num_parallel=num_parallel)
        else:
            raise NotImplementedError('?')
        
        # Search the optimal set of hyper-parameter
        search_result_table = param_opt.search()

        # search_result_table.show()
        logger.info(search_result_table.to_string())
        best_dir, best_valid_score, best_config = param_opt.best_result()
        best_model = MODEL_CLASS(dataset, best_config['Model'], device)
        
        # restore best parameters
        best_model.restore(best_dir)

        # evaluate model
        best_model.eval()

        # jhkim
        if config['Dataset']['split_type'] == 'hold-out-user-5-fold-test':
            CV_evaluator = {metric: Statistics('%s' % (metric)) for metric in best_valid_score}

            for i in range(5):
                test_eval_pos, test_eval_target, eval_neg_candidates = dataset.test_data(i)
                test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, dataset.split_type,
                                           dataset.item_popularity, dataset.num_users,  **config['Evaluator'], novelty=False) ######
                fold_test_score = test_evaluator.evaluate(best_model)

                for score in fold_test_score:
                    CV_evaluator[score].update(fold_test_score[score])
            test_score = {metric: CV_evaluator[metric].mean for metric in CV_evaluator}

            evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
            evaluation_table.add_row('Best valid', best_valid_score)
            evaluation_table.add_row('Test', test_score)

            logger.info(evaluation_table.to_string())

        else:
            test_eval_pos, test_eval_target, eval_neg_candidates = dataset.test_data()
            test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, dataset.split_type,
                                           dataset.item_popularity, dataset.num_users, **config['Evaluator'])
            test_score = test_evaluator.evaluate(best_model)

            evaluation_table = ResultTable(table_name='Result from Best Param', header=list(test_score.keys()))
            evaluation_table.add_row('Best valid', best_valid_score)
            evaluation_table.add_row('Test', test_score)

            logger.info(evaluation_table.to_string())

    # =============================================== Single Experiment
    else:
        grid_configs = config['GridSearch']
        grid_keys = list(grid_configs.keys())
        # seed = config.get_param('Experiment', 'seed') ##

        vals = []

        ####################################################################################
        # build model
        # set_random_seed(seed)
        model = MODEL_CLASS(dataset, config.model_config['Model'], device)

        # train model
        val_eval_pos, val_eval_target, eval_neg_candidates = dataset.valid_data()
        valid_evaluator = Evaluator(val_eval_pos, val_eval_target, eval_neg_candidates, dataset.split_type, item_pop, num_users,  **config['Evaluator'], novelty=False)

        valid_score, train_time = train_model(model, dataset, valid_evaluator, early_stop, logger, config) ### jwlee 봐야할 것

        valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

        logger.info('Train done: %s' % seconds_to_hms(train_time))
        logger.info(', '.join(valid_score_str))

        # restore best parameters
        #model.restore('saves/DAE/4_20210613-2142')
        #model.restore('saves/Dropout_DAE/121_20210613-2142')
        model.restore(log_dir)

        # evaluate model
        model.eval()

        # jhkim #####
        if config['Dataset']['split_type'] == 'hold-out-user-5-fold-test':
            CV_evaluator = {metric: Statistics('%s' % (metric)) for metric in valid_score}

            prefix_dir = "./diversity_evaluation" # jhkim, jwlee
            data_set_name = config['Dataset']['dataset']
            # store user-item matrix 
            if not os.path.exists(prefix_dir+'/'+data_set_name):
                os.makedirs(prefix_dir+'/'+data_set_name)
                with open(prefix_dir+'/'+data_set_name+'/'+'user_item_matrix', 'wb') as f:
                    pickle.dump( dataset.train_matrix, f) ##

            for i in range(5):
                test_eval_pos, test_eval_target, eval_neg_candidates = dataset.test_data(i)

                # 전체 user-item matrix로 novelty 측정
                test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, dataset.split_type,
                                        item_pop, num_users,  **config['Evaluator'], novelty=False)

                # train matrix로 novelty 측정
                # test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, dataset.split_type,
                #                            np.sum(dataset.train_matrix.toarray(),axis=0), dataset.train_matrix.shape[0],  **config['Evaluator'], novelty=True)

                #fold_test_score = test_evaluator.evaluate_test(model, data_set_name=config['Dataset']['dataset'], model_name=config['Experiment']['model_name'], test_fold=i) ######
                fold_test_score = test_evaluator.evaluate(model) ######

                for score in fold_test_score:
                    CV_evaluator[score].update(fold_test_score[score])
            test_score = {metric: CV_evaluator[metric].mean for metric in CV_evaluator}
            evaluation_table = ResultTable(table_name='Best Result', header=list(test_score.keys()))
            evaluation_table.add_row('Best valid', valid_score)
            evaluation_table.add_row('Test', test_score)

            logger.info(evaluation_table.to_string())

            # ################
            #f = open("/home/jwlee/cikm2020/item_pop_1m.txt", 'w')
            #for ssss in item_pop:
            #    f.write(str(ssss)+'\n')
            #f.close()

            # ############

        else:
            test_eval_pos, test_eval_target, eval_neg_candidates = dataset.test_data()
            test_evaluator = Evaluator(test_eval_pos, test_eval_target, eval_neg_candidates, dataset.split_type,
                                        dataset.item_popularity, dataset.num_users,  **config['Evaluator'])


            #test_score = test_evaluator.evaluate_test(model, data_set_name=config['Dataset']['dataset'], model_name=config['Experiment']['model_name'], test_fold=1) ######
            test_score = test_evaluator.evaluate(model) ######

            evaluation_table = ResultTable(table_name='Result from Best Param', header=list(test_score.keys()))
            evaluation_table.add_row('Best valid', valid_score)
            evaluation_table.add_row('Test', test_score)

            logger.info(evaluation_table.to_string())
