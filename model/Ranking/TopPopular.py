import os
import math
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.linear_model import ElasticNet

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher

class TopPopular(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(TopPopular, self).__init__(dataset, model_conf)

        self.item_popularity = None

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        log_dir = logger.log_dir

        train_matrix = dataset.train_matrix.tocsc()
        num_items = train_matrix.shape[1]

        start = time()
        
        self.item_popularity = np.ediff1d(train_matrix.tocsc().indptr)

        valid_score = evaluator.evaluate(self)

        total_train_time = time() - start

        return valid_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        # eval_pos_matrix
        num_eval = eval_pos_matrix.shape[0]
        eval_output = np.array(self.item_popularity, dtype=np.float32).reshape((1, -1))
        eval_output = np.repeat(eval_output, num_eval, axis = 0)
        if eval_items is not None:
            eval_output[np.logical_not(eval_items)]=float('-inf')

        return eval_output

    def restore(self, log_dir):
        pass