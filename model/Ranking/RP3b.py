import os
import math
from time import time
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.preprocessing import normalize

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher

class RP3b(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(RP3b, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.alpha = model_conf['alpha']
        self.beta = model_conf['beta']
        self.topk = model_conf['topk']

        self.positive_only=True

        self.device = device

        self.build_graph()

    def build_graph(self):
        pass

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        log_dir = logger.log_dir

        train_matrix = dataset.train_matrix.tocsc()
        num_items = train_matrix.shape[1]
        
        Pui = normalize(train_matrix, norm='l1', axis=1)
        
        #Piu is the column-normalized, "boolean" urm transposed
        X_bool = train_matrix.transpose(copy=True)
        X_bool.data = np.ones(X_bool.data.size, np.float32)

        # Taking the degree of each item to penalize top popular
        # Some rows might be zero, make sure their degree remains zero
        X_bool_sum = np.array(X_bool.sum(axis=1)).ravel()

        degree = np.zeros(train_matrix.shape[1])
        nonZeroMask = X_bool_sum!=0.0
        degree[nonZeroMask] = np.power(X_bool_sum[nonZeroMask], -self.beta)

        Piu = normalize(X_bool, norm='l1', axis=1)
        del(X_bool)

        if self.alpha != 1:
            Pui = Pui.power(self.alpha)
            Piu = Piu.power(self.alpha)

        block_dim = 200
        # d_t = Piu

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0
        
        start = time()
        item_blocks = range(0, num_items, block_dim)
        tqdm_iterator = tqdm(item_blocks, desc='# items blocks covered', total=len(item_blocks))
        for cur_items_start_idx in tqdm_iterator:

            if cur_items_start_idx + block_dim > num_items:
                block_dim = num_items - cur_items_start_idx

            # second * third transition matrix: # of ditinct path from item to item
            # block_dim x item
            Piui = Piu[cur_items_start_idx:cur_items_start_idx + block_dim, :] * Pui
            Piui = Piui.toarray()

            for row_in_block in range(block_dim):
                # Delete self connection
                row_data = np.multiply( Piui[row_in_block, :], degree)
                row_data[cur_items_start_idx + row_in_block] = 0

                # Top-k items
                best = row_data.argsort()[::-1][:self.topk]

                # add non-zero top-k path only (efficient)
                notZerosMask = row_data[best] != 0.0

                values_to_add = row_data[best][notZerosMask]
                cols_to_add = best[notZerosMask]

                for index in range(len(values_to_add)):

                    if numCells == len(rows):
                        rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                        cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                        values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                    rows[numCells] = cur_items_start_idx + row_in_block
                    cols[numCells] = cols_to_add[index]
                    values[numCells] = values_to_add[index]

                    numCells += 1

        self.W_sparse = sp.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])), shape=(Pui.shape[1], Pui.shape[1]))


        sp.save_npz(os.path.join(log_dir, 'best_model'), self.W_sparse)

        valid_score = evaluator.evaluate(self)

        total_train_time = time() - start

        return valid_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        # eval_pos_matrix
        eval_output = (eval_pos_matrix * self.W_sparse).toarray()
        if eval_items is not None:
            eval_output[np.logical_not(eval_items)]=float('-inf')

        return eval_output

    def restore(self, log_dir):
        self.W_sparse = sp.load_npz(os.path.join(log_dir, 'best_model.npz'))