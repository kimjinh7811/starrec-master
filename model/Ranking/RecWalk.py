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

class RecWalk(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(RecWalk, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.item_model_path = model_conf['item_model_path']
        self.item_model = self.load_item_model()

        self.stretagy = model_conf['strategy']
        assert self.stretagy in ['k_step', 'pr']
        self.k = model_conf['k']
        self.alpha = model_conf['alpha']
        self.damping = model_conf['damping']

        self.device = device

    def load_item_model(self):
        if not os.path.exists(self.item_model_path):
            raise FileNotFoundError('Item model not found. : %s' % self.item_model_path)
        return sp.load_npz(self.item_model_path)
        # return sp.identity(self.num_items)

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        log_dir = logger.log_dir

        train_matrix = dataset.train_matrix.tocsc()
        num_items = train_matrix.shape[1]

        start = time()

        # RecWalk P: (U + I, U + I)
        self.P = self.recwalk(train_matrix, self.item_model, self.alpha)

        if self.stretagy == 'k_step':
            # Multiply P K times
            self.P_star = None
        else:
            pass

        valid_score = evaluator.evaluate(self)

        total_train_time = time() - start

        return valid_score, total_train_time

    def recwalk(self, rating_matrix, item_model, alpha=0.005):
        n = self.num_users
        m = self.num_items

        Muu = sp.eye(n)
        Mii = self.RowStochastic(item_model, strategy="dmax")
        Hui = self.RowStochastic(rating_matrix)
        Hiu = self.RowStochastic(rating_matrix.T)
        
        # H
        Ha = sp.hstack([sp.csr_matrix((n, n), dtype='float'), Hui])
        Hb = sp.hstack([Hiu, sp.csr_matrix((m, m), dtype='float')])
        H = sp.vstack([Ha, Hb])

        # M
        Ma = sp.hstack([Muu, sp.csr_matrix((n, m), dtype='float')])
        Mb = sp.hstack([sp.csr_matrix((m, n), dtype='float'), Mii])
        M = sp.vstack([Ma, Mb])
        
        # P
        P = alpha * H + (1 - alpha) * M

        return P

    def RowStochastic(self, A, strategy='standard'):
        num_rows, num_cols = A.shape
        if strategy == "dmax":
            row_sums = A.sum(1)
            dmax = row_sums.max()
            A_temp = 1 / dmax * A
            return (sp.identity(num_rows) - sp.diags(A_temp.sum(1).A1)) + A_temp
        else:
            row_sums = A.sum(1).A1
            row_sums[row_sums == 0] = 1 # replacing the zero row sums with 1
            return sp.diags(1 / row_sums) * A

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        if self.stretagy == 'k_step':
            self.P_star = self.P[user_ids, :]
            for i in range(1, self.k):
                self.P_star = self.P_star * self.P
            eval_output = self.P_star[:, self.num_users:]
            return eval_output.toarray()
        else:
            dim = self.num_users + self.num_items
            e = sp.csr_matrix((np.ones(len(user_ids)), (user_ids, user_ids)), shape=(dim, dim))
            eval_output = []
            for i, u in tqdm(enumerate(user_ids)):
                converged = False
                x_k_1 = e[i, :]
                while not converged:
                    # x_k
                    x_k = self.damping * x_k_1 * self.P + (1 - self.damping) * e[i, :]
                    
                    # normalize
                    x_k = normalize(x_k, norm='l1', axis=1)

                    # check convergence
                    if (x_k - x_k_1).sum() < 1e-6:
                        converged = True
                        eval_output.append(x_k)
                    else:
                        print('not converged')
            eval_output = sp.vstack(eval_output)
            return eval_output.toarray()[:, self.num_users:]

    def restore(self, log_dir):
        # self.W_sparse = sp.load_npz(os.path.join(log_dir, 'best_model.npz'))
        pass