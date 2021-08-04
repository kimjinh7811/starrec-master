import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher

class Denoising_EASE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(Denoising_EASE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        # self.train_dict = dataset.train_dict
        self.reg1 = model_conf['similarity_reg']
        self.reg2 = model_conf['dissimilarity_reg']
        self.alpha = model_conf['alpha']
        self.ratio = model_conf['ratio']

        self.device = device

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        start = time()
        log_dir = logger.log_dir

        users = list(range(self.num_users))
        # P = (X^T * X + λI)^−1
        train_matrix = torch.FloatTensor(dataset.train_matrix.toarray()).to(self.device)

        X = train_matrix

        # B compute

        G = X.transpose(0, 1) @ X

        diag = list(range(G.shape[0]))
        G[diag, diag] += (self.reg1 + self.alpha)
        P = G.inverse()
        C = -self.alpha * (P @ self.b)

        #self.B = P / -torch.diag(P)
        self.B = (P / -torch.diag(P)) * (1 + torch.diag(C)) + C
        min_dim = min(*self.B.shape)
        self.B[range(min_dim), range(min_dim)] = 0

        neg_count = torch.sum((self.B < 0), axis=0)
        dissimillar_idx = torch.argsort(neg_count)



        # B_ compute
  
        D = torch.ones_like(X) - X
        D[:, dissimillar_idx] = 0 

        #D = X
        g = D.transpose(0, 1) @ D
        diag = list(range(g.shape[0]))
        g[diag, diag] += self.reg2
        p = g.inverse()


        # C = -P @ (D.T * (D-X))
        c = -p @ (D.T @ (D-X))

        # B = P * (X^T * X − diagMat(γ))
        self.b = ((p / -torch.diag(p)) * (1 + torch.diag(c))) + c
        min_dim = min(*self.b.shape)
        self.b[range(min_dim), range(min_dim)] = 0

        

        '''
        # Calculate the output matrix for prediction
        #self.output2 = X @ self.B
        #self.output = D @ self.b
        self.output = self.alpha * (X @ self.B) + (1 - self.alpha) * (D @ self.b)
        
        # Save
        torch.save(self.B, os.path.join(log_dir, 'B_best_model.p'))
        torch.save(self.b, os.path.join(log_dir, 'b_best_model.p'))
        '''

        #self.output = X @ self.B
        self.output = D @ self.b
        #self.output = X @ (self.B-self.b)
        #self.output = self.alpha*(X @ self.B) + (1-self.alpha)*(D @ self.b)
        #torch.save((self.B-self.b), os.path.join(log_dir, 'best_model.p'))
        #torch.save(self.B, os.path.join(log_dir, 'best_model.p'))
        torch.save(self.b, os.path.join(log_dir, 'best_model.p'))

        # Evaluate
        valid_score = evaluator.evaluate(self, dataset)
        valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]
        logger.info(', '.join(valid_score_str))

        total_train_time = time() - start

        return valid_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            X = torch.FloatTensor(batch_eval_pos.toarray()).to(self.device)
            D = torch.ones_like(X)- X
            eval_output = (D @ self.b).detach().cpu().numpy()
            #eval_output = (X @ self.B).detach().cpu().numpy()
            #eval_output = (self.alpha * (X@self.B) + (1-self.alpha)*(D @ self.b)).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')

        return eval_output

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            #self.B = torch.load(f)
        # with open(os.path.join(log_dir, 'B_best_model.p'), 'rb') as f:
        #     self.B = torch.load(f)
        # with open(os.path.join(log_dir, 'b_best_model.p'), 'rb') as f:
            self.b = torch.load(f)