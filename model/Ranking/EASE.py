import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher

import pickle

class EASE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(EASE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        # self.train_dict = dataset.train_dict
        self.reg = model_conf['reg']
        self.alpha = model_conf['alpha']

        self.device = device

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        start = time()
        log_dir = logger.log_dir

        users = list(range(self.num_users))
        # P = (X^T * X + λI)^−1
        train_matrix = torch.FloatTensor(dataset.train_matrix.toarray()).to(self.device)


        with open('data/'+self.dataset.data_name+'/item.side', 'rb') as f:
            side_inform_file = pickle.load(f)
            item_side_mat = side_inform_file['item_side_mat']

        side_information_matrix = torch.FloatTensor(item_side_mat.toarray()).to(self.device)


        X = train_matrix
        F = side_information_matrix



        G = X.transpose(0, 1) @ X + self.alpha * (F @ F.transpose(0, 1)) 

        diag = list(range(G.shape[0]))
        G[diag, diag] += self.reg
        P = G.inverse()

        # B = P * (X^T * X − diagMat(γ))
        self.enc_w = P / -torch.diag(P)
        min_dim = min(*self.enc_w.shape)
        self.enc_w[range(min_dim), range(min_dim)] = 0

        

        B = self.enc_w.detach().cpu().numpy()
        B_T = self.enc_w.T.detach().cpu().numpy()

        # Calculate the output matrix for prediction
        self.output = train_matrix @ self.enc_w

        # Save
        torch.save(self.enc_w, os.path.join(log_dir, 'best_model.p'))

        # Evaluate
        valid_score = evaluator.evaluate(self, dataset)
        valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]
        logger.info(', '.join(valid_score_str))

        total_train_time = time() - start

        return valid_score, total_train_time

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            input_matrix = torch.FloatTensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = (input_matrix @ self.enc_w).detach().cpu().numpy()
        
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')

        return eval_output

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            self.enc_w = torch.load(f)