import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from module.Sampler import generate_pairwise_data_from_matrix

class BPRMF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(BPRMF, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.hidden_dim = model_conf['hidden_dim']
        self.num_negatives = model_conf['num_negatives']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.learning_rate = model_conf['learning_rate']
        self.reg = model_conf['reg']

        self.device = device

        self.build_graph()

    def build_graph(self):
        # Variable
        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.hidden_dim)
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.to(self.device)

    def forward(self, users, items):
        user_latent = self.user_embedding(users)
        item_latent = self.item_embedding(items)
        return torch.mul(user_latent, item_latent).sum(1)

    def predict_one_user(self, user):
        user_latent = self.user_embedding(user) # 1, emb
        item_latent = self.item_embedding.weight.T  # emb, item
        return torch.mm(user_latent, item_latent).squeeze()

    def predict_at_once(self, users):
        user_latent = self.user_embedding(users) # 1, emb
        item_latent = self.item_embedding.weight.T  # emb, item
        return torch.mm(user_latent, item_latent)

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        train_matrix = dataset.train_matrix

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            user_ids, pos_ids, neg_ids = generate_pairwise_data_from_matrix(train_matrix, num_negatives=self.num_negatives)
            num_training = len(user_ids)

            # train_data_perm = np.random.permutation(num_training)
            train_data_perm = torch.randperm(num_training)
            batch_loader = DataBatcher(train_data_perm, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_user, batch_pos, batch_neg = user_ids[batch_idx].to(self.device), pos_ids[batch_idx].to(self.device), neg_ids[batch_idx].to(self.device)

                batch_loss = self.train_model_per_batch(batch_user, batch_pos, batch_neg)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                # evaluate model
                epoch_eval_start = time()
                # valid_score = evaluate_model(sess, model, dataset, evaluator)
                valid_score = evaluator.evaluate(self)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                elif updated:
                    torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time

    def train_model_per_batch(self, batch_user, batch_item, batch_neg):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        pos_output = self.forward(batch_user, batch_item)
        neg_output = self.forward(batch_user, batch_neg)
        output = pos_output - neg_output

        # loss
        loss = - torch.sigmoid(output).log().mean()

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        ''' TODO
        1. Should handle when eval_items is not None
        2. Should handle when num_items is too large to inference at once
        3. Eval only negative items
        '''
        num_users, num_items = eval_pos_matrix.shape
        with torch.no_grad():

            # batch_size = self.test_batch_size
            # for i, user in enumerate(user_ids):
            #     # users = [user] * num_items
            #     items = list(range(num_items))
            #     # user_tensor, item_tensor = torch.LongTensor(users).to(self.device), torch.LongTensor(items).to(self.device)
            #     # eval_output[i, items] = self.forward(user_tensor, item_tensor).detach().cpu()

            #     user = torch.LongTensor([user]).to(self.device)
            #     eval_output[i, items] = self.predict_one_user(user).detach().cpu()
            # return eval_output.numpy()

            users = torch.LongTensor(user_ids).to(self.device)
            eval_output = self.predict_at_once(users).detach().cpu().numpy()

            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')

            return eval_output.numpy()
    
    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)
