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

class CDAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(CDAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.hidden_dim = model_conf['hidden_dim']
        self.dropout = model_conf['dropout']
        self.reg = model_conf['reg']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        
        self.lr = model_conf['lr']
        self.lr_decay = model_conf['lr_decay']
        self.lr_decay_from = model_conf['lr_decay_from']
        self.lr_decay_step = model_conf['lr_decay_step']
        self.lr_decay_rate = model_conf['lr_decay_rate']

        self.device = device

        self.build_graph()

    def build_graph(self):
        # Variable
        self.enc_w = nn.Parameter(torch.ones(self.num_items, self.hidden_dim))
        self.enc_b = nn.Parameter(torch.ones(self.hidden_dim))
        nn.init.normal_(self.enc_w, 0, 0.01)
        nn.init.normal_(self.enc_b, 0, 0.01)

        self.dec_w = nn.Parameter(torch.ones(self.hidden_dim, self.num_items))
        self.dec_b = nn.Parameter(torch.ones(self.num_items))
        nn.init.normal_(self.dec_w, 0, 0.01)
        nn.init.normal_(self.dec_b, 0, 0.01)

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)

        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.reg)

        if self.lr_decay:
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.lr_decay_step, self.lr_decay_rate)
        else:
            self.scheduler = None

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, u, x):        
        denoised_x = F.dropout(F.normalize(x), self.dropout, training=self.training)
        enc = torch.tanh(denoised_x @ self.enc_w + self.enc_b + self.user_embedding(u))
        output = torch.sigmoid(enc @ self.dec_w + self.dec_b)

        return output

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir
        
        train_matrix = dataset.train_matrix
        train_users = train_matrix.shape[0]
        users = np.arange(train_users)

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)
                batch_idx = torch.LongTensor(batch_idx).to(self.device)

                batch_loss = self.train_model_per_batch(batch_idx, batch_matrix)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()
                
                valid_score = evaluator.evaluate(self, dataset)
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

    def train_model_per_batch(self, batch_user, batch_matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output = self.forward(batch_user, batch_matrix)

        # loss
        loss = F.binary_cross_entropy(output, batch_matrix, reduction='none').sum(1).mean()

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        with torch.no_grad():
            eval_user_ids = torch.LongTensor(user_ids).to(self.device)
            eval_input = torch.FloatTensor(eval_pos_matrix.toarray()).to(self.device)
            eval_output = self.forward(eval_user_ids, eval_input).detach().cpu().numpy()
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
   
            return eval_output
    
    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)