import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher

class MultDAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(MultDAE, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.enc_dims = [self.num_items] + model_conf['enc_dims']
        self.dec_dims = self.enc_dims[::-1]
        self.dims = self.enc_dims + self.dec_dims[1:]
        self.dropout = model_conf['dropout']

        self.learning_rate = model_conf['lr']
        self.reg = model_conf['reg']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        #==============================================
        self.popular_threshold = model_conf['popular_threshold']
        self.noise_target = model_conf['noise_target']
        self.pos_noise_ratio = model_conf['pos_noise_ratio']
        #==============================================

        self.eps = 1e-6
        self.device = device

        self.build_graph()

    def build_graph(self):
        self.encoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.enc_dims[:-1], self.enc_dims[1:])):
            self.encoder.append(nn.Linear(d_in, d_out))
            self.encoder.append(nn.Tanh())

        self.decoder = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dec_dims[:-1], self.dec_dims[1:])):
            self.decoder.append(nn.Linear(d_in, d_out))
            if i != len(self.dec_dims[:-1]) - 1:
                self.decoder.append(nn.Tanh())
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.reg)

        # Send model to device (cpu or gpu)
        self.to(self.device)

    def forward(self, x):        
        # encoder
        # encoder
        x = F.normalize(x)
        t = int(self.num_items * self.popular_threshold)

        pop_x, unpop_x = torch.zeros_like(x), torch.zeros_like(x)
        pop_x[:, :t], unpop_x[:, t:] = x[:, :t], x[:, t:]
        

        before_drop = torch.sum(x > 0., axis=1, keepdim=True)
        if self.noise_target == 'pop':
            pop_x = F.dropout(pop_x, p=self.pos_noise_ratio, training=self.training) * (1 - self.pos_noise_ratio)
            after_drop = torch.sum((pop_x + unpop_x) > 0., axis=1, keepdim=True)
            scale = before_drop / after_drop
            scale[after_drop == 0.] = 0
            h = (pop_x + unpop_x) * scale

        elif self.noise_target == 'unpop':
            unpop_x = F.dropout(unpop_x, p=self.pos_noise_ratio, training=self.training) * (1 - self.pos_noise_ratio)
            after_drop = torch.sum((pop_x + unpop_x) > 0., axis=1, keepdim=True)
            scale = before_drop / after_drop
            scale[after_drop == 0.] = 0
            h = (pop_x + unpop_x) * scale

        #h = F.dropout(F.normalize(x), p=self.dropout, training=self.training)
        for layer in self.encoder:
            h = layer(h)

        output = h
        for layer in self.decoder:
            output = layer(output)

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
        train_data_perm = np.arange(train_users)

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            batch_loader = DataBatcher(train_data_perm, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                batch_matrix = torch.FloatTensor(train_matrix[batch_idx].toarray()).to(self.device)

                batch_loss = self.train_model_per_batch(batch_matrix)
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

    def train_model_per_batch(self, batch_matrix):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output = self.forward(batch_matrix)

        # loss
        loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_matrix = torch.FloatTensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_matrix).detach().cpu().numpy()
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
            return eval_output
    
    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)