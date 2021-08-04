import os
import math
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from module.Sampler import generate_pairwise_data_from_matrix

class NGCF(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(NGCF, self).__init__(dataset, model_conf)
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items

        self.emb_dim = model_conf['hidden_dim']
        self.layer_dims = model_conf['layer_dims']
        self.num_layers = len(self.layer_dims)
        self.mess_dropout = nn.Dropout(p=model_conf['mess_dropout'])

        self.num_negatives = model_conf['num_negatives']
        self.adj_type = 'norm'
        self.n_fold = 2
        self.norm_adj = self.get_adj_mat()

        self.learning_rate = model_conf['learning_rate']
        self.reg = model_conf['reg']
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        
        self.device = device
        self.best_params = None

        self.build_graph()

    def build_graph(self):
        # Variable
        self.user_embedding = nn.Parameter(torch.ones(self.num_users, self.emb_dim))
        self.item_embedding = nn.Parameter(torch.ones(self.num_items, self.emb_dim))
        nn.init.normal_(self.user_embedding, 0, 0.01)
        nn.init.normal_(self.item_embedding, 0, 0.01)

        # Adjecent matrix

        # Layer weights
        self.gc_linear_layers = nn.ModuleList()
        self.bi_linear_layers = nn.ModuleList()
        dims = [self.emb_dim] + self.layer_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.gc_linear_layers.append(nn.Linear(in_dim, out_dim))
            self.bi_linear_layers.append(nn.Linear(in_dim, out_dim))
        
        # optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.to(self.device)

    def split_A_into_fold(self, A):
        return [A]

    def ngcf_embedding(self,):
        # Node dropout
        # A = torch.ones(self.num_users, self.num_items).to(self.device)
        # A_fold_hat = self.split_A_into_fold(A)
        A_fold_hat = self._split_A_hat(self.norm_adj)

        # ego embedding
        ego_embeddings = torch.cat([self.user_embedding, self.item_embedding], 0)
        all_embeddings = []

        for i, (gc_linear, bi_linear) in enumerate(zip(self.gc_linear_layers, self.bi_linear_layers)):
            # Aggregate messages
            tmp_emb = []
            for A_fold in A_fold_hat:
                tmp_emb.append(A_fold @ ego_embeddings)
            
            side_embeddings = torch.cat(tmp_emb, 0)

            # Side -> sum embeddings
            sum_embeddings = F.leaky_relu(gc_linear(side_embeddings))

            # bi embeddings
            bi_embeddings = F.leaky_relu(bi_linear(torch.mul(ego_embeddings, side_embeddings)))

            # ego embeddings
            ego_embeddings = sum_embeddings + bi_embeddings

            ego_embeddings = self.mess_dropout(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            
            all_embeddings += [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, dim=-1)
        user_embedding, item_embedding = all_embeddings.split_with_sizes((self.num_users, self.num_items), 0)
        return user_embedding, item_embedding

    def forward(self, user, pos, neg=None):
        # Embedding Lookup

        # NGCF Embedding
        u_embedding, i_embedding = self.ngcf_embedding()

        user_latent = F.embedding(user, u_embedding)
        pos_latent = F.embedding(pos, i_embedding)
        pos_score = torch.mul(user_latent, pos_latent).sum(1)
        if neg is not None:
            neg_latent = F.embedding(neg, i_embedding)
            neg_score = torch.mul(user_latent, neg_latent).sum(1)
            return pos_score, neg_score
        else:
            return pos_score

    def predict_at_once(self, users):
        # NGCF Embedding
        u_embedding, i_embedding = self.ngcf_embedding()

        user_latent = F.embedding(users, u_embedding)
        item_latent = i_embedding.T  # emb, item
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
        start = time.time()
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            user_ids, pos_ids, neg_ids = generate_pairwise_data_from_matrix(train_matrix, num_negatives=self.num_negatives)
            num_training = len(user_ids)

            # train_data_perm = np.random.permutation(num_training)
            train_data_perm = torch.randperm(num_training)
            batch_loader = DataBatcher(train_data_perm, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time.time()
            for b, batch_idx in enumerate(batch_loader):
                batch_user, batch_pos, batch_neg = user_ids[batch_idx].to(self.device), pos_ids[batch_idx].to(self.device), neg_ids[batch_idx].to(self.device)

                batch_loss = self.train_model_per_batch(batch_user, batch_pos, batch_neg)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time.time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                # evaluate model
                epoch_eval_start = time.time()
                # valid_score = evaluate_model(sess, model, dataset, evaluator)
                valid_score = evaluator.evaluate(self)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                elif updated:
                    torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))

                epoch_eval_time = time.time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time.time() - start

        return early_stop.best_score, total_train_time

    def train_model_per_batch(self, batch_user, batch_item, batch_neg):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        pos_output = self.forward(batch_user, batch_item)
        neg_output = self.forward(batch_user, batch_neg)
        output = pos_output - neg_output

        # loss
        loss = - torch.sigmoid(output).log().sum()

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
        eval_output = torch.zeros(eval_pos_matrix.shape)
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
            eval_output[np.logical_not(eval_items)] = float('-inf')

        return eval_output
    
    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)


    def get_adj_mat(self):
        A = sp.lil_matrix((self.num_users + self.num_items, self.num_users + self.num_items), dtype=np.float32)
        A[:self.num_users, self.num_users:] = self.dataset.train_matrix
        A[self.num_users:, :self.num_users] = self.dataset.train_matrix.T
        A = A.todok()
        if self.adj_type == 'plain':
            adj_mat = A
            print('use the plain adjacency matrix')
        elif self.adj_type == 'norm':  
            adj_mat = self.normalized_adj_single(A + sp.eye(A.shape[0]))
            print('use the normalized adjacency matrix')
        elif self.adj_type == 'gcmc':
            adj_mat = self.normalized_adj_single(A)
            print('use the gcmc adjacency matrix')
        else:
            adj_mat = self.normalized_adj_single(A) + sp.eye(A.shape[0])
            print('use the mean adjacency matrix')
    
        return adj_mat.tocsr()

    def normalized_adj_single(self,adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.num_users + self.num_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold -1:
                end = self.num_users + self.num_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        # indices = np.mat([coo.row, coo.col]).transpose()
        indices = torch.LongTensor([coo.row, coo.col])
        values = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(indices, values, coo.shape).to(self.device)
        # return sp.csr_matrix((coo.data, (coo.row, coo.col)),coo.shape)