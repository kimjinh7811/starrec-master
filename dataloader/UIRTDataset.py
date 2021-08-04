import os
import numpy as np
import torch
import scipy.sparse as sp
import copy

from base import BaseDataset
from dataloader.DataLoader import load_data_and_info

class UIRTDatset(BaseDataset):
    def __init__(self, data_dir, dataset, min_user_per_item=1, min_item_per_user=1, implicit=True, binarize_threshold=1.0, split_type="loo", split_random=True,
                                test_ratio=0.8, valid_ratio=0.1, leave_k=5, holdout_users=100, eval_neg_num=0, popularity_order=True):
        super(UIRTDatset, self).__init__(data_dir, dataset, min_user_per_item, min_item_per_user, implicit, binarize_threshold, split_type, split_random,
                                                test_ratio, valid_ratio, leave_k, holdout_users, popularity_order)

        if self.split_type == 'hold-user-out' or self.split_type =='hold-out-user-5-fold-test': # jhkim, jwlee
            self.train_matrix, self.valid_pos_matrix, self.valid_target_matrix,  self.test_pos_matrix, self.test_target_matrix, self.user_id_dict, self.user_to_num_items, self.item_id_dict, self.item_to_num_users \
                = load_data_and_info(self.data_file, self.info_file, self.split_type)
        else:
            self.train_matrix, self.valid_matrix, self.test_matrix, self.user_id_dict, self.user_to_num_items, self.item_id_dict, self.item_to_num_users \
                = load_data_and_info(self.data_file, self.info_file, self.split_type)
        
        self.num_users = len(self.user_id_dict)
        self.num_items = len(self.item_id_dict)
        self.split_type = split_type
        
        self.item_popularity = np.fromiter(self.item_to_num_users.values(), dtype=int) # jhkim, jwlee 추후 self.get_item_popularity() 랑 합치는 게 좋을 듯

        self.eval_neg_num = eval_neg_num
        if self.split_type == 'loo' and self.eval_neg_num > 0:
            self.eval_neg_candidates = self.load_eval_negatives()
        else:
            self.eval_neg_candidates = None


        # print('Generating negative items...')
        # self.neg_dict = self.generate_neg_items_user() if split_type == 'hold-user-out' else self.generate_neg_items()

        # self.eval_mode = 'valid'
        # self.eval_candidates = None
        # self.eval_target = None
        # self.eval_input_dict = None
        # self.output_mask = None

    def save_eval_negatives(self, savepath):
        full_matrix = self.train_matrix + self.valid_matrix + self.test_matrix
        eval_negatives_txt = []
        for u in range(self.num_users):
            items_u = full_matrix.indices[full_matrix.indptr[u]: full_matrix.indptr[u+1]]
            num_all_negatives = self.num_items - len(items_u)

            if len(items_u) <= 0:
                continue
            
            prob_u = np.full(self.num_items, 1 / num_all_negatives)
            prob_u[items_u] = 0

            neg_items = np.random.choice(self.num_items, self.eval_neg_num, replace=False, p=prob_u)

            eval_negatives_txt.append('\t'.join([str(u)] + [str(i) for i in neg_items.tolist()]))
        
        with open(savepath, 'wt') as f:
            f.write('\n'.join(eval_negatives_txt))
    
    def load_eval_negatives(self):
        eval_negative_file = os.path.join(self.file_prefix + '.eval_neg_%d' % self.eval_neg_num)
        if not os.path.exists(eval_negative_file):
            self.save_eval_negatives(eval_negative_file)
        
        rows = []
        cols = []
        vals = []
        eval_negatives = {u: [] for u in range(self.num_users)}
        with open(eval_negative_file, 'rt') as f:
            lines = f.readlines()
        for line in lines:
            eval_neg_u = line.strip().split('\t')
            u = int(eval_neg_u[0])
            neg_items = [int(x) for x in eval_neg_u[1:]]
            
            assert len(neg_items) == self.eval_neg_num, "User %d: # eval negatives in file (%d) not match to 'eval_num_negatives' (%d)" % (u, len(neg_items), self.eval_neg_num)

            eval_negatives[u] += neg_items
        # eval_negatives = sp.csr_matrix((vals, (rows, cols)), shape=(self.num_users, self.num_items))
        return eval_negatives

    def sparse_to_dict(self, sparse_matrix):
        ret_dict = {}
        num_users = sparse_matrix.shape[0]
        for u in range(num_users):
            items_u = sparse_matrix.indices[sparse_matrix.indptr[u]: sparse_matrix.indptr[u+1]]
            ret_dict[u] = items_u.tolist()
        return ret_dict

    def valid_data(self):
        if self.split_type == 'hold-user-out' or self.split_type =='hold-out-user-5-fold-test': # jhkim, jwlee
            eval_pos = self.valid_pos_matrix
            eval_target = self.sparse_to_dict(self.valid_target_matrix)
        else:
            eval_pos = self.train_matrix
            eval_target = self.sparse_to_dict(self.valid_matrix)
            # eval_target = self.valid_matrix
        return eval_pos, eval_target, self.eval_neg_candidates
    
    def test_data(self, test_fold_idx=None): # jhkim, jwlee
        if self.split_type == 'hold-user-out':
            eval_pos = self.test_pos_matrix
            
            '''
            temp = self.test_target_matrix[:, :int(0.05 * self.num_items)]
            # only unpopular
            temp_target_matrix = self.test_target_matrix[:, int(0.05 * self.num_items):]
            # only popular
            #temp_target_matrix = self.test_target_matrix[:, :int(0.05 * self.num_items)]

            non_empty_idx = []
            empty_idx = []
            for i, row in enumerate(temp_target_matrix):
                if np.sum(row) > 0.:
                    non_empty_idx.append(i)
                else:
                    empty_idx.append(i)
            non_empty_idx = np.array(non_empty_idx)

            print('popular_target_matrix_sum:', np.sum(temp))
            print('unpopular_target_matrix_sum:', np.sum(self.test_target_matrix))

            temp_pos_matrix = self.test_pos_matrix[non_empty_idx, :]
            temp_target_matrix = temp_target_matrix[non_empty_idx, :]
            eval_pos = temp_pos_matrix 
            
            eval_target = self.sparse_to_dict(temp_target_matrix)
            '''

            eval_target = self.sparse_to_dict(self.test_target_matrix)

        elif self.split_type == 'hold-out-user-5-fold-test': # jhkim, jwlee
            assert 0 <= test_fold_idx and test_fold_idx < len(self.test_pos_matrix)
            eval_pos = self.test_pos_matrix[test_fold_idx]
            eval_target = self.sparse_to_dict(self.test_target_matrix[test_fold_idx])
        else:
            eval_pos = self.train_matrix + self.valid_matrix
            eval_target = self.sparse_to_dict(self.test_matrix)
        return eval_pos, eval_target, self.eval_neg_candidates

    def __str__(self):
        ret_str ='\n'
        ret_str += 'Dataset: %s\n' % self.data_name
        ret_str += '# of users: %d\n' % self.num_users
        ret_str += '# of items: %d\n' % self.num_items     
        ret_str += 'Split type: %s\n' % self.split_type
        if self.split_type == 'holdout':
            train = 1 - self.test_ratio
            val =  train * self.valid_ratio
            train -= val
            
            ret_str += 'Split random: %s\n' % self.split_random
            ret_str += 'train / val / test: %.1f, %.1f, %.1f\n' % (train, val, self.test_ratio)
        ret_str += '# of eval negatiaves: %s\n' % 'all' if self.eval_neg_num < 1 else str(self.eval_neg_num)
        
        return ret_str

    #def convert_to_sparse_matrix(self):
    def get_item_popularity(self, test_fold_idx=None): # jhkim, jwlee
        #matrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        item_popularity = np.zeros(self.num_items)

        for user_id, rating_history in self.train_dict.items():
            for (item_id, rating) in rating_history:
                #matrix[user_id, item_id] = rating
                item_popularity[item_id] += 1

        if self.split_type == 'hold-user-out' or self.split_type == 'hold-out-user-5-fold-test':  # jhkim, jwlee 추후 수정해야 할 듯 (원본 코드도 수정 안 된 듯)
            for user_id, rating_history in self.valid_dict.items():
                for (item_id, rating) in rating_history:
                    #matrix[user_id, item_id] = rating
                    item_popularity[item_id] += 1

            for user_id, rating_history in self.valid_target_dict.items():
                for (item_id, rating) in rating_history:
                    #matrix[user_id, item_id] = rating
                    item_popularity[item_id] += 1

            for user_id, rating_history in self.test_input_dict.items():
                for (item_id, rating) in rating_history:
                    #matrix[user_id, item_id] = rating
                    item_popularity[item_id] += 1

            for user_id, rating_history in self.test_target_dict.items():
                for (item_id, rating) in rating_history:
                    #matrix[user_id, item_id] = rating
                    item_popularity[item_id] += 1
        else:
            for user_id, rating_history in self.valid_dict.items():
                for (item_id, rating) in rating_history:
                    #matrix[user_id, item_id] = rating
                    item_popularity[item_id] += 1
            for user_id, rating_history in self.test_dict.items():
                for (item_id, rating) in rating_history:
                    #matrix[user_id, item_id] = rating
                    item_popularity[item_id] += 1

        item_popularity /= self.num_users

        #return matrix, item_popularity
        return item_popularity

    @property
    def train_users(self):
        return list(self.train_dict.keys())

    @property
    def valid_users(self):
        return list(self.valid_dict.keys())
    
    @property
    def test_users(self):
        return list(self.test_dict.keys())