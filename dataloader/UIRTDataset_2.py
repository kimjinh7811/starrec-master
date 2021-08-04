import numpy as np

from base import BaseDataset
from dataloader.DataLoader import load_data_and_info
import scipy.sparse as sp

class UIRTDatset(BaseDataset):
    def __init__(self, data_dir, dataset, implicit, split_type="loo", train_ratio=0.8, valid_ratio=0.1, split_random=True, eval_neg_num=100, popularity_order=True):
        super(UIRTDatset, self).__init__(data_dir, dataset, implicit, split_type, train_ratio, valid_ratio, split_random, popularity_order)

        print('load_data')
        if split_type == 'user':
            self.train_dict, self.valid_input_dict, self.valid_target_dict, self.test_input_dict, self.test_target_dict, self.user_id_dict, self.user_to_num_items, self.item_id_dict, self.item_to_num_users \
                = load_data_and_info(self.data_file, self.info_file, split_type)
        else:
            self.train_dict, self.valid_dict, self.test_dict, self.user_id_dict, self.user_to_num_items, self.item_id_dict, self.item_to_num_users \
                = load_data_and_info(self.data_file, self.info_file, split_type)

        self.num_users = len(self.user_id_dict)
        self.num_items = len(self.item_id_dict)
        self.split_type = split_type

        # self.num_users, self.num_items, self.train_dict, self.valid_dict, self.test_dict \
        #     = read_data_file(self.train_file, self.valid_file, self.test_file, self.info_file, implicit)

        print('gen_neg_item')
        self.neg_dict = self.generate_neg_items_user() if split_type == 'user' else self.generate_neg_items()
        print('gen_neg_finish')
        self.eval_mode = 'valid'
        self.eval_candidates = None
        self.eval_target = None
        self.eval_input_dict = None
        self.output_mask = None

        self.eval_neg_nums = eval_neg_num

        if split_type == 'loo' and self.eval_neg_nums > 0:
            self.eval_neg_dict = self.generate_eval_neg()
        else:
            self.eval_neg_dict = self.neg_dict

    def generate_neg_items(self):
        neg_dict = {u: [] for u in range(self.num_users)}
        for user in list(self.train_dict.keys()):
            train_pos = self.train_dict[user]
            valid_pos = self.valid_dict[user]
            test_pos = self.test_dict[user]

            train_pos_items = [x[0] for x in train_pos]
            valid_pos_items = [x[0] for x in valid_pos]
            test_pos_items = [x[0] for x in test_pos]

            # 1. set operation
            # neg_items = list(set(range(self.num_items)) - set(pos_items))

            # 2. array
            neg_items = np.zeros(self.num_items)
            neg_items[train_pos_items] = 1
            neg_items[valid_pos_items] = 1
            neg_items[test_pos_items] = 1
            neg_items = np.where(neg_items==0)[0].tolist()

            neg_dict[user] = np.random.permutation(neg_items)[:1000]

        return neg_dict

    def generate_neg_items_user(self):
        neg_dict = {u: [] for u in range(self.num_users)}

        for user in range(self.num_users):
            if user in self.train_dict.keys():
                pos_item = [x[0] for x in self.train_dict[user]]
                #print('Train')
            elif user in self.valid_input_dict.keys():
                pos_item = [x[0] for x in self.valid_input_dict[user]] + [self.valid_target_dict[user][0]]
            else:
                pos_item = [x[0] for x in self.test_input_dict[user]] + [self.test_target_dict[user][0]]

            neg_items = np.zeros(self.num_items)
            try:
                neg_items[pos_item] = 1
            except:
                pass
                #print('pos_item:', pos_item)
            neg_items = np.where(neg_items==0)[0].tolist()

            neg_dict[user] = np.random.permutation(neg_items)[:10000]

        return neg_dict

    def generate_eval_neg(self):
        """
        generate negative samples for evaluation per user

        return dictionary: key - user, values - list of neg item ids
        """
        eval_neg_dict = {}
        for user in self.neg_dict:
            neg_items = self.neg_dict[user]
            if len(neg_items) > self.eval_neg_nums:
                eval_neg_idx = np.random.choice(len(neg_items), self.eval_neg_nums)
                eval_neg_dict[user] = [neg_items[i] for i in eval_neg_idx]
        return eval_neg_dict

    def set_eval_data(self, MODE):
        if self.eval_mode != MODE or self.eval_target is None:
            self.eval_mode = MODE
            if MODE == 'valid':
                self.set_valid_data()
            elif MODE == 'test':
                self.set_test_data()
            else:
                raise ValueError('Choose correct dataset mode. (valid or test)')
            self.eval_input_dict = None

    def set_valid_data(self):
        """
        eval_input_dict: input data to use
        eval_target: target items
        output_mask: point to mask out as '-inf'
        """
        if self.split_type == 'user':
            self.eval_target = self.valid_target_dict
            self.output_mask = {u: [x[0] for x in items] for u, items in self.valid_input_dict.items()}
        else:
            self.eval_target = self.valid_dict
            self.output_mask = {u: [x[0] for x in items] for u, items in self.train_dict.items()}
    

    def set_test_data(self):
        # input: eval_neg + valid +  test
        # target: test
        if self.split_type == 'user':
            self.eval_target = self.test_target_dict
            self.output_mask = {u: [x[0] for x in items] for u, items in self.test_input_dict.items()}
        else:
            self.eval_target = self.test_dict
            self.output_mask = {}
            for u in self.train_dict:
                # if u in self.train_dict:
                self.output_mask[u] = [x[0] for x in self.train_dict[u]]
                if u in self.valid_dict:
                    self.output_mask[u] += [x[0] for x in self.valid_dict[u]]
    
    def get_eval_data(self):
        return self.eval_target, self.output_mask

    def get_eval_input(self):
        if self.eval_input_dict is None:
            self.eval_input_dict = {}
            for u in self.eval_neg_dict:
                if self.eval_mode == 'valid':
                    self.eval_input_dict[u] = self.eval_neg_dict[u] + [x[0] for x in self.valid_dict[u]] + [x[0] for x in self.test_dict[u]]
                elif self.eval_mode == 'test':
                    self.eval_input_dict[u] = self.eval_neg_dict[u] + [x[0] for x in self.test_dict[u]]
                else:
                    raise ValueError('Choose correct dataset mode. (valid or test)')
        
        return self.eval_input_dict

    def get_listwise_eval_input(self):

        if self.split_type == 'user':
            if self.eval_mode == 'valid':
                self.eval_input_dict = self.valid_input_dict
            else:
                self.eval_input_dict = self.test_input_dict

        else:
            if self.eval_input_dict is None:
                self.eval_input_dict = {u: [] for u in self.train_dict}

                if self.eval_mode == 'valid':
                    self.eval_input_dict = self.train_dict

                elif self.eval_mode == 'test':
                    for u in self.train_dict:
                        self.eval_input_dict[u] += self.train_dict[u]
                    for u in self.valid_dict:
                        self.eval_input_dict[u] += self.valid_dict[u]
                else:
                    raise ValueError('Choose correct dataset mode. (valid or test)')
        
        return self.eval_input_dict

    def __str__(self):
        ret_str ='\n'
        ret_str += 'Dataset: %s' % self.data_name
        ret_str += '# of users: %d' % self.num_users
        ret_str += '# of items: %d' % self.num_items     
        ret_str += 'Split type: %s' % self.split_type
        if self.split_type == 'holdout':
            val = self.train_ratio * self.valid_ratio
            train = self.train_ratio - val
            ret_str += 'Split random: %s' % self.split_random
            ret_str += 'train / val / test: %.1f, %.1f, %.1f' % (train, val, 1 - self.train_ratio)
        ret_str += '# of eval negatiaves: %s' % 'all' if self.eval_neg_nums < 1 else str(self.eval_neg_nums)
        
        return ret_str

    #def convert_to_sparse_matrix(self):
    def get_item_popularity(self):
        #matrix = sp.dok_matrix((self.num_users, self.num_items), dtype=np.float32)
        item_popularity = np.zeros(self.num_items)

        for user_id, rating_history in self.train_dict.items():
            for (item_id, rating) in rating_history:
                #matrix[user_id, item_id] = rating
                item_popularity[item_id] += 1

        if self.split_type == 'user':
            for user_id, rating_history in self.valid_input_dict.items():
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