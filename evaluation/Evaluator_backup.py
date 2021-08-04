import time
import numpy as np
from collections import OrderedDict
import pickle
import os

from evaluation import HoldoutEvaluator, LOOEvaluator
from dataloader.DataBatcher import DataBatcher

class Evaluator:
    def __init__(self, eval_pos, eval_target, eval_neg_candidates, eval_type, item_popularity, top_k): # jhkim, jwlee
        """
        :param str eval_type: data split type, leave_one_out(loo) or holdout
        :param list or int topK: top-k values to compute.
        :param int num_threads: number of threads to use
        """
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        
        self.batch_size = 1024
        self.eval_type = eval_type
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.eval_neg_candidates = eval_neg_candidates

        top_k = sorted(top_k) if isinstance(top_k, list) else [top_k]

        if self.eval_type in ['holdout', 'lko', 'hold-user-out', 'hold-out-user-5-fold-test']: # jhkim, jwlee
            self.eval_runner = HoldoutEvaluator(top_k, self.eval_pos, self.eval_target, self.eval_neg_candidates, item_popularity) # jhkim, jwlee
        elif self.eval_type == 'loo':
            self.eval_runner = LOOEvaluator(top_k, self.eval_pos, self.eval_target, self.eval_neg_candidates)
        else:
            raise NotImplementedError('Choose correct \'eval_type (current input: %s)' % self.eval_type)
    
    def compute_item_self_info(self, item_popularity):
        self_info = item_popularity / np.sum(item_popularity)
        self_info = -np.log(self_info)
        return self_info

    def evaluate(self, model, mean=True):
        # eval users
        eval_users = list(range(self.eval_pos.shape[0]))
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        score_cumulator = None
        for batch_user_ids in user_iterator:
            batch_eval_pos = self.eval_pos[batch_user_ids]

            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}

            if self.eval_neg_candidates is not None:
                eval_items_mask = np.zeros(batch_eval_pos.shape, dtype=np.bool)
                for i, user in enumerate(batch_user_ids):
                    eval_items_user = self.eval_target[user] + self.eval_neg_candidates[user]
                    eval_items_mask[i, eval_items_user] = True
                batch_pred = model.predict(batch_user_ids, batch_eval_pos, eval_items_mask)
            else:
                batch_pred = model.predict(batch_user_ids, batch_eval_pos)
                batch_pred[batch_eval_pos.nonzero()] = float('-inf')

            score_cumulator = self.eval_runner.compute_metrics(batch_pred, batch_eval_target, score_cumulator)

        # aggregate batch result
        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores['%s@%d' % (metric, k)] = score_by_ks[k].mean

        # return
        return scores

    def evaluate_test(self, model, data_set_name=False, model_name=False, func_complexity= False, pop_bias=False, test_fold=False, mean=True): # jhkim, jwlee
        # test set 별 model의 추천 list 저장
        # ex) dataset_name/model_name/test1
        # ex) dataset_name/model_name/test2
        # ex) dataset_name/model_name/test3
        # ex) dataset_name/model_name/test4
        # ex) dataset_name/model_name/test5

        # eval users
        eval_users = list(range(self.eval_pos.shape[0]))
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        # dictionary recommended_items
        data_recommended_items = []
        
        score_cumulator = None
        for batch_user_ids in user_iterator:
            batch_eval_pos = self.eval_pos[batch_user_ids]

            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}

            if self.eval_neg_candidates is not None:
                eval_items_mask = np.zeros(batch_eval_pos.shape, dtype=np.bool)
                for i, user in enumerate(batch_user_ids):
                    eval_items_user = self.eval_target[user] + self.eval_neg_candidates[user]
                    eval_items_mask[i, eval_items_user] = True
                batch_pred = model.predict(batch_user_ids, batch_eval_pos, eval_items_mask)
            else:
                batch_pred = model.predict(batch_user_ids, batch_eval_pos)
                batch_pred[batch_eval_pos.nonzero()] = float('-inf')

            score_cumulator = self.eval_runner.compute_metrics(batch_pred, batch_eval_target, score_cumulator)

            # top_k item index (not sorted)
            relevant_items_partition = (-batch_pred).argpartition(max(self.top_k), 1)[:, 0:max(self.top_k)]
            # top_k item score (not sorted)
            # relevant_items_partition_original_value = prediction[relevant_items_partition]
            relevant_items_partition_original_value = np.take_along_axis(batch_pred, relevant_items_partition, 1)
            # top_k item sorted index for partition
            relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, 1)
            # sort top_k index
            prediction = np.take_along_axis(relevant_items_partition, relevant_items_partition_sorting, 1)

            for i, user in enumerate(batch_user_ids):
                # data_recommended_items[user] = prediction[i]
                data_recommended_items.append(prediction[i])

        # aggregate batch result
        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores['%s@%d' % (metric, k)] = score_by_ks[k].mean

        ## sotre data # jhkim, jwlee
        if data_set_name and model_name:
            prefix_dir = "./diversity_evaluation"
            
            # user-item matrix
            if not os.path.exists(prefix_dir+'/'+data_set_name):
                os.makedirs(prefix_dir+'/'+data_set_name)
            with open(prefix_dir+'/'+data_set_name+'/'+'user_item_matrix', 'wb') as f:
                pickle.dump(self.eval_pos, f) ##

            # recommended items
            if model_name == 'ND_MultDAE' or model_name == 'ND_MultVAE' or model_name == 'ND_RecVAE':
                if not os.path.exists(prefix_dir+'/'+data_set_name+'/'+ model_name+'/'+func_complexity+'/'+pop_bias):
                    os.makedirs(prefix_dir+'/'+data_set_name+'/'+ model_name+'/'+func_complexity+'/'+pop_bias)

                with open(prefix_dir+'/'+data_set_name+'/'+ model_name+'/'+func_complexity+'/'+pop_bias+'/' + 'recommended_items_' + str(test_fold), 'wb') as f:
                    pickle.dump(data_recommended_items, f) ##
            else:
                if not os.path.exists(prefix_dir+'/'+data_set_name+'/'+ model_name):
                    os.makedirs(prefix_dir+'/'+data_set_name+'/'+ model_name)
                with open(prefix_dir+'/'+data_set_name+'/'+ model_name +'/' + 'recommended_items_' + str(test_fold), 'wb') as f:
                    pickle.dump(data_recommended_items, f) ##

        # return
        return scores
# aa

