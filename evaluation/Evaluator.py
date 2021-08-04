import time
import numpy as np
import os
import pickle
from collections import OrderedDict

from evaluation.backend import HoldoutEvaluator, LOOEvaluator, NoveltyEvaluator, predict_topk
from dataloader.DataBatcher import DataBatcher

class Evaluator:
    def __init__(self, eval_pos, eval_target, eval_neg_candidates, eval_type, item_popularity, num_user, top_k, novelty=False):
        """
        :param str eval_type: data split type, leave_one_out(loo) or holdout
        :param list or int topK: top-k values to compute.
        :param int num_threads: number of threads to use
        """
        self.top_k = top_k if isinstance(top_k, list) else [top_k]
        self.max_k = max(self.top_k)
        self.novelty = novelty            
        
        self.batch_size = 1024
        self.eval_type = eval_type
        self.eval_pos = eval_pos
        self.eval_target = eval_target
        self.eval_neg_candidates = eval_neg_candidates

        top_k = sorted(top_k) if isinstance(top_k, list) else [top_k]

        if self.eval_type in ['holdout', 'lko', 'hold-user-out', 'hold-out-user-5-fold-test']:
            self.eval_runner = HoldoutEvaluator(top_k, self.eval_pos, self.eval_target, self.eval_neg_candidates)
        elif self.eval_type == 'loo':
            self.eval_runner = LOOEvaluator(top_k, self.eval_pos, self.eval_target, self.eval_neg_candidates)
        else:
            raise NotImplementedError('Choose correct \'eval_type (current input: %s)' % self.eval_type)
            

        if self.novelty:
            self.novelty_runner = NoveltyEvaluator(np.array(top_k, dtype=int))
            self.item_self_information = self.compute_item_self_info(item_popularity, num_user)


    def compute_item_self_info(self, item_popularity, num_user):
        self_info = item_popularity / num_user
        self_info[ self_info > 0 ] = -np.log( self_info[self_info >0] )
        
        return self_info

    def evaluate(self, model, mean=True):
        # eval users
        num_users, num_items = self.eval_pos.shape
        eval_users = list(range(num_users))
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        score_cumulator = None
        novelty_cumulator = None
        item_recommend_counter = np.zeros(num_items, dtype=np.int)

        # index_time = 0.0
        # pred_time = 0.0
        # mask_time = 0.0
        top_k_time = 0.0
        metric_time = 0.0
        for batch_user_ids in user_iterator:
            # batch_eval_pos = self.eval_pos[batch_user_ids]
            
            # need refactoring
            # s = time.time()
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}
            # index_time += time.time() - s

            #   make prediction
            if self.eval_neg_candidates is not None:
                eval_items = []
                eval_items_mask = np.zeros((len(batch_user_ids), num_items), dtype=np.bool)
                for i, user in enumerate(batch_user_ids):
                    eval_items_user = self.eval_target[user] + self.eval_neg_candidates[user]
                    eval_items_mask[i, eval_items_user] = True
                batch_pred = model.predict(batch_user_ids, self.eval_pos, eval_items_mask)
            else:
                # s = time.time()
                batch_pred = model.predict(batch_user_ids, self.eval_pos)
                # pred_time += time.time() - s

            
            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)


            if self.novelty:
                novelty_cumulator = self.novelty_runner.compute_metrics(batch_topk, self.item_self_information, novelty_cumulator)
                # print('novelty_time:', time.time() - s)

                # Gini-Diversity
                rec_item, rec_count = np.unique(batch_topk, return_counts=True)
                item_recommend_counter[rec_item] += rec_count
        
        # print('top_k_time:', top_k_time)
        # print('metric_time:', metric_time)

        # print('index: %.2f, pred: %.2f, mask: %.2f, metric: %.2f' % (index_time, pred_time, mask_time, metric_time))
        # aggregate batch result


        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores['%s@%d' % (metric, k)] = score_by_ks[k].mean

        if self.novelty:
            # Novelty
            for metric in novelty_cumulator:
                novelty_by_ks = novelty_cumulator[metric]
                for k in novelty_by_ks:
                    scores['%s@%d' % (metric, k)] = novelty_by_ks[k].mean

            # Gini-Diversity
            item_recommend_counter_mask = np.ones_like(item_recommend_counter, dtype = np.bool)
            item_recommend_counter_mask[item_recommend_counter == 0] = False
            item_recommend_counter = item_recommend_counter[item_recommend_counter_mask]
            num_eff_items = len(item_recommend_counter)

            item_recommend_counter_sorted = np.sort(item_recommend_counter)       # values must be sorted
            index = np.arange(1, num_eff_items+1)                                 # index per array element

            gini_diversity = 2 * np.sum((num_eff_items + 1 - index) / (num_eff_items + 1) * item_recommend_counter_sorted / np.sum(item_recommend_counter_sorted))
            scores['Gini-D'] = gini_diversity

        # return
        return scores

    def evaluate_test(self, model, data_set_name=False, model_name=False, test_fold=False, means=True):
        # test set 별 model의 추천 list 저장
        # ex) dataset_name/model_name/test1
        # ex) dataset_name/model_name/test2
        # ex) dataset_name/model_name/test3
        # ex) dataset_name/model_name/test4
        # ex) dataset_name/model_name/test5

        # eval users
        num_users, num_items = self.eval_pos.shape
        eval_users = list(range(num_users))
        user_iterator = DataBatcher(eval_users, batch_size=self.batch_size)

        score_cumulator = None
        novelty_cumulator = None
        item_recommend_counter = np.zeros(num_items, dtype=np.int)

        # dict recommended items
        data_recommended_items = []

        for batch_user_ids in user_iterator:
            batch_eval_target = {u: self.eval_target[u] for u in batch_user_ids}

            if self.eval_neg_candidates is not None:
                eval_items = []
                eval_items_mask = np.zeros((len(batch_user_ids), num_items), dtype=np.bool)
                for i, user in enumerate(batch_user_ids):
                    eval_items_user = self.eval_target[user] + self.eval_neg_candidates[user]
                    eval_items_mask[i, eval_items_user] = True
                batch_pred = model.predict(batch_user_ids, self.eval_pos, eval_items_mask)

            else:
                batch_pred = model.predict(batch_user_ids, self.eval_pos)

            batch_topk = predict_topk(batch_pred.astype(np.float32), self.max_k).astype(np.int64)
            score_cumulator = self.eval_runner.compute_metrics(batch_topk, batch_eval_target, score_cumulator)

            for i, user in enumerate(batch_user_ids):
                # data_recommended_items[user] = prediction[i]
                data_recommended_items.append(batch_topk[i])

            if self.novelty:
                novelty_cumulator = self.novelty_runner.compute_metrics(batch_topk, self.item_self_information, novelty_cumulator)

                # Gini-Diversity
                rec_item, rec_count = np.unique(batch_topk, return_counts=True)
                item_recommend_counter[rec_item] += rec_count

        # aggregate batch result
        scores = {}
        for metric in score_cumulator:
            score_by_ks = score_cumulator[metric]
            for k in score_by_ks:
                scores['%s@%d' % (metric, k)] = score_by_ks[k].mean

        if self.novelty:
            # Novelty
            for metric in novelty_cumulator:
                novelty_by_ks = novelty_cumulator[metric]
                for k in novelty_by_ks:
                    scores['%s@%d' % (metric, k)] = novelty_by_ks[k].mean

            # Gini-Diversity
            item_recommend_counter_mask = np.ones_like(item_recommend_counter, dtype = np.bool)
            item_recommend_counter_mask[item_recommend_counter == 0] = False
            item_recommend_counter = item_recommend_counter[item_recommend_counter_mask]
            num_eff_items = len(item_recommend_counter)

            item_recommend_counter_sorted = np.sort(item_recommend_counter)       # values must be sorted
            index = np.arange(1, num_eff_items+1)                                 # index per array element

            gini_diversity = 2 * np.sum((num_eff_items + 1 - index) / (num_eff_items + 1) * item_recommend_counter_sorted / np.sum(item_recommend_counter_sorted))
            scores['Gini-D'] = gini_diversity

        '''
        ## sotre data # jhkim, jwlee
        if data_set_name and model_name:
            prefix_dir = "./diversity_evaluation"
    
            # recommended items
            if model_name[:2] == 'ND':
                if not os.path.exists(prefix_dir+'/'+data_set_name+'/'+ model_name+'/'+config['Model']['func_complexity']+'/'+config['Model']['pop_bias']):
                    os.makedirs(prefix_dir+'/'+data_set_name+'/'+ model_name+'/'+config['Model']['func_complexity']+'/'+config['Model']['pop_bias'])
                with open(prefix_dir+'/'+data_set_name+'/'+ model_name+'/'+config['Model']['func_complexity']+'/'+config['Model']['pop_bias'] +'/' + 'recommended_items_' + str(test_fold), 'wb') as f:
                    pickle.dump(data_recommended_items, f) ##

            else:
                if not os.path.exists(prefix_dir+'/'+data_set_name+'/'+ model_name):
                    os.makedirs(prefix_dir+'/'+data_set_name+'/'+ model_name)
                with open(prefix_dir+'/'+data_set_name+'/'+ model_name +'/' + 'recommended_items_' + str(test_fold), 'wb') as f:
                    pickle.dump(data_recommended_items, f) ##
        '''

        # return
        return scores

    

    def predict_topk(self, scores):
        # top_k item index (not sorted)
        relevant_items_partition = (-scores).argpartition(self.max_k, 1)[:, 0:self.max_k]

        # top_k item score (not sorted)
        # relevant_items_partition_original_value = prediction[relevant_items_partition]
        relevant_items_partition_original_value = np.take_along_axis(scores, relevant_items_partition, 1)
        # top_k item sorted index for partition
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, 1)
        # sort top_k index
        # prediction = relevant_items_partition[relevant_items_partition_sorting]
        topk = np.take_along_axis(relevant_items_partition, relevant_items_partition_sorting, 1)

        return topk

    def update(self, eval_pos=None, eval_target=None, eval_neg_candidates=None):
        if eval_pos is not None:
            self.eval_pos = eval_pos
        if eval_target is not None:
            self.eval_target = eval_target
        if eval_neg_candidates is not None:
            self.eval_neg_candidates = eval_neg_candidates