import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class LOOEvaluator:
    def __init__(self, top_k, num_threads=1):
        self.top_k = top_k
        self.max_k = max(top_k)
        self.num_threads = num_threads

    def compute(self, model, eval_output, eval_target, output_mask=None):
        # eval_output: (# users, # items) matrix
        # eval_target: dictionary - {user_id: list of target items}
        # output_mask: dictionary - {user_id: list of items to be masked as -inf}

        def compute_one_user(user):
            predictions = eval_output[user]
            target_item = eval_target[user][0][0]

            # prediction = predictions.argsort()[::-1]
            # hit_at_k = np.where(prediction == target_item)[0][0] + 1
            #
            # hr = {k: 1 if hit_at_k <= k else 0 for k in self.top_k}
            # ndcg = {k: 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0 for k in self.top_k}
            # ap = 1 / hit_at_k
            # auc = 1 - (1 / 2) * (1 - 1 / hit_at_k)

            ###################################################### TO BE APPLIED
            # top_k item index (not sorted)
            relevant_items_partition = (-predictions).argpartition(self.max_k)[0:self.max_k]

            # top_k item score (not sorted)
            relevant_items_partition_original_value = predictions[relevant_items_partition]

            # top_k item sorted index for partition
            relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value)

            # sort top_k index
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            hit_at_k = np.where(ranking == target_item)[0][0] + 1 if target_item in ranking else self.max_k + 1

            hr = {k: 1 if hit_at_k <= k else 0 for k in self.top_k}
            ndcg = {k: 1 / math.log(hit_at_k + 1, 2) if hit_at_k <= k else 0 for k in self.top_k}
            nov = {k: np.sum(-np.log2(model.item_popularity[ranking] + 1e-10)[:k]) / k for k in self.top_k}
            unpop = {k: np.sum(1 - model.item_popularity[ranking][:k]) / k for k in self.top_k}

            # ap = 1 / hit_at_k
            # auc = 1 - (1 / 2) * (1 - 1 / hit_at_k)
            ######################################################

            return hr, ndcg


        eval_users = []
        for u in output_mask:
            mask_items = output_mask[u]
            if len(mask_items) > 0:
                eval_output[u, mask_items] = float('-inf')
                eval_users.append(u)

        # hrs, ndcgs, aps, aucs = {k: [] for k in self.top_k}, {k: [] for k in self.top_k}, [], []
        hrs, ndcgs, novs, unpops = {k: [] for k in self.top_k}, {k: [] for k in self.top_k}, \
                                   {k: [] for k in self.top_k}, {k: [] for k in self.top_k}

        if self.num_threads > 1:  # Multi-thread
            with ThreadPoolExecutor() as executor:
                res = executor.map(compute_one_user, eval_users)
            res = list(res)
            # for _hr, _ndcg, _ap, _auc in res:
            for _hr, _ndcg, _nov, _bnov in res:
                for k in self.top_k:
                    hrs[k].append(_hr[k])
                    ndcgs[k].append(_ndcg[k])
                    novs[k].append(_nov[k])
                    unpops[k].append(_bnov[k])
                # aps.append(_ap)
                # aucs.append(_auc)
        else:
            hr, ndcg, nov, unpop = compute_one_user(u)
            for k in self.top_k:
                hrs[k].append(hr[k])
                ndcgs[k].append(ndcg[k])
                novs[k].append(nov[k])
                unpops[k].append(unpop[k])

        score = OrderedDict({
            'HR': hrs,
            'NDCG': ndcgs,
            'Novelty': novs,
            'unpop': unpops
        })
        return score