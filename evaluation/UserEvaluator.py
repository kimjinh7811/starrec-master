import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class UserEvaluator:
    def __init__(self, top_k, num_threads=1):
        self.top_k = top_k
        self.max_k = max(top_k)
        self.num_threads = num_threads

    def compute(self, model, eval_output, eval_target, output_mask=None):
        # eval_output: (# users, # items) matrix
        # eval_target: dictionary - {user_id: list of target items}
        # output_mask: dictionary - {user_id: list of items to be masked as -inf}

        '''
        def compute_one_user(user):
            predictions = eval_output[user]
            target_item = eval_target[user][0][0]

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
            if hasattr(model, 'item_popularity'):
                nov =  {k: np.sum( -np.log2(model.item_popularity[ranking] + 1e-10)[:k] )/k for k in self.top_k}
                unpop = {k: np.sum(1 - model.item_popularity[ranking][:k])/k for k in self.top_k}
            else:
                nov =  {k: -1.0 for k in self.top_k}
                unpop = {k: -1.0 for k in self.top_k}

            return hr, ndcg, nov, unpop
        '''

        def compute_one_user(user):
            predictions = eval_output[user]
            target_items = [x[0] for x in eval_target[user]]
            num_target_items = len(target_items)

            # top_k item index (not sorted)
            relevant_items_partition = (-predictions).argpartition(self.max_k)[0:self.max_k]

            # top_k item score (not sorted)
            relevant_items_partition_original_value = predictions[relevant_items_partition]

            # top_k item sorted index for partition
            relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value)

            # sort top_k index
            prediction = relevant_items_partition[relevant_items_partition_sorting]

            prec, recall, ndcg, nov, unpop = {}, {}, {}, {}, {}
            for k in self.top_k:
                pred_k = prediction[:k]
                # hits_k = [(i + 1, item) for i, item in enumerate(target_items) if item in pred_k] # 변경 전입니다.
                hits_k = [(i + 1, item) for i, item in enumerate(pred_k) if item in target_items]
                num_hits = len(hits_k)
                idcg_k = 0.0
                # for i in range(1, k + 1):  # 변경 전입니다.
                for i in range(1, min(num_target_items, k) + 1):
                    idcg_k += 1 / math.log(i + 1, 2)
                dcg_k = 0.0
                for idx, item in hits_k:
                    dcg_k += 1 / math.log(idx + 1, 2)
                prec[k] = num_hits / k
                recall[k] = num_hits / num_target_items
                ndcg[k] = dcg_k / idcg_k

                if hasattr(model, 'item_popularity'):
                    nov[k] = np.sum( -np.log2(model.item_popularity[prediction] + 1e-10)[:k] )/k
                    unpop[k] = np.sum(1 - model.item_popularity[prediction][:k])/k
                else:
                    nov[k] = -1
                    unpop[k] = -1

            return prec, recall, ndcg, nov, unpop


        eval_users = []
        for u in output_mask:
            mask_items = output_mask[u]
            if len(mask_items) > 0:
                eval_output[u, mask_items] = float('-inf')
                eval_users.append(u)

        precs = {k: [] for k in self.top_k}
        recalls = {k: [] for k in self.top_k}
        ndcgs = {k: [] for k in self.top_k}
        novs ={k: [] for k in self.top_k}
        unpops = {k: [] for k in self.top_k}

        if self.num_threads > 1:  # Multi-thread
            with ThreadPoolExecutor() as executor:
                res = executor.map(compute_one_user, eval_users)
            res = list(res)

            for _prec, _recall, _ndcg, _nov, _unpop in res:
                for k in self.top_k:
                    precs[k].append(_prec[k])
                    recalls[k].append(_recall[k])
                    ndcgs[k].append(_ndcg[k])
                    novs[k].append(_nov[k])
                    unpops[k].append(_unpop[k])

        else:
            for u in eval_users:
                prec, recall, ndcg, nov, unpop = compute_one_user(u)
                for k in self.top_k:
                    precs[k].append(prec[k])
                    recalls[k].append(recall[k])
                    ndcgs[k].append(ndcg[k])
                    novs[k].append(nov[k])
                    unpops[k].append(unpop[k])


        score = OrderedDict({
            'Prec' : precs,
            'Recall': recalls,
            'NDCG': ndcgs,
            'Novelty': novs,
            'unpop': unpops,
        })
        return score