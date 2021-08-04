import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import bottleneck as bn


class HoldoutEvaluator:
    def __init__(self, top_k, num_threads=1):
        self.top_k = top_k
        self.max_k = max(top_k)
        self.num_threads = num_threads

    def compute(self, model, eval_output, eval_target, output_mask=None):
        # eval_output: (# users, # items) matrix
        # eval_target: dictionary - {user_id: list of target items}
        # output_mask: dictionary - {user_id: list of mask items}

        def compute_one_user(user):
            predictions = eval_output[user]
            target_items = [x[0] for x in eval_target[user]]
            num_target_items = len(target_items)

            # prediction = predictions.argsort()[::-1]
            # prediction = np.argpartition(predictions, self.max_top - 1)[0:self.max_top]

            # top_k item index (not sorted)
            
            # epoch=  2, loss=890623.812, train time=0.60, epoch time=8.69 (0.60 + 8.09), Prec@100=0.0088, Recall@100=0.0598, NDCG@100=0.0284
            relevant_items_partition = (-predictions).argpartition(self.max_k)[0:self.max_k]
            
            # epoch=  3, loss=826951.375, train time=0.60, epoch time=8.62 (0.60 + 8.02), Prec@100=0.0107, Recall@100=0.0780, NDCG@100=0.0364
            # relevant_items_partition = bn.argpartition(-predictions, self.max_k)[0:self.max_k]
            

            # top_k item score (not sorted)
            relevant_items_partition_original_value = predictions[relevant_items_partition]

            # top_k item sorted index for partition
            relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value)

            # sort top_k index
            prediction = relevant_items_partition[relevant_items_partition_sorting]

            prec, recall, ndcg = {}, {}, {}
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
                recall[k] = num_hits / min(num_target_items, k)
                #recall[k] = num_hits / num_target_items
                ndcg[k] = dcg_k / idcg_k
            return prec, recall, ndcg

        eval_users = []
        for u in output_mask:
            mask_items = output_mask[u]
            if len(mask_items) > 0:
                eval_output[u, mask_items] = float('-inf')
                eval_users.append(u)

        precs, recalls, ndcgs \
            = {k: [] for k in self.top_k}, {k: [] for k in self.top_k}, {k: [] for k in self.top_k}
        if self.num_threads > 1:  # Multi-thread
            with ThreadPoolExecutor() as executor:
                res = executor.map(compute_one_user, eval_users)
                res = list(res)
                for prec, recall, ndcg in res:
                    for k in self.top_k:
                        precs[k].append(prec[k])
                        recalls[k].append(recall[k])
                        ndcgs[k].append(ndcg[k])
        else:
            for u in eval_users:
                (prec, recall, ndcg) = compute_one_user(u)
                for k in self.top_k:
                    precs[k].append(prec[k])
                    recalls[k].append(recall[k])
                    ndcgs[k].append(ndcg[k])

        score = OrderedDict({
            'Prec': precs,
            'Recall': recalls,
            'NDCG': ndcgs
        })
        return score