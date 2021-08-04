import math
from collections import OrderedDict
import numpy as np

from utils.Statistics import Statistics

class NoveltyEvaluator:
    def __init__(self, top_k):
        self.top_k = top_k
        self.max_k = max(top_k)

    def init_score_cumulator(self):
        score_cumulator = OrderedDict()
        score_cumulator['Nov.'] = {k: Statistics('Nov.@%d' % k) for k in self.top_k}
        return score_cumulator

    def compute_metrics(self, topk, item_self_information, score_cumulator=None):
        if score_cumulator is None:
            score_cumulator = self.init_score_cumulator()
        
        topk_info = np.take(item_self_information, topk)
        topk_info_sum = np.cumsum(topk_info, 1)[:, self.top_k - 1]
        novelty = topk_info_sum / np.atleast_2d(self.top_k)

        for u in range(len(novelty)):
            novelty_u = novelty[u]
            for i, k in enumerate(self.top_k):
                score_cumulator['Nov.'][k].update(novelty_u[i])

        return score_cumulator

if __name__ == '__main__':
    pop = np.array([0.1, 0.3, 0.2, 0.1, 0.05, 0.2, 0.05])
    info = -np.log(pop)
    topk = np.array([
        [1, 0, 6, 4, 2],
        [6, 1, 0, 3, 2]
    ])

    evaluator = NoveltyEvaluator(np.array([2, 4, 5]))
    score = evaluator.compute_metrics(topk, info)

    print(score)