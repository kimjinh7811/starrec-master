from evaluation.backend.python.NoveltyEvaluator import NoveltyEvaluator

try:
    from evaluation.backend.cython.HoldoutEvaluator import HoldoutEvaluator
    from evaluation.backend.cython.LOOEvaluator import LOOEvaluator
    from evaluation.backend.cython.tool import predict_topk
except:
    print('evaluation with python backend...')
    from evaluation.backend.python.HoldoutEvaluator import HoldoutEvaluator
    from evaluation.backend.python.LOOEvaluator import LOOEvaluator
    from evaluation.backend.python.tool import predict_topk