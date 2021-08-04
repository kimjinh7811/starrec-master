import math

import torch
import torch.nn
import torch.nn.functional as F

def activation_function(act, act_input):
    act_func = None
    if act == "sigmoid":
        act_func = F.sigmoid(act_input)
    elif act == "tanh":
        act_func = F.tanh(act_input)
    elif act == "relu":
        act_func = F.relu(act_input)
    elif act == "elu":
        act_func = F.elu(act_input)
    elif act == "identity":
        act_func = act_input
    else:
        raise NotImplementedError("ERROR")
    return act_func

def kernel(kernel_type, kernel_h, dist):
    if kernel_type.lower() == 'epanechnikov':
        # kernel_h보다 dist가 크면 0, 작으면 계산
        # 0 ~ 0.75
        return (3 / 4) * max((1 - (dist / kernel_h) ** 2), 0)
    if kernel_type.lower() == 'uniform':
        return (dist < kernel_h)
    if kernel_type.lower() == 'triangular':
        return max((1 - dist / kernel_h), 0)
    if kernel_type.lower() == 'random':
        return torch.rand(0, 1) * (dist < kernel_h)
        # return np.random.uniform(0, 1) * (dist < kernel_h)

def dist(dist_type, a, b):
    if torch.eq(a, b): return 0
    if dist_type.lower() == 'arccos':
        return (2 / math.pi) * torch.acos(torch.mul(a, b) / (torch.norm(a) * torch.norm(b)))
    elif dist_type == 'cos':
        return (1 - torch.mul(a, b) / (torch.norm(a) * torch.norm(b)))  # 0 ~ 2
    else:
        raise NameError("Please write correct dist_type")