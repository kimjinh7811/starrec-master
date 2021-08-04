import torch

def optimizer(learner, parameters, learning_rate, momentum=0.9):
    """

    Generate optimizer.

    :param str learner: name of the optimizer
    :param Tensor loss: loss to optimize from computational graph
    :param float learning_rate: learning rate
    :param float momentum: momentum value
    :return: optimizer
    """
    opt = None
    if learner.lower() == "adagrad":
        opt = torch.optim.Adagrad(parameters, learning_rate)
    elif learner.lower() == "rmsprop":
        opt = torch.optim.RMSprop(parameters, learning_rate)
    elif learner.lower() == "adam":
        opt = torch.optim.Adam(parameters, learning_rate)
    elif learner.lower() == "gd":
        opt = torch.optim.SGD(parameters, learning_rate)
    elif learner.lower() == "momentum":
        opt = torch.optim.SGD(parameters, learning_rate, momentum)
    else:
        raise ValueError("please select a suitable optimizer")
    return opt