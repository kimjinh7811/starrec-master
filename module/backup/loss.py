import torch
import torch.nn as nn
import torch.nn.functional as F

def pairwise_loss(loss_function, y, margin=0.5):
    """
    Calculate pairwise loss

    :param loss_function: name of loss to use
    :param y: difference between ground truth & prediction
    :param margin: margin value for hinge loss
    :return: loss operation
    """
    loss = None
    if loss_function.lower() == "bpr":
        nn.Loss
        loss = - torch.sigmoid(y).log.sum()
    elif loss_function.lower() == "hinge":
        loss = torch.max(y + margin, 0).sum()
    elif loss_function.lower() == "square":
        loss = tf.reduce_sum(tf.square(1 - y))
    else:
        raise Exception("please choose a suitable loss function")
    return loss

# def pointwise_loss(loss_function, y, pred_y, eps=1e-10):
#     """
#     Calculate pointwise loss

#     :param loss_function: name of loss to use
#     :param y: ground truth
#     :param pred_y: prediction
#     :return: loss operation
#     """

#     loss = None
#     if loss_function.lower() == "cross_entropy":
#         loss = -tf.reduce_sum(y * tf.log(tf.sigmoid(pred_y) + eps) + (1 - y) * tf.log(1 - tf.sigmoid(pred_y) + eps))
#     elif loss_function.lower() == "multinominal":
#         loss = -tf.reduce_mean(tf.reduce_sum(y * tf.nn.log_softmax(pred_y, axis=-1)))
#     elif loss_function.lower() == "square":
#         loss = tf.reduce_sum(tf.square(y - pred_y))
#     else:
#         raise Exception("please choose a suitable loss function")
#     return loss

# def weighted_pointwise_loss(loss_function, y, pred_y, weight, eps=1e-10):
#     if loss_function.lower() == "cross_entropy":
#         loss = - tf.reduce_sum(weight * (y * tf.log(tf.sigmoid(pred_y) + eps) + (1 - y) * tf.log(1 - tf.sigmoid(pred_y) + eps)))
#     elif loss_function.lower() == "square":
#         loss = tf.reduce_sum(weight * (tf.square(y - pred_y)))
#     else:
#         raise Exception("please choose a suitable loss function")
#     return loss