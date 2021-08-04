import os
from time import time

import tensorflow as tf

from base.BaseTrainer import BaseTrainer

class NonNeuralTrainer(BaseTrainer):
    def __init__(self, sess, model, dataset, evaluator, logger, saver, early_stop,
                 num_epochs, verbose=0, print_step=10, test_step=1, test_from=1):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.logger = logger
        self.saver = saver

    def train(self):
        # Train model until convergence
        self.sess.run(tf.global_variables_initializer())
        self.dataset.switch_mode('valid')

        exp_start = time()

        epoch_loss = self.model.train_model(self.verbose)

        score = self.evaluate()
        score_str = ', '.join(['%s = %.4f' % (k, score[k]) for k in score])

        self.logger.info('\n')
        exp_time = time() - exp_start
        best_epoch, best_score = self.early_stop.best_epoch, self.early_stop.best_score
        return best_epoch, best_score, exp_time
