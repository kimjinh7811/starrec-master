import os
from time import time

import tensorflow as tf

class Trainer:
    def __init__(self, sess, model, dataset, evaluator, logger, saver, early_stop,
                 num_epochs, verbose=0, print_step=10, test_step=1, test_from=1):
        self.sess = sess
        self.model = model
        self.dataset = dataset
        self.evaluator = evaluator
        self.logger = logger
        self.saver = saver

        self.num_epochs = num_epochs
        self.early_stop = early_stop

        self.verbose = verbose
        self.print_step = print_step
        self.test_step = test_step
        self.test_from = test_from

    def train(self):
        # Train model until convergence
        self.sess.run(tf.global_variables_initializer())
        self.dataset.switch_mode('valid')

        exp_start = time()
        for epoch in range(1, self.num_epochs + 1):
            train_start = time()
            epoch_loss = self.model.train_model(self.verbose)
            train_time = time() - train_start

            epoch_str = '[Epoch %3d] loss = %.4f ' % (epoch, epoch_loss)

            test_time = 0.0
            if epoch >= self.test_from or epoch == self.num_epochs:
                test_start = time()
                score = self.evaluate()
                score_str = ', '.join(['%s = %.4f' % (k, score[k]) for k in score])
                test_time += time() - test_start

                updated, should_stop = self.early_stop.step(score, epoch)
                self.logger.info('%s %s [time = %.1f+%.1f=%.1f]' %
                                 (epoch_str, score_str, train_time, test_time, train_time + test_time))
                if should_stop:
                    self.logger.info('Early stop triggered.')
                    break

                if updated and self.saver:
                    self.saver.save(self.sess, os.path.join(self.logger.log_dir, 'best_model'),
                                    write_meta_graph=False, global_step=epoch)
            else:
                self.logger.info('%s [time = %.1f+%.1f=%.1f]' %(epoch_str, train_time, test_time, train_time + test_time))

        self.logger.info('\n')
        exp_time = time() - exp_start
        best_epoch, best_score = self.early_stop.best_epoch, self.early_stop.best_score
        return best_epoch, best_score, exp_time

    def evaluate(self):
        # test model
        eval_output = self.model.predict(self.dataset)
        eval_input = self.dataset.eval_input
        eval_target = self.dataset.eval_target
        score = self.evaluator.compute(eval_input, eval_target, eval_output)

        return score
