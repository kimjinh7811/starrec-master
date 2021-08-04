import os
import math
from time import time

import numpy as np
import tensorflow as tf

from base.BaseRecommender import BaseRecommender
# from utils import Tool
from dataloader.DataBatcher import DataBatcher

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MF(BaseRecommender):
    def __init__(self, dataset, model_conf):
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = dataset.train_dict

        self.hidden_dim = model_conf['hidden_dim']
        self.batch_size = model_conf['batch_size']
        self.loss_function = model_conf['loss_function']
        self.learner = model_conf['learner']
        self.learning_rate = model_conf['learning_rate']
        self.reg = model_conf['reg']
        self.test_batch_size = model_conf['test_batch_size']

        self.eps = 1e-6

        self.item_popularity = self.get_item_popularity()

        # build graph
        self.build_graph()

    def _create_placeholders(self):
        with tf.name_scope("placeholders"):
            self.user_ph = tf.placeholder(dtype=tf.int32, shape=[None])
            self.item_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items])

    def _create_variables(self):
        with tf.name_scope("variables"):
            self.U = tf.Variable(tf.random_uniform([self.num_users, self.hidden_dim], dtype=tf.float32))
            self.V = tf.Variable(tf.random_uniform([self.num_items, self.hidden_dim], dtype=tf.float32))

    def _create_inference(self):
        with tf.name_scope("inference"):
            self.u_emb = tf.nn.embedding_lookup(self.U, self.user_ph)
            self.output = tf.matmul(self.u_emb, self.V, transpose_b=True)

    def _create_loss(self):
        with tf.name_scope("loss"):
            # Perform loss function and apply regularization to weights.
            # Tensorflow l2 regularization multiply 0.5 to the l2 norm
            # Multiply 2 so that it is back in the same scale.
            # loss = Tool.pointwise_loss(self.loss_function, self.item_ph, self.output)
            ce = self.item_ph * tf.log(self.output + self.eps) + (1 - self.item_ph) * tf.log(1 - self.output + self.eps)
            loss = -tf.reduce_sum(ce)
            reg_loss = tf.nn.l2_loss(self.U) + tf.nn.l2_loss(self.V)
            self.loss = loss + 2 * self.reg * reg_loss

    def _create_optimizer(self):
        with tf.name_scope("learner"):
            # self.optimizer = Tool.optimizer(self.learner, self.loss, self.learning_rate)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()

    def train_model(self, sess, dataset, evaluator, early_stop, saver, logger, config):
        exp_config = config['Experiment']

        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # prepare dataset
        dataset.set_eval_data('valid')
        users = np.array(list(self.train_dict.keys())) \
            if self.dataset.split_type == 'user' \
            else np.arange(self.num_users)

        # dataset iterator
        train_dict = dataset.train_dict

        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            epoch_loss = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=True)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_user_idx in enumerate(batch_loader):
                batch_uid, batch_matrix = 0, np.zeros((len(batch_user_idx), self.num_items))
                for user_id in batch_user_idx:
                    values_by_user = train_dict[user_id]
                    for value in values_by_user:
                        batch_matrix[batch_uid, value[0]] = value[1]
                    batch_uid = batch_uid + 1
                batch_loss = self.train_model_per_batch(sess, batch_matrix, batch_user_idx)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                # evaluate model
                epoch_eval_start = time()
                dataset.set_eval_data('valid')
                valid_score = evaluator.evaluate(sess, self, dataset)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                # ------------------- For Research Experiment--------------------
                # dataset.set_eval_data('test')
                # test_score = evaluator.evaluate(sess, self, dataset)
                # valid_score_str += ['[Test]%s=%.4f' % (k, test_score[k]) for k in test_score]
                #
                # if early_stop.early_stop - early_stop.endure == 1:
                #     dataset.set_eval_data('valid')
                #     eval_target, output_mask = dataset.get_eval_data()
                #     eval_output = self.predict(sess, dataset)
                #     evaluator.evaluator.compute_tmp(eval_output, eval_target, output_mask)
                #     dataset.set_eval_data('test')
                #     eval_target, output_mask = dataset.get_eval_data()
                #     eval_output = self.predict(sess, dataset)
                #     evaluator.evaluator.compute_tmp(eval_output, eval_target, output_mask)
                # ------------------- For Research Experiment--------------------

                updated, should_stop = early_stop.step(valid_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered. epoch=%s %s' % (early_stop.best_epoch, early_stop.best_score))
                    break
                else:
                    '''
                    if saver:
                        saved = False
                        while not saved:
                            try:
                                saver.save(sess, os.path.join(log_dir, 'best_model.ckpt'), global_step=epoch,
                                        write_meta_graph=False)
                                saved = True
                            except:
                                pass
                    '''
                    dataset.set_eval_data('test')
                    test_score = evaluator.evaluate(sess, self, dataset)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += valid_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time, test_score

    def train_model_per_batch(self, sess, batch_matrix, batch_user_idx):
        feed_dict = {self.item_ph: batch_matrix,
                     self.user_ph: batch_user_idx}
        _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss

    def predict(self, sess, dataset):
        eval_input_dict = dataset.get_listwise_eval_input()
        eval_output = np.zeros(shape=(self.num_users, self.num_items))

        batch_size = self.test_batch_size

        users = np.array(list(eval_input_dict.keys())) \
            if self.dataset.split_type == 'user' \
            else np.arange(self.num_users)

        batch_loader = DataBatcher(users, batch_size=batch_size, drop_remain=False, shuffle=False)

        for b, batch_user_idx in enumerate(batch_loader):
            batch_uid, batch_matrix = 0, np.zeros((len(batch_user_idx), self.num_items))
            for user_id in batch_user_idx:
                values_by_user = eval_input_dict[user_id]
                for value in values_by_user:
                    batch_matrix[batch_uid, value[0]] = value[1]
                batch_uid = batch_uid + 1
            eval_output[batch_user_idx] = sess.run(self.output, feed_dict={self.item_ph: batch_matrix,
                                                                           self.user_ph: batch_user_idx})
        return eval_output


    def get_item_popularity(self):
        item_popularity = self.dataset.get_item_popularity()
        return item_popularity