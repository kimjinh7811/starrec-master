import os
import math
from time import time
'''
Reference: Yao Wu et al. ""Collaborative Denoising Auto-Encoders for Top-N Recommender Systems" in WSDM 2016
'''

import numpy as np
import tensorflow as tf

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
#import matplotlib.pyplot as plt

import copy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class VAE(BaseRecommender):
    def __init__(self,dataset, model_conf):
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = dataset.train_dict

        self.enc_dim = [self.num_items] + model_conf['enc_dim']
        self.dec_dim = model_conf['enc_dim'][::-1] + [self.num_items]
        self.dim = self.enc_dim + self.dec_dim[1:]
        self.hidden_dim = self.enc_dim[-1]

        self.batch_size = model_conf['batch_size']
        self.learning_rate = model_conf['learning_rate']
        self.reg = model_conf['reg']
        self.act = model_conf['act']
        self.test_batch_size = model_conf['test_batch_size']

        self.anneal_cap = model_conf['anneal_cap']
        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.update_count=0

        self.pos_noise_ratio = model_conf['pos_noise_ratio']
        self.neg_noise_ratio = model_conf['neg_noise_ratio']
        self.neg_noise_val = model_conf['neg_noise_val']

        self.pop_bias = model_conf['pop_bias']
        self.func_complexity = model_conf['func_complexity']

        #print('item_popularity')
        self.item_popularity = self.get_item_popularity()
        item_idx = np.arange(self.num_items)

        '''
        sorted_item_pop = np.sort(self.item_popularity)[::-1]
        plt.plot(item_idx, sorted_item_pop)
        plt.xlabel('item_idx')
        plt.ylabel('popularity')
        plt.title('popularity distribution')
        plt.show()
        '''

        #print('ranking popularity')
        self.rank_by_popularity()
        #print('sampling prob')
        self.set_sample_probability()

        self.build_graph()


    def _create_placeholders(self):
        with tf.name_scope("placeholders"):
            self.input_item_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.num_items])
            self.output_item_ph = tf.placeholder(dtype=tf.float32, shape=[None, self.dataset.num_items])
            self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
            self.is_training_ph = tf.placeholder_with_default(0., shape=None)
            self.anneal_ph = tf.placeholder_with_default(1., shape=None)

    def _create_variables(self):
        with tf.name_scope("variables"):
            self.weights_enc, self.biases_enc = [], []

            for i , (d_in, d_out) in enumerate(zip(self.enc_dim[:-1], self.enc_dim[1:])):
                if i == len(self.enc_dim[:-1]) - 1 :
                    d_out *= 2

                weight_key = "weight_enc_{}to{}".format(i, i+1)
                bias_key = "bias_enc{}".format(i+1)

                self.weights_enc.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()))

                self.biases_enc.append(tf.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.001)))

            self.weights_dec, self.biases_dec = [], []
            for i, (d_in, d_out) in enumerate(zip(self.dec_dim[:-1], self.dec_dim[1:])):

                weight_key = "weight_dec_{}to{}".format(i, i + 1)
                bias_key = "bias_dec{}".format(i + 1)

                self.weights_dec.append(tf.get_variable(
                    name=weight_key, shape=[d_in, d_out],
                    initializer=tf.contrib.layers.xavier_initializer()))

                self.biases_dec.append(tf.get_variable(
                    name=bias_key, shape=[d_out],
                    initializer=tf.truncated_normal_initializer(
                        stddev=0.001)))


            # Normalize the input and make some noise for input.

    def _create_inference(self):
        with tf.name_scope('inference'):
            h = tf.nn.l2_normalize(self.input_item_ph, 1)

            for i, (w, b) in enumerate(zip(self.weights_enc, self.biases_enc)):
                h = tf.matmul(h, w) + b

                if i < len(self.weights_enc) - 1:
                    h = self.act(self.act, h)

                else:
                    self.hidden = h
                    mu_enc = h[:, :self.hidden_dim]
                    logvar_enc = h[:, self.hidden_dim:]

            std_q = tf.exp(0.5 * logvar_enc)
            epsilon = tf.random_normal(tf.shape(std_q))
            sampled_z = mu_enc +  epsilon * std_q

            h = sampled_z

            for i, (w, b) in enumerate(zip(self.weights_dec, self.biases_dec)):
                h = tf.matmul(h, w) + b

                if i < len(self.weights_dec) - 1:
                    if self.act == 'tanh':
                        h = tf.nn.tanh(h)

            self.output = h

            self.KL = self.anneal_ph * tf.reduce_mean( 0.5 *
                tf.reduce_sum((tf.exp(logvar_enc) + tf.pow(mu_enc, 2) - 1 -logvar_enc), axis=1))

    def _create_loss(self):
        with tf.name_scope("loss"):

            # Perform loss function and apply regularization to weights.
            # Tensorflow l2 regularization multiply 0.5 to the l2 norm
            # Multiply 2 so that it is back in the same scale.

            loss = -tf.reduce_mean(tf.reduce_sum(self.output_item_ph * tf.nn.log_softmax(self.output, axis=-1)))
            reg_loss = sum(map(tf.nn.l2_loss, self.weights_enc)) + sum(map(tf.nn.l2_loss, self.weights_dec))

            #self.loss = loss + self.anneal_ph * self.KL + 2 * reg_loss
            self.loss = loss + self.KL + 2 * self.reg * reg_loss

    def _create_optimizer(self):
        with tf.name_scope("learner"):
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

            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            batch_loss = self.train_model_per_batch(sess, batch_matrix, batch_user_idx, anneal)
            epoch_loss += batch_loss

            self.update_count += 1

            if verbose and (b + 1) % verbose == 0:
                print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))

            epoch_train_time = time() - epoch_train_start
            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]



            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                dataset.set_eval_data('valid')
                # evaluate model
                epoch_eval_start = time()
                # valid_score = evaluate_model(sess, model, dataset, evaluator)
                valid_score = evaluator.evaluate(sess, self, dataset)
                valid_score_str = ['%s=%.4f' % (k, valid_score[k]) for k in valid_score]

                updated, should_stop = early_stop.step(valid_score, epoch)
                #print('after:', early_stop.best_score)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    '''
                    # Temporary solution
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


    def train_model_per_batch(self, sess, batch_matrix, batch_set_idx, anneal):
        copied_batch_matrix = copy.copy(batch_matrix)
        noised_batch_matrix = self.input_corruption(copied_batch_matrix, batch_set_idx)

        feed_dict = {self.input_item_ph: noised_batch_matrix,
                     self.output_item_ph:batch_matrix,
                     self.is_training_ph: 1,
                     self.anneal_ph: anneal}
        _, loss = sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
        return loss


    def predict(self, sess, dataset):
        eval_input_dict = dataset.get_listwise_eval_input()
        eval_output = np.zeros(shape=(self.num_users, self.num_items))

        batch_size = self.test_batch_size
        #users = np.arange(self.num_users)
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
            eval_output[batch_user_idx] = sess.run(self.output, feed_dict={self.input_item_ph: batch_matrix,
                                                                            self.output_item_ph: batch_matrix})
        return eval_output

    def get_item_popularity(self):
        item_popularity = self.dataset.get_item_popularity()
        return item_popularity

    def rank_by_popularity(self):

        # Rank all items. the more rating has, the less rank value.
        temp = np.argsort(self.item_popularity)[::-1]
        self.ranks = np.empty_like(temp)
        self.ranks[temp] = np.arange(self.num_items)

    def set_sample_probability(self):
        self.pos_noise_prob_dicts = {}
        self.neg_noise_prob_dicts = {}

        all_items = set(list(range(self.num_items)))
        self.neg_items = self.dataset.neg_dict
        self.pos_items = {}

        if self.func_complexity == 'linear':
            prob = np.arange(self.num_items, 0, -1)
        elif self.func_complexity == 'exp':
            prob = 1 / np.arange(1, self.num_items + 1, 1)

        # for all users, set the sampling probability of each item.
        for user, neg_item in self.neg_items.items():
            pos_item = list(all_items - set(neg_item))
            self.pos_items[user] = pos_item

            # In this case, noise lower the popularity bias.
            if self.pop_bias == 'high':
                # Normalize to sum of prob 1.
                self.pos_noise_prob_dicts[user] = prob[-self.ranks[pos_item] - 1] / np.sum(
                    prob[-self.ranks[pos_item] - 1])
                self.neg_noise_prob_dicts[user] = prob[self.ranks[neg_item]] / np.sum(prob[self.ranks[neg_item]])

            # In this case, noise higher the popularity bias
            elif self.pop_bias == 'low':
                # Normalize to sum of prob 1.
                self.pos_noise_prob_dicts[user] = prob[self.ranks[pos_item]] / np.sum(prob[self.ranks[pos_item]])
                self.neg_noise_prob_dicts[user] = prob[-self.ranks[neg_item] - 1] / np.sum(
                    prob[-self.ranks[neg_item] - 1])



    def input_corruption(self, batch_matrix, batch_set_idx):
        neg_sample_size = np.sum(np.ones(batch_matrix.shape) - batch_matrix, axis=1) * self.neg_noise_ratio
        pos_sample_size = np.sum(batch_matrix, axis=1) * self.pos_noise_ratio

        for i, (row_pos, row_neg, row_batch, idx) in enumerate(zip(pos_sample_size, neg_sample_size, batch_matrix, batch_set_idx)):
            # when neg_noise == True, noise is given.


            sample_size = int(row_neg)
            if self.pop_bias == 'random':
                sampled_neg_item = np.random.choice(self.neg_items[idx], sample_size)
            else:
                sampled_neg_item = np.random.choice(self.neg_items[idx], sample_size, p=self.neg_noise_prob_dicts[idx])
            batch_matrix[i][sampled_neg_item] = self.neg_noise_val

            # when pos_noise == True, noise is given.

            sample_size = int(row_pos)
            if self.pop_bias == 'random':
                sampled_pos_item = np.random.choice(self.pos_items[idx], sample_size)
            else:
                sampled_pos_item = np.random.choice(self.pos_items[idx], sample_size, p=self.pos_noise_prob_dicts[idx])
            batch_matrix[i][sampled_pos_item] = 0

        return batch_matrix
