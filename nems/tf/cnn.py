"""code for fitting LN network models based on Sam Norman-Haignere's cnn library"""

import numpy as np
import scipy.io as sio
import h5py
import tensorflow as tf
import pickle
import os
from matplotlib import pyplot as plt
import importlib
import inspect
from copy import deepcopy
import logging
log = logging.getLogger(__name__)


def weights_tnorm(shape, sig=0.1, seed=0):
    W = tf.Variable(tf.random.truncated_normal(
        shape, stddev=sig, mean=sig, seed=seed))
    return W

def weights_norm(shape, sig=0.1, seed=0):
    W = tf.Variable(tf.random.normal(shape, stddev=sig, mean=0, seed=seed))
    return W

def weights_zeros(shape, sig=0.1, seed=0):
    #W = tf.Variable(tf.random_normal(shape, stddev=0.001, mean=0, seed=seed))
    W = tf.Variable(tf.zeros(shape))
    return W

def weights_uniform(shape, minval=0, maxval=1, sig=0.1, seed=0):
    W = tf.Variable(tf.random_uniform(shape, minval=minval, maxval=maxval, seed=seed))
    return W

def weights_matrix(d):
    """ variable with specified initial values """
    W = tf.Variable(d)
    return W

def poisson(response, prediction):
    return tf.reduce_mean(prediction - response * tf.log(prediction + 1e-5), name='poisson')


def conv1d(k, Y):
    # convolution function
    # k: batch x time x feature
    # W: time x feature x output channel
    # returns: batch x time x output channel
    return tf.nn.conv1d(k, Y, stride=1, padding='SAME')


def act(name):
    if name == 'relu':
        fn = tf.nn.relu
    elif name == 'identity':
        fn = tf.identity
    else:
        raise NameError('No matching activation')
    return fn

def seed_to_randint(seed):
    np.random.seed(seed)
    return np.random.randint(1e9)

def kern2D(n_x, n_y, n_kern, sig, rank=None, seed=0, distr='tnorm'):
    """
    :param n_x: temporal bins (earliest to most recent)
    :param n_y: spectral bins (input channels)
    :param n_kern: number of filters
    :param sig:
    :param rank:
    :param seed:
    :param distr:
    :return:
    """
    log.info(distr)

    if type(distr) is np.ndarray:
        fn = weights_matrix
    elif distr == 'tnorm':
        fn = weights_tnorm
    elif distr == 'norm':
        fn = weights_norm
    elif distr == 'zeros':
        fn = weights_zeros
    elif distr == 'uniform':
        fn = weights_uniform
    else:
        raise NameError('No matching distribution')

    if type(distr) is np.ndarray:
        # TODO : break out to separate kern
        W = weights_matrix(distr)
    elif rank is None:
        log.info('seed: %d',seed_to_randint(seed))
        W = fn([n_x, n_y, n_kern], sig=sig, seed=seed_to_randint(seed))
    else:
        W_list = []
        for i in range(n_kern):
            log.info('seed for kern', i, ':', seed_to_randint(seed))
            W_list.append(tf.matmul(fn([n_x, rank], sig=sig, seed=seed_to_randint(seed)+i),
                                    fn([rank, n_y], sig=sig, seed=seed_to_randint(seed)+i+n_kern)))
            # A = tf.Variable(tf.orthogonal_initializer(gain=sig/10, dtype=tf.float32)([int(n_x), int(rank)]))
            # B = tf.Variable(tf.orthogonal_initializer(gain=sig/10, dtype=tf.float32)([int(rank), int(n_y)]))
            # W_list.append(tf.matmul(A, B))
            W_list[i] = tf.reshape(W_list[i], [int(n_x), int(n_y), 1])

        W = tf.concat(W_list, 2)

    log.info("W: %s", W.shape)

    return W



class Net:

    def __init__(self, data_dims, n_feats, sr_Hz, layers, loss_type='squared_error',
                 weight_scale=0.1, seed=0, log_dir=None, log_id=None, optimizer='Adam', n_states=0):

        # dimensionality of feature sand data
        self.n_stim = data_dims[0]
        self.n_tps_input = data_dims[1]
        self.n_resp = data_dims[2]
        self.n_feats = n_feats

        # place holders for input features and data
        if 'S' in layers[0].keys():
            self.S = layers[0]['S']
        if 'X' in layers[0].keys():
            self.F = layers[0]['X']
        else:
            self.F = tf.compat.v1.placeholder(
                'float32', shape=[None, self.n_tps_input, self.n_feats])
        self.D = tf.compat.v1.placeholder(
            'float32', shape=[None, self.n_tps_input, self.n_resp])

        # other parameters
        self.weight_scale = weight_scale
        self.seed = seed

        # layer parameters
        self.layers = layers
        self.n_layers = len(layers)
        for i in range(self.n_layers):
            self.layers[i]['time_win_smp'] = np.int32(
                np.round(sr_Hz * self.layers[i]['time_win_sec']))
        assert(self.layers[self.n_layers - 1]['n_kern'] == self.n_resp)

        # other misc parameters
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.iteration = []
        self.learning_rate = tf.compat.v1.placeholder('float32')
        self.best_loss = 1e100
        self.best_loss_index = 0

        # directory and file info
        if log_dir is None:
            self.log_dir = '/Users/svnh2/Desktop/projects/scratch'
        elif log_dir[-1] == '/':
            self.log_dir = log_dir[:-1]
        else:
            self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if log_id is None:
            self.log_id = 'seed' + str(seed)
        else:
            self.log_id = log_id

    def initialize(self):

        log.info('Net.initialize(): setting output, loss, optimizer, globals, tf session')
        self.Y = self.layers[self.n_layers - 1]['Y'][:, 0:self.n_tps_input, :]

        # loss
        if self.loss_type == 'squared_error':
            self.loss = tf.reduce_mean(tf.square(self.D - self.Y)) / tf.reduce_mean(tf.square(self.D))
        elif self.loss_type == 'poisson':
            self.loss = poisson(self.D, self.Y)
        else:
            raise NameError('Loss must be squared_error or poisson')

        # gradient optimizer
        if self.optimizer == 'Adam':
            self.train_step = tf.compat.v1.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'GradientDescent':
            self.train_step = tf.compat.v1.train.GradientDescentOptimizer(
                self.learning_rate).minimize(self.loss)
        elif self.optimizer == 'RMSProp':
            self.train_step = tf.compat.v1.train.RMSPropOptimizer(
                self.learning_rate).minimize(self.loss)
        else:
            raise NameError('No matching optimizer')

        # initialize global variables
        session_conf = tf.compat.v1.ConfigProto(
             intra_op_parallelism_threads=1,
             inter_op_parallelism_threads=1)
        self.sess = tf.compat.v1.Session(config=session_conf)

        #log.info('Initialize variables')
        self.sess.run(tf.compat.v1.global_variables_initializer())

        #log.info('Initialize saver')
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1)

    def parse_layers(self, initialize=True):
        """deprecated for use in NEMS. replaced by modelspec2tf"""
        raise Warning("DEPRECATED?")
        log.info('Building CNN Model (Loc 0)')

        self.W = []
        self.b = []
        self.L = []
        for i in range(self.n_layers):

            if i == 0:
                X = self.F
                n_input_feats = self.n_feats

            else:
                X = self.layers[i - 1]['Y']
                n_input_feats = np.int32(X.shape[2])

            if self.layers[i]['type'] == 'conv':

                log.info('Adding conv layer')

                # pad input to ensure causality
                pad_size = np.int32(
                    np.floor(self.layers[i]['time_win_smp'] / 2))
                X_pad = tf.pad(X, [[0, 0], [pad_size, 0], [0, 0]])

                if self.layers[i].get('init_W', None) is not None:
                    self.layers[i]['W'] = tf.Variable(self.layers[i]['init_W'])
                    self.layers[i]['b'] = tf.Variable(self.layers[i]['init_b'])
                else:
                    self.layers[i]['W'] = kern2D(self.layers[i]['time_win_smp'], n_input_feats,
                                                 self.layers[i]['n_kern'],
                                                 self.weight_scale, seed=seed_to_randint(self.seed) + i,
                                                 rank=self.layers[i]['rank'],
                                                 distr='norm')
                    self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                        self.weight_scale,
                                                        seed=seed_to_randint(self.seed) + i + self.n_layers,
                                                        distr='norm'))

                log.info("W shape: %s", self.layers[i]['W'].shape)
                log.info("X_pad shape: %s", X_pad.shape)
                #self.layers[i]['Y'] = act(self.layers[i]['act'])(
                #    conv1d(X_pad, self.layers[i]['W']) + self.layers[i]['b'])
                self.layers[i]['Y'] = act(self.layers[i]['act'])(
                    tf.nn.conv1d(X_pad, self.layers[i]['W'], stride=1, padding='SAME') + self.layers[i]['b'])
                log.info("Y shape: %s", self.layers[i]['Y'].shape)
            elif self.layers[i]['type'] == 'conv_bank_1d':

                log.info('Adding conv_bank (alt) layer')
                # split inputs into the different kernels
                n_input_chans = int(n_input_feats / self.layers[i]['n_kern'])
                rate = self.layers[i].get('rate', 1)

                # pad input to ensure causality, insert placeholder dim on axis=1
                pad_size = np.int32((self.layers[i]['time_win_smp']-1)*rate)
                X_pad = tf.expand_dims(tf.pad(X, [[0, 0], [pad_size,0], [0, 0]]), 1)

                if self.layers[i].get('init_W', None) is not None:
                    self.layers[i]['W'] = tf.Variable(self.layers[i]['init_W'])
                    self.layers[i]['b'] = tf.Variable(self.layers[i]['init_b'])
                else:
                    self.layers[i]['W'] = weights_norm(
                        [1, self.layers[i]['time_win_smp'], self.layers[i]['n_kern'], 1],
                        sig=self.weight_scale, seed=seed_to_randint(self.seed)+i)
                    self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                        self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers,
                                                        distr='norm'))

                log.info("W shape: %s", self.layers[i]['W'].shape)
                log.info("X_pad shape: %s", X_pad.shape)

                #import pdb
                #pdb.set_trace()

                tY = tf.squeeze(tf.nn.depthwise_conv2d(
                    X_pad, self.layers[i]['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, rate]),
                    axis=1)

                log.info("tY shape: %s", tY.shape)
                self.layers[i]['Y'] = act(self.layers[i]['act'])(tY + self.layers[i]['b'])

                log.info("Y shape: %s", self.layers[i]['Y'].shape)

            elif self.layers[i]['type'] == 'conv_bank':

                log.info('Adding conv_bank layer')
                # split inputs into the different kernels
                n_input_chans = int(n_input_feats / self.layers[i]['n_kern'])

                # pad input to ensure causality
                pad_size = np.int32(self.layers[i]['time_win_smp'] - 1)
                X_pad = tf.expand_dims(tf.pad(X, [[0, 0], [pad_size,0], [0, 0]]), 3)

                if self.layers[i].get('init_W', None) is not None:
                    self.layers[i]['W'] = tf.Variable(self.layers[i]['init_W'])
                    self.layers[i]['b'] = tf.Variable(self.layers[i]['init_b'])
                else:
                    self.layers[i]['W'] = weights_norm([self.layers[i]['time_win_smp'], 1,
                                                1, self.layers[i]['n_kern']], sig=self.weight_scale,
                                                seed=seed_to_randint(self.seed)+i)
                    self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                        self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers,
                                                        distr='norm'))

                log.info("W shape: %s", self.layers[i]['W'].shape)
                log.info("X_pad shape: %s", X_pad.shape)
                #self.layers[i]['Y'] = act(self.layers[i]['act'])(
                #    tf.reverse(tf.squeeze(tf.nn.conv2d(X_pad, self.layers[i]['W'], strides=[1, 1, 1, 1], padding='VALID'),
                #               axis=3), axis=[2]) + self.layers[i]['b'])
                tY = tf.matrix_diag_part(
                    tf.nn.conv2d(X_pad, self.layers[i]['W'], strides=[1, 1, 1, 1], padding='VALID'))

                log.info("tY shape: %s", tY.shape)

                self.layers[i]['Y'] = act(self.layers[i]['act'])(tY + self.layers[i]['b'])

                log.info("Y shape: %s", self.layers[i]['Y'].shape)

            elif self.layers[i]['type'] == 'dlog':

                log.info('Adding dlog layer')
                if self.layers[i].get('init_b', None) is not None:
                    self.layers[i]['b'] = tf.Variable(self.layers[i]['init_b'])
                    #self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                    #                                    self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers,
                    #                                    distr=self.layers[i]['init_b']))
                else:
                    self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                        self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers,
                                                        distr='tnorm'))
                # clip b at +/-2 to avoid huge compression/expansion
                self.layers[i]['eb'] = tf.pow(tf.constant(10, dtype=tf.float32),
                                              tf.clip_by_value(self.layers[i]['b'], -2, 2))
                self.layers[i]['Y'] = tf.log((X + self.layers[i]['eb']) / self.layers[i]['eb'])

            elif self.layers[i]['type'] == 'stp':
                log.info('Loc stp')

                if self.layers[i].get('init_u', None) is not None:
                    self.layers[i]['u'] = tf.Variable(self.layers[i]['init_u'])
                    self.layers[i]['tau'] = tf.Variable(self.layers[i]['init_tau'])
                else:
                    self.layers[i]['u'] = kern2D(1, 1, self.layers[i]['n_kern'],
                                                 self.weight_scale, seed=seed_to_randint(self.seed) + i + self.n_layers,
                                                 distr='uniform')
                    self.layers[i]['tau'] = kern2D(1, 1, self.layers[i]['n_kern'],
                                                 self.weight_scale,
                                                 seed=seed_to_randint(self.seed) + 20 + i + self.n_layers,
                                                 distr='uniform')

                # input (X) is output (Y) of previous layer
                # di[i, tt - 1]  # previous time bin depression
                # delta[tt] = (1 - di[i, tt - 1]) / tau[i] - u[i] * di[i, tt - 1] * X[i, tt - 1]
                # delta[tt] = 1/tau[i] + di[i, tt-1] * (-1/tau[i] - u[i] * X[i, tt-1])
                # di[i, tt] = di[i, tt - 1] + delta[tt]
                # di[i, tt] = di[i, tt - 2] + delta[tt-1] + delta[tt]
                # if di[i, tt] < 0:
                #    di[i, tt] = 0
                # Y[i, tt] *= X[i, tt] * di[i, tt]
                self.layers[i]['Wraw'] = tf.exp(-0.5 * tf.square((tf.reshape(
                    tf.range(0, 1, 1 / n_input_feats, dtype=tf.float32), [1, n_input_feats, 1]) - self.layers[i]['m']) /
                                                                 (self.layers[i]['s'] / 10)))
                self.layers[i]['W'] = self.layers[i]['Wraw'] / tf.reduce_sum(self.layers[i]['Wraw'], axis=1)
                self.layers[i]['Y'] = act(self.layers[i]['act'])(conv1d(X, self.layers[i]['W']))


            elif self.layers[i]['type'] == 'reweight':

                log.info('Adding reweight layer')

                if self.layers[i].get('init_W', None) is not None:
                    self.layers[i]['W'] = tf.Variable(self.layers[i]['init_W'])
                    self.layers[i]['b'] = tf.Variable(self.layers[i]['init_b'])
                    #self.layers[i]['W'] = kern2D(1, n_input_feats, self.layers[i]['n_kern'],
                    #                             self.weight_scale, seed=seed_to_randint(self.seed)+i,
                    #                             distr=self.layers[i]['init_W'])
                    #self.layers[i]['b'] = kern2D(1, 1, self.layers[i]['n_kern'],
                    #                             self.weight_scale, seed=seed_to_randint(self.seed) + i,
                    #                             distr=self.layers[i]['init_b'])
                else:
                    self.layers[i]['W'] = kern2D(1, n_input_feats, self.layers[i]['n_kern'],
                                                 self.weight_scale, seed=seed_to_randint(self.seed)+i,
                                                 distr='norm')
                    self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                        self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers,
                                                        distr='norm'))
                self.layers[i]['Y'] = act(self.layers[i]['act'])(conv1d(X, self.layers[i]['W']) + self.layers[i]['b'])

            elif self.layers[i]['type'] == 'reweight-positive':

                log.info('Loc reweight-positive')

                self.layers[i]['W'] = tf.abs(kern2D(1, n_input_feats, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i,
                                                    distr='norm'))
                self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i,
                                                    distr='norm'))
                self.layers[i]['Y'] = act(self.layers[i]['act'])(conv1d(X, self.layers[i]['W']) + self.layers[i]['b'])

            elif self.layers[i]['type'] == 'reweight-positive-zero':

                log.info('Loc reweight-positive-zero')
                self.layers[i]['W'] = tf.abs(kern2D(1, n_input_feats, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i,
                                                    distr='tnorm'))
                self.layers[i]['Y'] = act(self.layers[i]['act'])(conv1d(X, self.layers[i]['W']))

            elif self.layers[i]['type'] == 'reweight-gaussian':

                log.info('Adding reweight-gaussian layer')

                if self.layers[i].get('init_m', None) is not None:
                    self.layers[i]['m'] = tf.Variable(self.layers[i]['init_m'])
                    self.layers[i]['s'] = tf.Variable(self.layers[i]['init_s'])
                    #self.layers[i]['m'] = kern2D(1, 1, self.layers[i]['n_kern'],
                    #                             self.weight_scale,
                    #                             seed=seed_to_randint(self.seed) + i + self.n_layers,
                    #                             distr=self.layers[i]['init_m'])
                    #self.layers[i]['s'] = kern2D(1, 1, self.layers[i]['n_kern'],
                    #                             self.weight_scale,
                    #                             seed=seed_to_randint(self.seed) + i + self.n_layers,
                    #                             distr=self.layers[i]['init_s'])
                else:
                    self.layers[i]['m'] = kern2D(1, 1, self.layers[i]['n_kern'],
                                                 self.weight_scale, seed=seed_to_randint(self.seed) + i + self.n_layers,
                                                 distr='uniform')
                    self.layers[i]['s'] = kern2D(1, 1, self.layers[i]['n_kern'],
                                                 self.weight_scale, seed=seed_to_randint(self.seed) + 20 + i + self.n_layers,
                                                 distr='uniform')
                self.layers[i]['Wraw'] = tf.exp(-0.5 * tf.square((tf.reshape(tf.range(0, 1, 1/n_input_feats, dtype=tf.float32), [1, n_input_feats, 1])-self.layers[i]['m'])/
                                                (self.layers[i]['s'] / 10)))
                self.layers[i]['W'] = self.layers[i]['Wraw'] / tf.reduce_sum(self.layers[i]['Wraw'], axis=1)
                self.layers[i]['Y'] = act(self.layers[i]['act'])(conv1d(X, self.layers[i]['W']))

            elif self.layers[i]['type'] == 'normalization':

                # pad input to ensure causality
                pad_size = np.int32(
                    np.floor(self.layers[i]['time_win_smp'] / 2))
                X_pad = tf.pad(X, [[0, 0], [pad_size, 0], [0, 0]])

                self.layers[i]['W'] = tf.abs(kern2D(self.layers[i]['time_win_smp'], n_input_feats, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i, rank=self.layers[i]['rank']))
                self.layers[i]['b'] = tf.abs(kern2D(1, 1, self.layers[i]['n_kern'],
                                                    self.weight_scale, seed=seed_to_randint(self.seed)+i+self.n_layers))
                self.layers[i]['Y'] = X_pad / (act(self.layers[i]['act'])(conv1d(tf.abs(X_pad), self.layers[i]['W'])) + self.layers[i]['b'])

                #import pdb; pdb.set_trace()

            else:
                raise NameError('No matching layer type')

        if initialize:
            self.initialize()

    def feed_dict(self, F, D=None, S=None, inds=None, learning_rate=None):

        if inds is None:
            inds = np.arange(F.shape[0])

        d = {self.F: F[inds, :, :]}

        if not (D is None):
            d[self.D] = D[inds, :, :]
        if not (S is None):
            d[self.S] = S[inds, :, :]

        if not (learning_rate is None):
            d[self.learning_rate] = learning_rate

        return d

    def save(self):
        fname = self.log_dir + '/' + self.log_id + '-model.ckpt'
        self.saver.save(self.sess, fname)

    def load(self):
        fname = self.log_dir + '/' + self.log_id + '-model.ckpt'
        self.saver.restore(self.sess, fname)

    def train(self, F, D, learning_rate=0.5, max_iter=300, eval_interval=30, batch_size=None,
              train_val_test=None, early_stopping_steps=5, print_iter=True, S=None):

        self.save()
        if train_val_test is None:
            train_val_test = np.zeros(D.shape[0])

        # samples used for training, validation, and testing
        train_inds = np.where(train_val_test == 0)[0]
        val_inds = np.where(train_val_test == 1)[0]
        test_inds = np.where(train_val_test == 2)[0]

        # dictionaries for validation and test
        # can initialize them now, because they are not batch dependent
        if len(val_inds) > 0:
            val_dict = self.feed_dict(F, D=D, S=S, inds=val_inds)
        if len(test_inds) > 0:
            test_dict = self.feed_dict(F, D=D, S=S, inds=test_inds)

        # by default batch size equals the size of the training data
        if batch_size is None:
            batch_size = len(train_inds)

        with self.sess.as_default():

            # indices for this batch
            batch_inds = np.arange(0, batch_size)

            # evaluate loss before any training
            if len(self.train_loss) == 0:
                train_dict = self.feed_dict(F, D=D, S=S, inds=train_inds[batch_inds],
                                            learning_rate=learning_rate)
                self.train_loss.append(self.loss.eval(feed_dict=train_dict))
                if len(val_inds) > 0:
                    self.val_loss.append(self.loss.eval(feed_dict=val_dict))
                if len(test_inds) > 0:
                    self.test_loss.append(self.loss.eval(feed_dict=test_dict))
                self.iteration = [0]
            
            not_improved = 0
            for i in range(max_iter):

                # update
                train_dict = self.feed_dict(F, D=D, S=S, inds=train_inds[batch_inds],
                                            learning_rate=learning_rate)
                self.train_step.run(feed_dict=train_dict)

                # evaluate loss
                if np.mod(i + 1, eval_interval) == 0:
                    self.iteration.append(
                        self.iteration[len(self.iteration) - 1] + eval_interval)
                    train_loss = self.loss.eval(feed_dict=train_dict)
                    self.train_loss.append(train_loss)
                    if len(val_inds) > 0:
                        validation_loss = self.loss.eval(feed_dict=val_dict)
                        stop_loss = validation_loss
                        self.val_loss.append(validation_loss)
                    if len(test_inds) > 0:
                        self.test_loss.append(
                            self.loss.eval(feed_dict=test_dict))
                    if print_iter:
                        log.info("%d e=%.7f v=%.7f", self.iteration[len(self.iteration) - 1],
                                 train_loss, validation_loss)

                    # early stopping / saving
                    if early_stopping_steps > 0:

                        # prefer to use validation loss
                        if len(val_inds) > 0:
                            stop_loss = validation_loss
                        else:
                            stop_loss = train_loss

                        # check if improved, if so save
                        if stop_loss < self.best_loss:
                            self.best_loss = stop_loss
                            self.best_loss_index = len(self.val_loss) - 1
                            self.save()
                            not_improved = 0
                        else:
                            not_improved += 1

                        # check if we should stop
                        if not_improved == early_stopping_steps:
                            not_improved = 0
                            break

                # update batch
                batch_inds = batch_inds + batch_size
                if batch_inds[-1] > len(train_inds):
                    np.random.shuffle(train_inds)
                    batch_inds = np.arange(batch_size)

            # load the best loss
            if early_stopping_steps > 0:
                self.load()

    def predict(self, F, S=None, sess=None):

        if sess is None:
            sess = self.sess

        with sess.as_default():

            return self.Y.eval(feed_dict=self.feed_dict(F, S=S))

    def eval_to_layer(self, F=None, i=None, S=None, sess=None):

        if i is None:
            i = len(self.layers)-1
        if sess is None:
            sess = self.sess

        with sess.as_default():
            return self.layers[i]['Y'].eval(feed_dict=self.feed_dict(F, S=S))

    def layer_vals(self, sess=None):
        """
        get matrix values out TF variables
        :param sess:
        :return:
        """
        if sess is None:
            sess = self.sess

        with sess.as_default():

            layers = []
            for i in range(self.n_layers):
                layer = {}
            #    for j in self.layers[i].keys():
            #        layer[j] = self.layers[i][j].eval()
                if 'W' in self.layers[i]:
                    layer['W'] = self.layers[i]['W'].eval()
                if 'b' in self.layers[i]:
                    layer['b'] = self.layers[i]['b'].eval()
                if 'm' in self.layers[i]:
                    layer['m'] = self.layers[i]['m'].eval()
                if 's' in self.layers[i]:
                    layer['s'] = self.layers[i]['s'].eval()
                if 'g' in self.layers[i]:
                    layer['g'] = self.layers[i]['g'].eval()
                if 'd' in self.layers[i]:
                    layer['d'] = self.layers[i]['d'].eval()
                if 'gain' in self.layers[i]:
                    layer['gain'] = self.layers[i]['gain'].eval()
                if 'f1' in self.layers[i]:
                    layer['f1'] = self.layers[i]['f1'].eval()
                if 'delay' in self.layers[i]:
                    layer['delay'] = self.layers[i]['delay'].eval()
                if 'tau' in self.layers[i]:
                    layer['tau'] = self.layers[i]['tau'].eval()
                if 'base' in self.layers[i]:
                    layer['base'] = self.layers[i]['base'].eval()
                if 'amplitude' in self.layers[i]:
                    layer['amplitude'] = self.layers[i]['amplitude'].eval()
                if 'kappa' in self.layers[i]:
                    layer['kappa'] = self.layers[i]['kappa'].eval()
                if 'shift' in self.layers[i]:
                    layer['shift'] = self.layers[i]['shift'].eval()
                layers.append(layer)

            return layers


class MultiNet(Net):

    def __init__(self, nets, log_dir=None, log_id=None):

        self.nets = nets
        self.weights = np.ones(len(nets)) / len(nets)
        self.D = nets[0].D

        # other misc parameters
        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.iteration = []
        self.learning_rate = tf.placeholder('float32')
        self.best_loss = 1e100
        self.best_loss_index = 0

        # directory and file info
        if log_dir is None:
            self.log_dir = '/Users/svnh2/Desktop/projects/scratch'
        else:
            self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if log_id is None:
            self.log_id = 'seed' + str(self.nets[0].seed)

    def build(self):

        for i in range(len(self.nets)):
            self.nets[i].build(initialize=False)
        #             if i == 0:
        #                 self.Y = self.nets[i].Y * self.weights[i]
        #             else:
        #                 self.Y = self.Y + self.nets[i].Y * self.weights[i]

        self.Y = self.nets[0].Y * self.weights[0] + \
            self.nets[1].Y * self.weights[1]

        # loss
        self.loss = tf.reduce_mean(tf.square(self.D - self.Y))

        # gradient optimizer
        self.train_step = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # initialize global variables
        self.initialize()

    def feed_dict(self, F, D=None, inds=None, learning_rate=None):

        assert(len(self.nets) == len(F))

        if inds is None:
            inds = np.arange(F[0].shape[0])

        d = {}
        for i in range(len(F)):
            d[self.nets[i].F] = F[i][inds, :, :]

        if not (D is None):
            for i in range(len(self.nets)):
                d[self.nets[i].D] = D[inds, :, :]

        if not (learning_rate is None):
            d[self.learning_rate] = learning_rate

        return d

    def layer_vals(self):

        nets = []
        for i in range(len(self.nets)):
            nets.append(self.nets[i].layer_vals(sess=self.sess))

        return nets
