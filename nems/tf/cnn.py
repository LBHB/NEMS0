"""Code for fitting LN network models based on Sam Norman-Haignere's cnn library"""

import itertools
import logging
import os

import numpy as np
import tensorflow as tf

from nems.tf import initializers

log = logging.getLogger(__name__)


def poisson(response, prediction):
    return tf.reduce_mean(prediction - response * tf.log(prediction + 1e-5), name='poisson')


def loss_tf_nmse_shrinkage(response, prediction):
    """NMSE with shrinkage"""
    return tf_nmse_shrinkage(response, prediction)


def loss_tf_nmse(response, prediction):
    """NMSE"""
    mE, sE = tf_nmse(response, prediction)

    return mE


def tf_nmse_shrinkage(response, prediction, shrink_factor=0.5, per_cell=False, thresh=False):
    """Calculates the normalized mean squared error, with an adjustment for error.

    Averages across the batches, but optionally can return a per cell error.

    :param response:
    :param prediction:
    :param float shrink_factor:
    :param bool per_cell: Whether to also average over cells or not
    :param bool thresh:
    :return: a "shrunk" normalized mean squared error
    """
    mE, sE = tf_nmse(response, prediction, per_cell)

    def shrink(mE, sE, shrink_factor, thresh):
        def shrink_all(mE, sE, shrink_factor, thresh):
            return tf_shrinkage(mE, sE, shrink_factor, thresh)

        def shrink_some(mE, sE, shrink_factor, thresh):
            mask_gt, mask_lt = mE >= 1, mE < 1
            # make zero where mE was > 1
            shrunk = tf_shrinkage(mE, sE, shrink_factor, thresh) * tf.dtypes.cast(mask_lt, mE.dtype)
            # add back in
            mE = shrunk + mE * tf.dtypes.cast(mask_gt, mE.dtype)
            return mE

        mE = tf.cond(tf.math.reduce_all(mE < 1), lambda: shrink_all(mE, sE, shrink_factor, thresh),
                     lambda: shrink_some(mE, sE, shrink_factor, thresh))
        return mE

    mE = tf.cond(tf.math.reduce_any(mE < 1), lambda: shrink(mE, sE, shrink_factor, thresh), lambda: mE)

    if per_cell:
        mE = tf.math.reduce_mean(mE)

    return mE


def tf_nmse(response, prediction, per_cell=False):
    """Calculates the normalized mean squared error across batches.

    Optionally can return an average per cell.

    :param response:
    :param prediction:
    :param per_cell: Whether to average across all cells or not
    :return: 2 tensors, one of the mean error, the other of the std of the error. If not per cell, then
     tensor is of shape (), else tensor if of shape (n_cells) (i.e. last dimension of the resp/pred tensor)
    """
    n_drop = response.shape[-2].value % 10
    if n_drop:
        # use slices to handle varying tensor shapes
        drop_slice = [slice(None) for i in range(len(response.shape))]

        # second last dim is time
        drop_slice[-2] = slice(None, -n_drop)
        drop_slice = tuple(drop_slice)

        _response = response[drop_slice]
        _prediction = prediction[drop_slice]
    else:
        _response = response
        _prediction = prediction

    if per_cell:
        _response = tf.transpose(_response, np.roll(np.arange(len(response.shape)), 1))
        _prediction = tf.transpose(_prediction, np.roll(np.arange(len(response.shape)), 1))

        _response = tf.reshape(_response, shape=(_response.shape[0], 10, -1))
        _prediction = tf.reshape(_prediction, shape=(_prediction.shape[0], 10, -1))
    else:
        _response = tf.reshape(_response, shape=(10, -1))
        _prediction = tf.reshape(_prediction, shape=(10, -1))

    squared_error = ((_response - _prediction) ** 2)
    nmses = tf.math.reduce_mean(squared_error, axis=-1) ** 0.5 / tf.math.reduce_std(_response, axis=-1)

    mE = tf.math.reduce_mean(nmses, axis=-1)
    sE = tf.math.reduce_std(nmses, axis=-1) / 10 ** 0.5

    return mE, sE


def tf_shrinkage(mE, sE, shrink_factor=0.5, thresh=False):
    """Adjusts the mean error based on the standard error"""
    mE = 1 - mE
    smd = tf.math.divide_no_nan(abs(mE), sE) / shrink_factor
    smd = 1 - smd ** -2

    if thresh:
        return 1 - mE * tf.dtypes.cast(smd > 1, mE.dtype)

    smd = smd * tf.dtypes.cast(smd > 0, smd.dtype)

    return 1 - mE * smd


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
        fn = initializers.weights_matrix
    elif distr == 'tnorm':
        fn = initializers.weights_tnorm
    elif distr == 'norm':
        fn = initializers.weights_norm
    elif distr == 'zeros':
        fn = initializers.weights_zeros
    elif distr == 'uniform':
        fn = initializers.weights_uniform
    elif distr == 'glorot_uniform':
        fn = initializers.weights_glorot_uniform
    elif distr == 'he_uniform':
        fn = initializers.weights_he_uniform
    else:
        raise NameError('No matching distribution')

    if type(distr) is np.ndarray:
        # TODO : break out to separate kern
        W = initializers.weights_matrix(distr)
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

        # dimensionality of features and data
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
        self.last_iter = None

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
        elif self.loss_type == 'nmse':
            self.loss = loss_tf_nmse(self.D, self.Y)
        elif self.loss_type == 'nmse_shrinkage':
            self.loss = loss_tf_nmse_shrinkage(self.D, self.Y)
        else:
            raise NameError(f'Loss must be "squared_error", "poisson", "nmse", or "nmse_shrinkage", not {self.loss_type}')

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

    def train(self, F, D, learning_rate=0.01, max_iter=300, eval_interval=30, batch_size=None,
              early_stopping_steps=5, early_stopping_tolerance=5e-4, print_iter=True, S=None):

        self.save()

        # samples used for training
        train_inds = np.arange(D.shape[0])

        # by default batch size equals the size of the training data
        if batch_size is None:
            batch_size = len(train_inds)

        log.info(f'Training with batch_size={batch_size}, LR={learning_rate}, max_iter={max_iter}, '
                 f'early_stopping_steps={early_stopping_steps}, early_stopping_tolerance={early_stopping_tolerance}.')

        with self.sess.as_default():

            # indices for this batch
            batch_inds = np.arange(0, batch_size)

            # evaluate loss before any training
            if len(self.train_loss) == 0:
                train_dict = self.feed_dict(F, D=D, S=S, inds=train_inds[batch_inds], learning_rate=learning_rate)
                self.train_loss.append(self.loss.eval(feed_dict=train_dict))
                self.iteration = [0]
            
            for i in itertools.count():
                if max_iter is not None and i > max_iter:
                    break

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

                    if print_iter:
                        log.info("%04d e=%8.7f, delta= %+8.7f", self.iteration[-1],
                                 train_loss, train_loss - self.best_loss)

                    # early stopping / saving
                    if early_stopping_steps > 0:
                        if self.best_loss > train_loss:
                            self.best_loss, self.best_loss_index = train_loss, len(self.train_loss)
                            self.save()

                        # early stopping for > 5 non improving iterations
                        # TODO: redundant with tolerance early stopping?
                        if self.best_loss_index < len(self.train_loss) - early_stopping_steps:
                            log.info('Best epoch > 5 iterations ago, stopping early!')
                            break

                        # early stopping for not exceeding tolerance
                        elif np.all(abs(np.diff(np.array(self.train_loss[-early_stopping_steps - 1:]))) <
                                    early_stopping_tolerance):
                            log.info('5 epochs without significant change, stopping early!')
                            break

                # update batch
                batch_inds = batch_inds + batch_size
                if batch_inds[-1] > len(train_inds) - 1:
                    np.random.shuffle(train_inds)
                    batch_inds = np.arange(batch_size)

            # load the best loss
            if early_stopping_steps > 0:
                self.load()

            # record the last iter step
            self.last_iter = i + 1

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
