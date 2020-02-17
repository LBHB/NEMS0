"""
Tools for mapping NEMS modelspecs to and from Tensorflow CNNs
Uses Sam Norman-Haignere's CNN library as a front end for TF

"""
import tensorflow as tf
import numpy as np
import time
import copy

import nems
import nems.modelspec as ms
import nems.tf.cnn as cnn
from nems.initializers import init_dexp, init_logsig, init_relsat
import nems.metrics.api as nmet
import nems.utils
import logging
log = logging.getLogger(__name__)

modelspecs_dir = nems.get_setting('NEMS_RESULTS_DIR')


def modelspec2tf(modelspec, tps_per_stim=550, feat_dims=1, data_dims=1, state_dims=0, fs=100,
                 net_seed=1, weight_scale=0.1, use_modelspec_init=True):
    """
    Convert NEMS modelspec to TF layers.
    Translations:
        wc -> reweight
        fir -> conv
        firbank
        do
        dexp
        dlog
        relu
        lvl
        wg.g
    :param modelspec:
    :param feat_dims: [n_stim, n_tps_per_stim, n_feats] (tps = timepoints per stim, n_feats = input channels)
    :param data_dims: [n_stim, n_tps_per_stim, n_resp] (n_resp = output channels)
    :param fs: sampling rate (used?)
    :param net_seed:
    :param use_modelspec_init: if True, initialize with existing phi. Otherwise random init.
    :return: layers: list of layers compatible with cnn.net
    """
    F = tf.placeholder('float32', shape=[None, tps_per_stim, feat_dims])
    if state_dims>0:
        S = tf.placeholder('float32', shape=[None, tps_per_stim, state_dims])
    D = tf.placeholder('float32', shape=[None, tps_per_stim, data_dims])

    layers = []
    for i, m in enumerate(modelspec):
        log.info('modelspec2tf: ' + m['fn'])

        layer = {}
        # input to each layer is output of previous layer
        if i == 0:
            layer['X'] = F
            if state_dims > 0:
                layer['S'] = S
            layer['D'] = D
        else:
            layer['X'] = layers[-1]['Y']
            if 'L' in layers[-1].keys():
                layer['L'] = layers[-1]['L']

        n_input_feats = np.int32(layer['X'].shape[2])
        # default integration time is one bin
        layer['time_win_smp'] = 1 # default

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            layer['type'] = 'relu'
            c = -modelspec[i]['phi']['offset'].astype('float32').T
            layer['n_kern'] = c.shape[1]
            log.info('relu init %s', c)
            if use_modelspec_init:
                layer['b'] = tf.Variable(np.reshape(c, (1, c.shape[0], c.shape[1])))
            else:
                layer['b'] = tf.abs(cnn.kern2D(1, c.shape[0], c.shape[1],
                                               weight_scale, seed=net_seed,
                                               distr='norm'))
            layer['Y'] = cnn.act('relu')(layer['X'] + layer['b'])

        elif 'levelshift' in m['fn']:
            layer['type'] = 'offset'
            c = m['phi']['level'].astype('float32').T
            layer['n_kern'] = c.shape[1]

            if use_modelspec_init:
                layer['b'] = tf.Variable(np.reshape(c, (1, c.shape[0], c.shape[1])))
            else:
                layer['b'] = tf.abs(cnn.kern2D(1, c.shape[0], c.shape[1],
                                               weight_scale, seed=net_seed,
                                               distr='norm'))
            layer['Y'] = cnn.act('identity')(layer['X'] + layer['b'])

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            layer['type'] = 'dlog'

            c = m['phi']['offset'].astype('float32').T
            layer['n_kern'] = c.shape[1]

            if use_modelspec_init:
                layer['b'] = tf.Variable(np.reshape(c, (1, c.shape[0], c.shape[1])))
            else:
                layer['b'] = tf.abs(cnn.kern2D(1, c.shape[0], c.shape[1],
                                               weight_scale, seed=cnn.seed_to_randint(net_seed)+i,
                                               distr='tnorm'))

            # clip b at +/-2 to avoid huge compression/expansion
            layer['eb'] = tf.pow(tf.constant(10, dtype=tf.float32),
                                          tf.clip_by_value(layer['b'], -2, 2))
            layer['Y'] = tf.math.log((layer['X'] + layer['eb']) / layer['eb'])

        elif m['fn'] in ['nems.modules.nonlinearity.double_exponential']:
            layer['type'] = 'dexp'
            layer['n_kern'] = m['phi']['base'].size
            s = (1, layer['n_kern'], 1)

            if use_modelspec_init:
                layer['base'] = tf.Variable(np.reshape(m['phi']['base'].astype('float32'), s))
                layer['amplitude'] = tf.Variable(np.reshape(m['phi']['amplitude'].astype('float32'), s))
                layer['shift'] = tf.Variable(np.reshape(m['phi']['shift'].astype('float32'), s))
                layer['kappa'] = tf.Variable(np.reshape(m['phi']['kappa'].astype('float32'), s))
            else:
                log.info('Using TF rand for double exponential')
                layer['base'] = tf.Variable(tf.random.uniform(
                    s, minval=0, maxval=1, seed=cnn.seed_to_randint(net_seed + i)))
                layer['amplitude'] = tf.Variable(tf.random.uniform(
                    s, minval=0.1, maxval=0.5, seed=cnn.seed_to_randint(net_seed + 20 + i)))
                layer['shift'] = tf.Variable(tf.random.normal(
                    s, stddev=weight_scale, mean=0, seed=cnn.seed_to_randint(net_seed + 40 + i)))
                layer['kappa'] = tf.Variable(tf.random.uniform(
                    s, minval=0, maxval=2, seed=cnn.seed_to_randint(net_seed + 60 + i)))

            # base + amplitude * exp(  -exp(np.array(-exp(kappa)) * (x - shift))  )
            layer['Y'] = layer['base'] + layer['amplitude'] * tf.math.exp(
                -tf.math.exp(-tf.math.exp(layer['kappa']) * (layer['X'] - layer['shift'])))

        elif m['fn'] in ['nems.modules.fir.basic']:
            layer['type'] = 'conv'
            layer['time_win_smp'] = m['phi']['coefficients'].shape[1]
            pad_size = np.int32(np.floor(layer['time_win_smp']-1))
            X_pad = tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]])

            c = np.fliplr(m['phi']['coefficients']).astype('float32').T
            c = np.reshape(c, (c.shape[0], c.shape[1], 1))
            if np.sum(np.abs(c)) == 0:
                c = np.ones(c.shape, dtype='float32')/10
            layer['n_kern'] = 1
            chan_count = c.shape[1]
            if use_modelspec_init:
                layer['W'] = tf.Variable(c)
            else:
                layer['W'] = cnn.kern2D(layer['time_win_smp'], chan_count, 1,
                                        weight_scale, seed=net_seed, distr='norm')
            #log.info("W shape: %s", layer['W'].shape)
            #log.info("X_pad shape: %s", X_pad.shape)
            layer['Y'] = tf.nn.conv1d(X_pad, layer['W'], stride=1, padding='VALID')
            #log.info("Y shape: %s", layer['Y'].shape)

        elif m['fn'] in ['nems.modules.fir.damped_oscillator']:
            layer['type'] = 'conv'
            layer['time_win_smp'] = m['fn_kwargs']['n_coefs']
            layer['rate'] = m['fn_kwargs'].get('rate', 1)
            cross_channels = m['fn_kwargs'].get('cross_channels', False)
            bank_count = m['fn_kwargs']['bank_count']
            layer['n_kern'] = bank_count

            chan_count = int(m['phi']['f1s'].size / bank_count)
            in_chan_count = int(layer['X'].shape[2])
            pad_size = np.int32(np.floor(layer['time_win_smp']-1))
            if cross_channels and (bank_count > 1):
                raise Warning('cross_channels not supported for bank_count>1')
                cross_channels=False

            if cross_channels:
                s = (1, 1, 1, chan_count*bank_count)
            elif bank_count == 1:
                # revert to simple conv1d, traditional FIR filter
                s = (1, chan_count, 1)
            elif in_chan_count == bank_count*chan_count:
                # break up inputs to feed into each bank
                s = (1, 1, chan_count*bank_count, 1)
            else:
                # apply each filter to all inputs
                if chan_count != in_chan_count:
                    raise ValueError('Either chan_count*bank_count or chan_count must equal in_chan_count')
                s = (1, 1, chan_count, bank_count)

            if use_modelspec_init:
                layer['f1'] = tf.Variable(np.reshape(m['phi']['f1s'].astype('float32'), s))
                layer['tau'] = tf.Variable(np.reshape(m['phi']['taus'].astype('float32'), s))
                layer['delay'] = tf.Variable(np.reshape(m['phi']['delays'].astype('float32'), s))
                layer['gain'] = tf.Variable(np.reshape(m['phi']['gains'].astype('float32'), s))
            else:
                log.info('Using TF rand for damped oscillator')
                layer['f1'] = tf.Variable(tf.random.uniform(
                    s, minval=0, maxval=1, seed=cnn.seed_to_randint(net_seed + i)))
                layer['tau'] = tf.Variable(tf.random.uniform(
                    s, minval=0.1, maxval=0.5, seed=cnn.seed_to_randint(net_seed + 20 + i)))
                layer['gain'] = tf.Variable(tf.random.normal(
                    s, stddev=weight_scale, mean=0, seed=cnn.seed_to_randint(net_seed + 40 + i)))
                layer['delay'] = tf.Variable(tf.random.uniform(
                    s, minval=0, maxval=2, seed=cnn.seed_to_randint(net_seed + 60 + i)))

            # time lag reversed
            if len(s) == 3:
                layer['t'] = tf.reshape(tf.range(layer['time_win_smp']-1, -1, -1, dtype=tf.float32), [layer['time_win_smp'], 1, 1]) - layer['delay']
            elif cross_channels and (bank_count == 1):
                layer['t'] = tf.reshape(tf.range(layer['time_win_smp'] - 1, -1, -1, dtype=tf.float32),
                                        [layer['time_win_smp'], 1, 1, 1]) - layer['delay']
            else:
                layer['t'] = tf.reshape(tf.range(layer['time_win_smp'] - 1, -1, -1, dtype=tf.float32),
                                        [1, layer['time_win_smp'], 1, 1]) - layer['delay']
            coefficients = tf.math.sin(layer['f1'] * layer['t']) * tf.math.exp(-layer['tau'] * layer['t']) * layer['gain']
            layer['b'] = tf.math.greater(layer['t'], tf.constant(0, dtype=tf.float32))
            layer['W'] = tf.multiply(coefficients, tf.cast(layer['b'], tf.float32))

            log.info("f1 shape: %s", layer['f1'].shape)
            log.info("W shape: %s", layer['W'].shape)

            if cross_channels & (bank_count == 1):
                # special case: "outer product" convolve each channel with each filter
                # insert placeholder dim on axis=3
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 3)
                layer['tY'] = tf.nn.conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID')
                layer['Y'] = tf.reshape(layer['tY'], [-1, layer['tY'].shape[1], layer['tY'].shape[2]*layer['tY'].shape[3]])
            elif bank_count == 1:
                # original implementation (no filter bank concept)
                X_pad = tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]])
                layer['Y'] = tf.nn.conv1d(X_pad, layer['W'], stride=1, padding='VALID')
            elif in_chan_count == bank_count*chan_count:
                # each bank applied to a segment of the input channels
                # insert placeholder dim on axis=1
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
                layer['tY'] = tf.nn.depthwise_conv2d(
                    X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, layer['rate']])
                s = tf.shape(layer['tY'])
                layer['Y'] = tf.reduce_sum(tf.reshape(
                    layer['tY'], [s[0], layer['tY'].shape[2],
                                  tf.Dimension(bank_count), tf.Dimension(chan_count)]), axis=3)
            else:
                # apply each fir bank to same input channels
                # insert placeholder dim on axis=1
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
                layer['tY'] = tf.nn.depthwise_conv2d(
                    X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, layer['rate']])
                s = tf.shape(layer['tY'])
                layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'],
                                                      [s[0], layer['tY'].shape[2], tf.Dimension(chan_count),
                                                       tf.Dimension(bank_count)]), axis=2)
            log.info("X_pad shape: %s", X_pad.shape)
            log.info("Y shape: %s", layer['Y'].shape)

        elif m['fn'] in ['nems.modules.fir.filter_bank']:

            layer['type'] = 'conv_bank_1d'
            layer['time_win_smp'] = m['phi']['coefficients'].shape[1]

            layer['rank'] = None  # we're handling rank with explicit spectral filters
            bank_count = m['fn_kwargs']['bank_count']
            layer['n_kern'] = bank_count
            layer['rate'] = m['fn_kwargs'].get('rate', 1)

            c = np.fliplr(m['phi']['coefficients']).astype('float32').T
            chan_count = int(c.shape[1]/bank_count)
            in_chan_count = int(layer['X'].shape[2])

            # split inputs into the different kernels
            if bank_count == 1:
                c = np.reshape(c, (c.shape[0], 1, 1, bank_count*chan_count))
            elif in_chan_count == bank_count*chan_count:
                c = np.reshape(c, (1, c.shape[0], chan_count*bank_count, 1))
            else:
                c = np.reshape(c, (1, c.shape[0], chan_count, bank_count))

            if np.sum(np.abs(c)) == 0:
                c[:, :, :, :] = 1

            # figure out input padding to ensure causality,
            pad_size = np.int32((layer['time_win_smp']-1)*layer['rate'])

            if use_modelspec_init:
                layer['W'] = tf.Variable(c)
            else:
                layer['W'] = cnn.weights_norm(c.shape, sig=weight_scale, seed=cnn.seed_to_randint(net_seed)+i)

            if bank_count == 1:
                # "outer product" convolve each channel with each filter
                # insert placeholder dim on axis=3
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 3)
                layer['tY'] = tf.nn.conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID')
                layer['Y'] = tf.reshape(layer['tY'],[-1, layer['tY'].shape[1], layer['tY'].shape[2]*layer['tY'].shape[3]])
            elif in_chan_count == bank_count*chan_count:
                # each bank applied to a segment of the input channels
                # insert placeholder dim on axis=1
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
                layer['tY'] = tf.nn.depthwise_conv2d(
                    X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, layer['rate']])
                s = tf.shape(layer['tY'])
                layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'],[s[0], layer['tY'].shape[2], tf.Dimension(bank_count), tf.Dimension(chan_count)]), axis=3)
            else:
                # apply each fir bank to same input channels
                # insert placeholder dim on axis=1
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
                layer['tY'] = tf.nn.depthwise_conv2d(
                    X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID', rate=[1, layer['rate']])
                s = tf.shape(layer['tY'])
                layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'],
                                                      [s[0], layer['tY'].shape[2], tf.Dimension(chan_count),
                                                       tf.Dimension(bank_count)]), axis=2)

            log.info("W shape: %s", layer['W'].shape)
            log.info("X_pad shape: %s", X_pad.shape)
            log.info("tY shape: %s", layer['tY'].shape)
            log.info("Y shape: %s", layer['Y'].shape)

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            layer['type'] = 'reweight'
            c = m['phi']['coefficients'].astype('float32').T
            layer['n_kern'] = c.shape[1]

            if use_modelspec_init & (np.sum(np.abs(c)) > 0):
                layer['W'] = tf.Variable(np.reshape(c, (1, c.shape[0], c.shape[1])))
            else:
                s = [1, c.shape[0], c.shape[1]]
                layer['W'] = tf.Variable(tf.random.normal(s, stddev=weight_scale, mean=0,
                                                          seed=cnn.seed_to_randint(net_seed + i)))
                #   cnn.kern2D(1, c.shape[0], c.shape[1],
                #                       weight_scale, seed=net_seed, distr='norm')
                #print('rand in init: %s',layer['W'].eval())

            if m['fn_kwargs']['i'] == 'resp':
                layer['L'] = tf.nn.conv1d(layer['D'], layer['W'], stride=1, padding='SAME')
            else:
                layer['Y'] = tf.nn.conv1d(layer['X'], layer['W'], stride=1, padding='SAME')

        elif m['fn'] in ['nems.modules.weight_channels.gaussian']:
            layer['n_kern'] = m['phi']['mean'].shape[0]

            # HACK : scale sd by 10 to play well with TF fitter
            mn = m['phi']['mean'].astype('float32')
            sd = m['phi']['sd'].astype('float32') * 10

            if use_modelspec_init:
                layer['m'] = tf.Variable(np.reshape(mn, (1, 1, mn.shape[0])))
                layer['s'] = tf.Variable(np.reshape(sd, (1, 1, sd.shape[0])))
            else:
                log.info('Using TF rand for wcg')
                layer['m'] = tf.Variable(tf.random.uniform(
                    [1, 1, layer['n_kern']], minval=0, maxval=1,
                    seed=cnn.seed_to_randint(net_seed + i)))
                layer['s'] = tf.Variable(tf.random.uniform(
                    [1, 1, layer['n_kern']], minval=0.1, maxval=0.5,
                    seed=cnn.seed_to_randint(net_seed + 20 + i)))

            # trying to impose NEMS bounds
            b = m['bounds']
            layer['m0'] = tf.clip_by_value(layer['m'], b['mean'][0][0], b['mean'][1][0])
            layer['s0'] = tf.clip_by_value(layer['s']/10, b['sd'][0][0], b['sd'][1][0])
            #sd[sd < 0.00001] = 0.00001
            #layer['s0'] = tf.clip_by_value(layer['s'], 0.00001*10, 100)
            layer['f'] = tf.reshape(tf.range(n_input_feats, dtype=tf.float32),
                                    [1, n_input_feats, 1]) / n_input_feats
            layer['Wraw'] = tf.exp(-0.5 * tf.square((layer['f'] - layer['m0']) / layer['s0']))
            layer['W'] = layer['Wraw'] / tf.reduce_sum(layer['Wraw'], axis=1)
            layer['Y'] = tf.nn.conv1d(layer['X'], layer['W'], stride=1, padding='SAME')

        elif m['fn'] == 'nems.modules.state.state_dc_gain':
            # match this function from nems.modules.state.state_dc_gain
            #fn = lambda x: np.matmul(g, rec[s]._data) * x + np.matmul(d, rec[s]._data)
            if 'phi' in m.keys():
                g = m['phi']['g'].astype('float32').T
                d = m['phi']['d'].astype('float32').T
                g = np.reshape(g, (1, g.shape[0], g.shape[1]))
                d = np.reshape(d, (1, d.shape[0], d.shape[1]))

                if use_modelspec_init:
                    layer['g'] = tf.Variable(g)
                    layer['d'] = tf.Variable(d)
                else:
                    layer['g'] = tf.Variable(tf.random_normal(
                        g.shape, stddev=weight_scale, seed=cnn.seed_to_randint(net_seed + i)))
                    layer['d'] = tf.Variable(tf.random_normal(
                        d.shape, stddev=weight_scale, seed=cnn.seed_to_randint(net_seed + 20 + i)))
            else:
                # dc/gain values are fixed
                g = m['fn_kwargs']['g'].astype('float32').T
                d = m['fn_kwargs']['d'].astype('float32').T
                g = np.reshape(g, (1, g.shape[0], g.shape[1]))
                d = np.reshape(d, (1, d.shape[0], d.shape[1]))

                layer['g'] = tf.constant(g)
                layer['d'] = tf.constant(d)

            layer['n_kern'] = g.shape[2]
            layer['Sg'] = tf.nn.conv1d(layers[0]['S'], layer['g'], stride=1, padding='SAME')
            layer['Sd'] = tf.nn.conv1d(layers[0]['S'], layer['d'], stride=1, padding='SAME')
            #layer['Sg'] = tf.multiply(layers[0]['S'], layer['g'])
            #layer['Sd'] = tf.multiply(layers[0]['S'], layer['d'])
            layer['Y'] = layer['X'] * layer['Sg'] + layer['Sd']

            log.info("g shape: %s", layer['g'].shape)  # [filter_width, in_channels, out_channels]
            log.info("S shape: %s", layers[0]['S'].shape)  # [batch, in_width, in_channels]
            log.info("X shape: %s", layer['X'].shape)  # [batch, in_width, in_channels]
            log.info("Sg shape: %s", layer['Sg'].shape)
            log.info("Y shape: %s", layer['Y'].shape)
        else:
            raise ValueError('Module %s not supported', m['fn'])

        # may not be necessary?
        layer['time_win_sec'] = layer['time_win_smp'] / fs

        layers.append(layer)
    return layers


def tf2modelspec(net, modelspec):
    """
    pass TF cnn fit back into modelspec phi.
    TODO: Generate new modelspec if not provided
    DONE: Make sure that the dimension mappings work reliably for filter banks and such
    """

    net_layer_vals = net.layer_vals()
    for i, m in enumerate(modelspec):
        log.info('tf2modelspec: ' + m['fn'])

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            m['phi']['offset'] = -net_layer_vals[i]['b'][0, :, :].T

        elif 'levelshift' in m['fn']:
            m['phi']['level'] = net_layer_vals[i]['b'][0, :, :].T

        elif m['fn'] in ['nems.modules.nonlinearity.double_exponential']:
            # base + amplitude * exp(  -exp(np.array(-exp(kappa)) * (x - shift))  )
            m['phi']['base'] = net_layer_vals[i]['base'][:, :, 0].T
            m['phi']['amplitude'] = net_layer_vals[i]['amplitude'][:, :, 0].T
            m['phi']['kappa'] = net_layer_vals[i]['kappa'][:, :, 0].T
            m['phi']['shift'] = net_layer_vals[i]['shift'][:, :, 0].T

        elif m['fn'] in ['nems.modules.fir.basic']:
            m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, :, 0].T)

        elif m['fn'] in ['nems.modules.fir.damped_oscillator']:
            if (m['fn_kwargs']['bank_count'] == 1) and (not m['fn_kwargs']['cross_channels']):
                m['phi']['f1s'] = net_layer_vals[i]['f1'][:, :, 0].T
                m['phi']['taus'] = net_layer_vals[i]['tau'][:, :, 0].T
                m['phi']['gains'] = net_layer_vals[i]['gain'][:, :, 0].T
                m['phi']['delays'] = net_layer_vals[i]['delay'][:, :, 0].T
            else:
                # new depthwise_conv2d
                m['phi']['f1s'] = np.reshape(net_layer_vals[i]['f1'][0, 0, :, :].T, [-1, 1])
                m['phi']['taus'] = np.reshape(net_layer_vals[i]['tau'][0, 0, :, :].T, [-1, 1])
                m['phi']['gains'] = np.reshape(net_layer_vals[i]['gain'][0, 0, :, :].T, [-1, 1])
                m['phi']['delays'] = np.reshape(net_layer_vals[i]['delay'][0, 0, :, :].T, [-1, 1])

        elif m['fn'] in ['nems.modules.fir.filter_bank']:
            if m['fn_kwargs']['bank_count'] == 1:
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T)
            else:
                # new depthwise_conv2d
                c = net_layer_vals[i]['W'][0, :, :, :]
                c = np.transpose(c, (2, 1, 0))
                c = np.reshape(c, [-1, c.shape[-1]])
                m['phi']['coefficients'] = np.fliplr(c)
            #else:
            #    # inefficient conv2d
            #    m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T)

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            m['phi']['coefficients'] = net_layer_vals[i]['W'][0, :, :].T

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            m['phi']['offset'] = net_layer_vals[i]['b'][0, :, :].T

        elif m['fn'] in ['nems.modules.weight_channels.gaussian']:
            #m['phi']['mean'] = net_layer_vals[i]['m'][0, 0, :].T
            m['phi']['mean'] = np.clip(net_layer_vals[i]['m'][0, 0, :].T,
                                     m['bounds']['mean'][0], m['bounds']['mean'][1])
            m['phi']['sd'] = np.clip(net_layer_vals[i]['s'][0, 0, :].T / 10,
                                     m['bounds']['sd'][0], m['bounds']['sd'][1])

        elif m['fn'] in ['nems.modules.state.state_dc_gain']:
            # if init.st, not fitting these params, no phi, so skip
            if 'phi' in m.keys():
                m['phi']['g'] = net_layer_vals[i]['g'][0, :, :].T
                m['phi']['d'] = net_layer_vals[i]['d'][0, :, :].T

        else:
            raise ValueError("NEMS module fn=%s not supported", m['fn'])

    return modelspec


def _fit_net(F, D, modelspec, seed, fs, train_val_test, optimizer='Adam',
             max_iter=1000, learning_rate=0.01, use_modelspec_init=False, S=None):

    n_feats = F.shape[2]
    data_dims = D.shape
    sr_Hz = fs

    tf.reset_default_graph()
    if 1:
        if S is not None:
            state_dims = S.shape[2]
        else:
            state_dims = 0

        layers = modelspec2tf(modelspec, tps_per_stim=D.shape[1], feat_dims=n_feats,
                              data_dims=D.shape[2], state_dims=state_dims, fs=fs,
                              use_modelspec_init=use_modelspec_init)
        net2 = cnn.Net(data_dims, n_feats, sr_Hz, layers, seed=seed, log_dir=modelspec.meta['modelpath'])
    else:
        layers = modelspec2cnn(modelspec, n_inputs=n_feats, fs=fs, use_modelspec_init=use_modelspec_init)
        # layers = [{'act': 'identity', 'n_kern': 1,
        #  'time_win_sec': 0.01, 'type': 'reweight-positive'},
        # {'act': 'relu', 'n_kern': 1, 'rank': None,
        #  'time_win_sec': 0.15, 'type': 'conv'}]

        net2 = cnn.Net(data_dims, n_feats, sr_Hz, layers, seed=seed, log_dir=modelspec.meta['modelpath'])
        net2.parse_layers()

    net2.initialize()
    net2.optimizer = optimizer
    # net2_layer_init = net2.layer_vals()
    # log.info(net2_layer_init)

    log.info('Train set: %d  Test set (early stopping): %d',
             np.sum(train_val_test == 0),np.sum(train_val_test == 1))
    net2.train(F, D, max_iter=max_iter, train_val_test=train_val_test,
               learning_rate=learning_rate, S=S)

    if 1:
        modelspec = tf2modelspec(net2, modelspec)
    else:
        modelspec = cnn2modelspec(net2, modelspec)

    return modelspec, net2


def fit_tf_init(modelspec=None, est=None, use_modelspec_init=True,
                optimizer='Adam', max_iter=500, cost_function='mse', **context):
    """
    pre-fit a model with the final output NL stripped. TF equivalent of
    nems.initializers.prefit_to_target()
    :param est: A recording object
    :param modelspec: A modelspec object
    :param use_modelspec_init: [True] use input modelspec phi for initialization. Otherwise use random inits
    :param optimizer:
    :param max_iter: max number of training iterations
    :param cost_function: TODO, not implemented yet
    :param context: extra stuff from xforms context
    :return: dictionary with modelspec, compatible with xforms
    """

    # preserve input modelspec
    modelspec = modelspec.copy()

    target_module = ['levelshift', 'relu']
    extra_exclude = ['stp', 'rdt_gain', 'state_dc_gain', 'state_gain']

    # figure out last modelspec module to fit
    target_i = None
    if type(target_module) is not list:
        target_module = [target_module]
    for i, m in enumerate(modelspec.modules):
        tlist = [True for t in target_module if t in m['fn']]

        if len(tlist):
            target_i = i + 1
            # don't break. use last occurrence of target module

    if not target_i:
        log.info("target_module: {} not found in modelspec."
                 .format(target_module))
        return modelspec
    else:
        log.info("target_module: {0} found at modelspec[{1}]."
                 .format(target_module, target_i-1))

    # identify any excluded modules and take them out of temp modelspec
    # that will be fit here
    exclude_idx = []
    tmodelspec = ms.ModelSpec()
    for i in range(len(modelspec)):
        m = copy.deepcopy(modelspec[i])

        for fn in extra_exclude:
            # log.info('exluding '+fn)
            # log.info(m['fn'])
            # log.info(m.get('phi'))
            if (fn in m['fn']):
                if (m.get('phi') is None):
                    m = priors.set_mean_phi([m])[0]  # Inits phi
                    log.info('Mod %d (%s) fixing phi to prior mean', i, fn)
                else:
                    log.info('Mod %d (%s) fixing phi', i, fn)

                m['fn_kwargs'].update(m['phi'])
                del m['phi']
                del m['prior']
                exclude_idx.append(i)
                # log.info(m)
        if ('relu' in m['fn']):
            log.info('found relu')

        elif ('levelshift' in m['fn']):
            #m = priors.set_mean_phi([m])[0]
            output_name = modelspec.meta.get('output_name', 'resp')
            try:
                mean_resp = np.nanmean(est[output_name].as_continuous(), axis=1, keepdims=True)
            except NotImplementedError:
                # as_continuous only available for RasterizedSignal
                mean_resp = np.nanmean(est[output_name].rasterize().as_continuous(), axis=1, keepdims=True)
            log.info('Mod %d (%s) fixing level to %s mean %.3f',
                     i, m['fn'], output_name, mean_resp[0])
            log.info('resp has %d channels', len(mean_resp))
            m['phi']['level'][:] = mean_resp

        if (i < target_i) or ('merge_channels' in m['fn']):
            tmodelspec.append(m)
    log.info(tmodelspec)

    # fit the subset of modules - this is instead of calling analysis_function in
    # nems.initializers.prefit_to_target
    new_context = fit_tf(modelspec=tmodelspec, est=est, use_modelspec_init=use_modelspec_init,
                     optimizer=optimizer, max_iter=max_iter, cost_function=cost_function,
                     **context)
    tmodelspec = new_context['modelspec']

    for i in np.setdiff1d(np.arange(target_i), np.array(exclude_idx)).tolist():
        modelspec[int(i)] = tmodelspec[int(i)]

    # pre-fit static NL if it exists
    for m in modelspec.modules:
        if 'double_exponential' in m['fn']:
            modelspec = init_dexp(est, modelspec)
            break
        elif 'logistic_sigmoid' in m['fn']:
            log.info("initializing priors and bounds for logsig ...\n")
            modelspec = init_logsig(est, modelspec)
            break
        elif 'saturated_rectifier' in m['fn']:
            log.info('initializing priors and bounds for relat ...\n')
            modelspec = init_relsat(est, modelspec)
            break

    return {'modelspec': modelspec}


def fit_tf(modelspec=None, est=None,
           use_modelspec_init=True, init_count=1,
           optimizer='Adam', max_iter=1000, cost_function='mse', **context):
    """
    :param est: A recording object
    :param modelspec: A modelspec object
    :param use_modelspec_init: [True] use input modelspec phi for initialization. Otherwise use random inits
    :param init_count: number of random initializations (if use_modelspec_init==False)
    :param optimizer:
    :param max_iter: max number of training iterations
    :param cost_function: not implemented
    :param metaname:
    :param context:
    :return: dictionary with modelspec, compatible with xforms
    """

    start_time = time.time()

    net1_seed = 50
    sr_Hz = est['resp'].fs
    #time_win_sec = 0.1

    new_est = est.apply_mask()
    n_feats = new_est['stim'].shape[0]
    epoch_name = 'REFERENCE'
    e = est['stim'].get_epoch_indices(epoch_name, mask=est['mask'])
    if 'state' in est.signals.keys():
        n_states = est['state'].shape[0]
    else:
        n_states = 0
        S = None
    # length of each segment is length of a reference
    de = e[:, 1] - e[:, 0]
    n_tps_per_stim = de[0]
    if np.sum(np.abs(de-n_tps_per_stim)) > 0:
        epoch_name = 'TRIAL'

    F = np.transpose(est['stim'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    D = np.transpose(est['resp'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    if n_states > 0:
        S = np.transpose(est['state'].extract_epoch(epoch=epoch_name, mask=est['mask']),[0, 2, 1])

    feat_dims = F.shape
    data_dims = D.shape
    log.info('feat_dims: %s', feat_dims)
    log.info('data_dims: %s', data_dims)

    train_val_test = np.zeros(data_dims[0])
    val_n = int(0.9 * data_dims[0])
    train_val_test[val_n:] = 1
    train_val_test = np.roll(train_val_test, int(data_dims[0]/init_count*modelspec.fit_index))
    seed = net1_seed + modelspec.fit_index

    modelspec_pre = modelspec.copy()
    modelspec, net = _fit_net(F, D, modelspec, seed, new_est['resp'].fs,
                         train_val_test=train_val_test,
                         optimizer=optimizer, max_iter=np.min([max_iter]),
                         use_modelspec_init=use_modelspec_init, S=S)

    new_est = eval_tf(modelspec, new_est)
    y = new_est['pred'].as_continuous()
    y2 = new_est['pred_nems'].as_continuous()
    E = np.nanstd(y[:,10:]-y2[:,10:])
    """
    try:
        new_est = modelspec.evaluate(new_est)
    except:
        log.info('evaluation of tf->nems models failed')
        import pdb
        pdb.set_trace()

    # test that TF and NEMS models have same prediction
    y = net.predict(F, S=S)
    p1 = y[0, :, 0]
    if not trial_based_reshape:
        #p2 = new_est['pred'].as_continuous()[0,:n_tps_per_stim]
        y2 = np.reshape(new_est['pred'].as_continuous().copy().T, data_dims)
    else:
        y2 = np.transpose(new_est['pred'].extract_epoch(epoch='TRIAL', mask=est['mask']),[0, 2, 1])
    p2 = y2[0, :, 0]
    E = np.nanstd(p1-p2)
    """
    log.info('Mean difference between NEMS and TF model pred: %e', E)
    #import pdb; pdb.set_trace()

    if np.isnan(E) or (E > 1e-2):
        log.info('E too big? Jumping to debug mode.')
        import matplotlib.pyplot as plt
        plt.figure()
        ax1=plt.subplot(2, 1, 1)
        ax1.plot(y[0, :1000], 'b')
        ax1.plot(y2[0, :1000], 'r')
        ax1.plot(y2[0, :1000]-y[0, :1000], '--')
        ax1.legend(('TF', 'NEMS', 'diff'))

        plt.show()
        log.info(modelspec.phi)
        net_layer_vals = net.layer_vals()
        m = modelspec[2]
        if m['fn'] == 'nems.modules.fir.damped_oscillator':
            from nems.modules.fir import do_coefficients
            args = m['fn_kwargs']
            args.update(m['phi'])
            w_nems = do_coefficients(**args)
            w_tf = net_layer_vals[2]['W']
            ax3=plt.subplot(2, 2, 3)
            ax3.plot(np.flipud(np.squeeze(w_tf)))
            ax4=plt.subplot(2, 2, 4)
            ax4.plot(w_nems.T)
        #from nems.modules.weight_channels import gaussian_coefficients
        #log.info(gaussian_coefficients(modelspec.phi[1]['mean'], modelspec.phi[1]['sd'],
        #                      modelspec[1]['fn_kwargs']['n_chan_in']))
        import pdb
        pdb.set_trace()

    nems.utils.progress_fun()

    elapsed_time = (time.time() - start_time)
    modelspec.meta['fitter'] = 'fit_tf'
    modelspec.meta['fit_time'] = elapsed_time
    modelspec.meta['n_parms'] = len(modelspec.phi_vector)

    #import pdb
    #pdb.set_trace()

    return {'modelspec': modelspec}


def eval_tf(modelspec, est):
    """
    TODO : evaluate a NEMS model through TF
    :param modelspec:
    :param est:
    :return:
    """

    new_est = modelspec.evaluate(est)
    new_est['pred_nems'] = new_est['pred'].copy()
    
    # extract stim. does it need to be reshaped to be multiple batches? probably not.
    n_feats = new_est['stim'].shape[0]

    # don't need batches, so can use a single "stim" that contains the whole recording
    n_stim = 1
    n_resp = new_est['resp'].shape[0]
    n_tps_per_stim = new_est['resp'].shape[1]

    feat_dims = [n_stim, n_tps_per_stim, n_feats]
    data_dims = [n_stim, n_tps_per_stim, n_resp]

    # extract stimulus matrix
    F = np.reshape(new_est['stim'].as_continuous().copy().T, feat_dims)
    #D = np.reshape(new_est['resp'].as_continuous().copy().T, data_dims)

    if 'state' in est.signals.keys():
        n_states = est['state'].shape[0]
        state_dims = [n_stim, n_tps_per_stim, n_states]
        S = np.reshape(new_est['state'].as_continuous().copy().T, state_dims)
    else:
        n_states = 0
        S = None

    log.info('feat_dims: %s', feat_dims)
    log.info('data_dims: %s', data_dims)

    fs = est['resp'].fs

    tf.reset_default_graph()

    # initialize tf and evaluate
    layers = modelspec2tf(modelspec, tps_per_stim=n_tps_per_stim, feat_dims=n_feats,
                          data_dims=n_resp, state_dims=n_states, fs=fs,
                          use_modelspec_init=True)
    net = cnn.Net(data_dims, n_feats, fs, layers, seed=0, log_dir=modelspec.meta['modelpath'])
    net.initialize()

    y = np.reshape(net.predict(F, S=S).T, [n_resp, n_tps_per_stim])

    # paste back into rec
    new_est['pred'] = new_est['pred']._modified_copy(data=y)

    # test that TF and NEMS models have same prediction
    y2 = new_est['pred_nems'].as_continuous()

    #plt.figure()
    #plt.plot(y[0,:1000,0])
    #plt.plot(y2[0,:1000,0])

    #E = np.nanstd(new_est['pred'].as_continuous()-new_est['pred_nems'].as_continuous())

    return new_est
