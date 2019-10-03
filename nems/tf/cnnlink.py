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
    #D = tf.placeholder('float32', shape=[None, tps_per_stim, data_dims])

    layers = []
    for i, m in enumerate(modelspec):
        log.info('modelspec2tf: ' + m['fn'])

        layer = {}
        # input to each layer is output of previous layer
        if i == 0:
            layer['X'] = F
            if state_dims > 0:
                layer['S'] = S
        else:
            layer['X'] = layers[-1]['Y']

        n_input_feats = np.int32(layer['X'].shape[2])
        # default integration time is one bin
        layer['time_win_smp'] = 1 # default

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            layer['type'] = 'relu'
            c = -modelspec[i]['phi']['offset'].astype('float32').T
            layer['n_kern'] = c.shape[1]
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

        elif m['fn'] in ['nems.modules.fir.filter_bank']:

            layer['type'] = 'conv_bank_1d'
            layer['time_win_smp'] = m['phi']['coefficients'].shape[1]

            layer['rank'] = None  # we're handling rank with explicit spectral filters
            bank_count = m['fn_kwargs']['bank_count']
            layer['n_kern'] = bank_count
            layer['rate'] = m['fn_kwargs'].get('rate', 1)

            c = np.fliplr(m['phi']['coefficients']).astype('float32').T
            chan_count = int(c.shape[1]/bank_count)

            # split inputs into the different kernels
            #c = np.reshape(c, (1, c.shape[0], bank_count, chan_count))
            if bank_count==1:
                c = np.reshape(c, (c.shape[0], 1, 1, bank_count*chan_count))
            else:
                c = np.reshape(c, (1, c.shape[0], bank_count*chan_count, 1))

            if np.sum(np.abs(c)) == 0:
                c[0, 0, :, :] = 1

            # figure out input padding to ensure causality,
            pad_size = np.int32((layer['time_win_smp']-1)*layer['rate'])

            if use_modelspec_init:
                layer['W'] = tf.Variable(c)
            else:
                layer['W'] = cnn.weights_norm(c.shape,
                    sig=weight_scale, seed=cnn.seed_to_randint(net_seed)+i)

            if bank_count==1:
                # special case, convolve each channel with each filters
                # insert placeholder dim on axis=2
                #import pdb; pdb.set_trace()
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 3)
                layer['tY'] = tf.nn.conv2d(X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID')
                layer['Y'] = tf.reshape(layer['tY'],[-1, layer['tY'].shape[1], layer['tY'].shape[2]*layer['tY'].shape[3]])
            else:
                # insert placeholder dim on axis=1
                X_pad = tf.expand_dims(tf.pad(layer['X'], [[0, 0], [pad_size, 0], [0, 0]]), 1)
                layer['tY'] = tf.squeeze(tf.nn.depthwise_conv2d(
                    X_pad, layer['W'], strides=[1, 1, 1, 1], padding='VALID',
                    rate=[1, layer['rate']]), axis=1)
                s = tf.shape(layer['tY'])
                layer['Y'] = tf.reduce_sum(tf.reshape(layer['tY'],[s[0], layer['tY'].shape[1], tf.Dimension(bank_count), tf.Dimension(chan_count)]), axis=3)

            #log.info("W shape: %s", layer['W'].shape)
            #log.info("X_pad shape: %s", X_pad.shape)
            #log.info("tY shape: %s", layer['tY'].shape)
            #log.info("Y shape: %s", layer['Y'].shape)

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
            g = m['phi']['g'].astype('float32').T
            d = m['phi']['d'].astype('float32').T
            layer['n_kern'] = g.shape[1]
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

            layer['Sg'] = tf.nn.conv1d(layers[0]['S'], layer['g'], stride=1, padding='SAME')
            layer['Sd'] = tf.nn.conv1d(layers[0]['S'], layer['d'], stride=1, padding='SAME')
            layer['Y'] = layer['X'] * layer['Sg'] + layer['Sd']
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
    TODO: Make sure that the dimension mappings work reliably for filter banks and such
    """

    net_layer_vals = net.layer_vals()
    current_layer = 0
    for i, m in enumerate(modelspec):
        log.info('tf2modelspec: ' + m['fn'])

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            m['phi']['offset'] = -net_layer_vals[i]['b'][0, :, :].T

        elif 'levelshift' in m['fn']:
            m['phi']['level'] = net_layer_vals[i]['b'][0, :, :].T

        elif m['fn'] in ['nems.modules.fir.basic']:
            m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, :, 0].T)

        elif m['fn'] in ['nems.modules.fir.filter_bank']:
            #import pdb; pdb.set_trace()
            if m['fn_kwargs']['bank_count']==1:
                #m['phi']['coefficients'] = np.flipud(np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T))
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T)
            elif net_layer_vals[current_layer]['W'].shape[1]>1:
                # new depthwise_conv2d
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][0, :, :, 0].T)
            else:
                # inefficient conv2d
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[i]['W'][:, 0, 0, :].T)

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

    if 'mask' in est.signals.keys():
        est = est.apply_mask()

    #modelspec = modelspec.copy()

    sr_Hz = est['resp'].fs
    time_win_sec = 0.1

    n_feats = est['stim'].shape[0]
    e = est['stim'].get_epoch_indices('REFERENCE')

    # length of each segment is length of a reference
    n_tps_per_stim = e[0][1]-e[0][0]
    # before: hard coded as n_tps_per_stim = 550

    n_stim = int(est['stim'].shape[1] / n_tps_per_stim)
    n_resp = est['resp'].shape[0]

    feat_dims = [n_stim, n_tps_per_stim, n_feats]
    data_dims = [n_stim, n_tps_per_stim, n_resp]
    net1_seed = 50
    log.info('feat_dims: %s', feat_dims)
    log.info('data_dims: %s', data_dims)

    # extract stimulus matrix
    try:
        F = np.reshape(est['stim'].as_continuous().copy().T, feat_dims)
        D = np.reshape(est['resp'].as_continuous().copy().T, data_dims)
        if 'state' in est.signals.keys():
            n_states = est['state'].shape[0]
            state_dims = [n_stim, n_tps_per_stim, n_states]
            S = np.reshape(est['state'].as_continuous().copy().T, state_dims)
        else:
            S = None
    except:
        import pdb
        pdb.set_trace()

    # MOVED FUNCTIONALITY OUT TO xforms
    #new_est = est.tile_views(init_count)
    #r_fit = np.zeros((init_count, n_resp), dtype=np.float)
    #log.info('fitting from {} initial conditions'.format(init_count))

    # add a new fit for each initial condition
    #modelspec = modelspec.tile_fits(init_count)

    #for i in range(init_count):

     #   new_est = new_est.set_view(i)
     #   modelspec.set_fit(i)
    new_est = est.copy()

    train_val_test = np.zeros(data_dims[0])
    val_n = int(0.9 * data_dims[0])
    train_val_test[val_n:] = 1
    train_val_test = np.roll(train_val_test, int(data_dims[0]/init_count*modelspec.fit_index))
    seed = net1_seed + modelspec.fit_index

    modelspec, net = _fit_net(F, D, modelspec, seed, est['resp'].fs,
                         train_val_test=train_val_test,
                         optimizer=optimizer, max_iter=np.min([max_iter]),
                         use_modelspec_init=use_modelspec_init, S=S)

    try:
        new_est = modelspec.evaluate(new_est)
    except:
        import pdb
        pdb.set_trace()

    #    r_fit[i], se_fit = nmet.j_corrcoef(new_est, 'pred', 'resp')
    #    log.info('r_fit this iteration (%d/%d): %s', i+1, init_count, r_fit[i])
    #    nems.utils.progress_fun()
    #    #log.info(modelspec.phi)

    # test that TF and NEMS models have same prediction
    y = net.predict(F, S=S)
    p1 = y[0,:,0]
    p2 = new_est['pred'].as_continuous()[0,:n_tps_per_stim]
    E = np.std(p1-p2)
    log.info('Mean difference between NEMS and TF model pred: %e', E)
    if np.isnan(E) or (E > 1e-2):
        log.info('E too big? Jumping to debug mode.')
        import matplotlib.pyplot as plt
        plt.figure()
        ax1=plt.subplot(1, 1, 1)
        ax1.plot(y[0:2,:,0].T)
        ax1.plot(new_est['pred'].as_continuous()[0:2,:n_tps_per_stim].T,'--')
        ax1.plot(new_est['pred'].as_continuous()[0:2,:n_tps_per_stim].T-y[0:2,:,0].T)
        plt.show()
        log.info(modelspec.phi)
        #from nems.modules.weight_channels import gaussian_coefficients
        #log.info(gaussian_coefficients(modelspec.phi[1]['mean'], modelspec.phi[1]['sd'],
        #                      modelspec[1]['fn_kwargs']['n_chan_in']))
        import pdb
        pdb.set_trace()

    # figure out which modelspec does best on fit data
    #mean_r_fit = np.mean(r_fit, axis=1)
    #log.info('mean r_fit per init: %s', mean_r_fit)
    #ibest = np.argmax(mean_r_fit)
    #log.info("Best fit_index is {} mean r_fit={:.3f}".format(ibest, mean_r_fit[ibest]))
    #modelspec = modelspec.copy(fit_index=ibest)

    if 0:
        # now continue fitting the best model!
        log.info('Now running final fit on best pre-fit. Pre phi:')
        log.info(modelspec.phi)
        train_val_test = np.zeros(data_dims[0])
        val_n = int(0.9 * data_dims[0])
        train_val_test[val_n:] = 1
        train_val_test = np.roll(train_val_test, int(data_dims[0]/init_count*ibest))
        seed = net1_seed + ibest
        modelspec, net = _fit_net(F, D, modelspec, seed, est['resp'].fs,
                             train_val_test=train_val_test,
                             optimizer=optimizer, max_iter=max_iter,
                             use_modelspec_init=True, S=S)
        new_est = modelspec.evaluate(new_est)
        r_fit_final, se_fit = nmet.j_corrcoef(new_est, 'pred', 'resp')
        log.info('Final phi:')
        log.info(modelspec.phi)
        log.info('r_fit_final: %s', r_fit_final)

    nems.utils.progress_fun()

    elapsed_time = (time.time() - start_time)
    modelspec.meta['fitter'] = 'fit_tf'
    modelspec.meta['fit_time'] = elapsed_time
    modelspec.meta['n_parms'] = len(modelspec.phi_vector)

    #import pdb
    #pdb.set_trace()

    return {'modelspec': modelspec}

##################
# JUNK
#################


def modelspec2cnn(modelspec, data_dims=1, n_inputs=18, fs=100,
                  net_seed=1, use_modelspec_init=True):
    """convert NEMS modelspec to TF network.
    Initialize with existing phi?
    Translations:
        wc -> reweight-positive-zeros, identity (require lvl?)
        fir+lvl -> conv, identity
        wc+relu -> reweight, relu
        fir+relu -> conv2d, relu

    """
    raise Warning("DEPRECATED?")
    layers = []
    for i, m in enumerate(modelspec):
        log.info('modelspec2cnn: ' + m['fn'])
        if i < len(modelspec)-1:
            next_fn = modelspec[i+1]['fn']
        else:
            next_fn = None

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            pass # already handled

        elif 'levelshift' in m['fn']:
            pass # already handled

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            layer = {}
            layer['type'] = 'dlog'
            layer['time_win_sec'] = 1 / fs
            layer['act'] = ''
            layer['n_kern'] = 1  # m['prior']['coefficients'][1]['mean'].shape[0]
            layer['rank'] = None  # P['rank']
            if use_modelspec_init:
                c = m['phi']['offset'].astype('float32').T
                layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))

            layers.append(layer)

        elif m['fn'] in ['nems.modules.fir.basic', 'nems.modules.fir.filter_bank']:
            layer = {}

            layer['time_win_sec'] = m['phi']['coefficients'].shape[1] / fs
            if next_fn == 'nems.modules.nonlinearity.relu':
                layer['act'] = 'relu'
                if use_modelspec_init:
                    c = -modelspec[i + 1]['phi']['offset'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            elif next_fn == 'nems.modules.levelshift.levelshift':
                layer['act'] = 'identity'
                if use_modelspec_init:
                    c = modelspec[i + 1]['phi']['level'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            else:
                layer['act'] = 'identity'
                if use_modelspec_init:
                    c = np.zeros((1,1)).astype('float32')
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))

            layer['rank'] = None  # we're handling rank with explicit spectral filters
            if 'filter_bank' in m['fn']:
                bank_count = m['fn_kwargs']['bank_count']
                layer['n_kern'] = bank_count
                layer['rate'] = m['fn_kwargs'].get('rate', 1)

                if True:
                    # temporary testing
                    layer['type'] = 'conv_bank_1d'
                    log.info('using conv_bank_1d')
                    c = np.fliplr(m['phi']['coefficients']).astype('float32').T
                    chan_count = int(c.shape[1]/bank_count)
                    c = np.reshape(c, (1, c.shape[0], bank_count, chan_count))
                else:
                    layer['type'] = 'conv_bank'
                    c = np.fliplr(m['phi']['coefficients']).astype('float32').T
                    chan_count = int(c.shape[1]/bank_count)
                    c = np.reshape(c, (c.shape[0], chan_count, 1, bank_count))
            else:
                layer['type'] = 'conv'
                bank_count = 1
                layer['n_kern'] = 1
                c = np.fliplr(m['phi']['coefficients']).astype('float32').T
                chan_count = int(c.shape[1]/bank_count)
                c = np.reshape(c, (c.shape[0], chan_count, bank_count))

            if use_modelspec_init & (np.sum(np.abs(c)) > 0):
                layer['init_W'] = c
            elif use_modelspec_init:
                c[0, 0, :, :]=1
                layer['init_W'] = c

            layers.append(layer)

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            layer = {}
            layer['time_win_sec'] = 1 / fs
            if next_fn == 'nems.modules.nonlinearity.relu':
                layer['type'] = 'reweight'
                layer['act'] = 'relu'
                if use_modelspec_init & (np.sum(np.abs(modelspec[i + 1]['phi']['offset']))>0):
                    c = -modelspec[i + 1]['phi']['offset'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            elif next_fn == 'nems.modules.levelshift.levelshift':
                layer['type'] = 'reweight'
                layer['act'] = 'identity'
                if use_modelspec_init & (np.sum(np.abs(modelspec[i + 1]['phi']['level']))>0):
                    c = modelspec[i + 1]['phi']['level'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            else:
                layer['type'] = 'reweight'
                layer['act'] = 'identity'
                if use_modelspec_init:
                    c = np.zeros((1, 1)).astype('float32')
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))

            layer['n_kern'] = m['phi']['coefficients'].shape[0]
            if use_modelspec_init & (np.sum(np.abs(m['phi']['coefficients']))>0):
                #m['phi']['coefficients'] = net_layer_vals[current_layer]['W'][0, :, :].T
                c = m['phi']['coefficients'].astype('float32').T
                layer['init_W'] = np.reshape(c,(1,c.shape[0],c.shape[1]))
            #layer['rank'] = None  # P['rank']
            layers.append(layer)

        elif m['fn'] in ['nems.modules.weight_channels.gaussian']:
            layer = {}
            layer['time_win_sec'] = 1 / fs
            if next_fn == 'nems.modules.nonlinearity.relu':
                layer['type'] = 'reweight-gaussian'
                layer['act'] = 'relu'
                if use_modelspec_init:
                    c = -modelspec[i + 1]['phi']['offset'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            else:
                layer['type'] = 'reweight-gaussian'
                layer['act'] = 'identity'
                if use_modelspec_init:
                    c = np.zeros((1, 1)).astype('float32')
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            layer['n_kern'] = m['phi']['mean'].shape[0]

            if use_modelspec_init:
                c = m['phi']['mean'].astype('float32')
                layer['init_m'] = np.reshape(c, (1, 1, c.shape[0]))
                c = m['phi']['sd'].astype('float32')
                layer['init_s'] = np.reshape(c, (1, 1, c.shape[0])) * 10

            #layer['rank'] = None  # P['rank']
            layers.append(layer)

        else:
            raise ValueError("fn %s not supported", m['fn'])

    #print(layers)
    return layers

def cnn2modelspec(net, modelspec):
    """
    pass TF cnn fit back into modelspec phi.
    TODO: Generate new modelspec if not provided
    TODO: Make sure that the dimension mappings work reliably for filter banks and such
    """
    raise Warning("DEPRECATED?")
    net_layer_vals = net.layer_vals()
    current_layer = 0
    for i, m in enumerate(modelspec):
        log.info('cnn2modelspec: ' + m['fn'])
        if i < len(modelspec)-1:
            next_fn = modelspec[i+1]['fn']
        else:
            next_fn = None
        if m['fn'] == 'nems.modules.nonlinearity.relu':
            pass  # already handled

        elif 'levelshift' in m['fn']:
            pass  # already handled

        elif m['fn'] in ['nems.modules.fir.basic']:
            m['phi']['coefficients'] = np.fliplr(net_layer_vals[current_layer]['W'][:,:,0].T)
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            elif next_fn == 'nems.modules.levelshift.levelshift':
                modelspec[i+1]['phi']['level'] = net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1

        elif m['fn'] in ['nems.modules.fir.filter_bank']:
            #m['phi']['coefficients'] = np.fliplr(net_layer_vals[current_layer]['W'][:,0,0,:].T)
            if net_layer_vals[current_layer]['W'].shape[1]>1:
                # new depthwise_conv2d
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[current_layer]['W'][0,:,:,0].T)
            else:
                # inefficient conv2d
                m['phi']['coefficients'] = np.fliplr(net_layer_vals[current_layer]['W'][:,0,0,:].T)
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            elif next_fn == 'nems.modules.levelshift.levelshift':
                modelspec[i+1]['phi']['level'] = net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            m['phi']['coefficients'] = net_layer_vals[current_layer]['W'][0,:,:].T
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            elif next_fn == 'nems.modules.levelshift.levelshift':
                modelspec[i+1]['phi']['level'] = net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            modelspec[i]['phi']['offset'] = net_layer_vals[current_layer]['b'][0, :, :].T
            #log.info(net_layer_vals[current_layer])
            current_layer += 1

        elif m['fn'] in ['nems.modules.weight_channels.gaussian']:
            modelspec[i]['phi']['mean'] = net_layer_vals[current_layer]['m'][0, 0, :].T
            modelspec[i]['phi']['sd'] = net_layer_vals[current_layer]['s'][0, 0, :].T / 10
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            elif next_fn == 'nems.modules.levelshift.levelshift':
                modelspec[i+1]['phi']['level'] = net_layer_vals[current_layer]['b'][0,:,:].T
            #log.info(net_layer_vals[current_layer])
            current_layer += 1
        else:
            raise ValueError("fn %s not supported", m['fn'])

    return modelspec
