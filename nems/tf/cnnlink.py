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

modelspecs_dir = nems.get_setting('NEMS_RESULTS_DIR')

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
    layers = []
    for i, m in enumerate(modelspec):
        print(m['fn'])
        if i < len(modelspec)-1:
            next_fn = modelspec[i+1]['fn']
        else:
            next_fn = None

        if m['fn'] == 'nems.modules.nonlinearity.relu':
            pass # already handled

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            layer = {}
            layer['type'] = 'dlog'
            layer['time_win_sec'] = 1 / fs
            layer['act'] = ''
            layer['n_kern'] = 1  # m['prior']['coefficients'][1]['mean'].shape[0]
            layer['rank'] = None  # P['rank']
            if use_modelspec_init:
                c = np.log(10 ** m['phi']['offset'].astype('float32').T)
                layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))

            layers.append(layer)

        elif m['fn'] in ['nems.modules.fir.basic', 'nems.modules.fir.filter_bank']:
            layer = {}
            layer['type'] = 'conv'
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
            layer['n_kern'] = 1 # m['prior']['coefficients'][1]['mean'].shape[0]
            layer['rank'] = None  # P['rank']

            #m['phi']['coefficients'] = np.fliplr(net_layer_vals[current_layer]['W'][:,:,0].T)

            if use_modelspec_init:
                c = np.fliplr(m['phi']['coefficients']).astype('float32').T
                layer['init_W'] = np.reshape(c,(c.shape[0],c.shape[1], 1))
            layers.append(layer)

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            layer = {}
            layer['time_win_sec'] = 1 / fs
            if next_fn == 'nems.modules.nonlinearity.relu':
                layer['type'] = 'reweight'
                layer['act'] = 'relu'
                if use_modelspec_init:
                    c = -modelspec[i + 1]['phi']['offset'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            elif next_fn == 'nems.modules.levelshift.levelshift':
                layer['type'] = 'reweight'
                layer['act'] = 'identity'
                if use_modelspec_init:
                    c = modelspec[i + 1]['phi']['level'].astype('float32').T
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))
            else:
                layer['type'] = 'reweight'
                layer['act'] = 'identity'
                if use_modelspec_init:
                    c = np.zeros((1, 1)).astype('float32')
                    layer['init_b'] = np.reshape(c, (1, c.shape[0], c.shape[1]))

            layer['n_kern'] = m['phi']['coefficients'].shape[0]
            if use_modelspec_init:
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
                layer['init_s'] = np.reshape(c, (1, 1, c.shape[0]))
                #modelspec[i]['phi']['mean'] = net_layer_vals[current_layer]['m'][0, 0, :].T
                #modelspec[i]['phi']['sd'] = net_layer_vals[current_layer]['s'][0, 0, :].T / 10
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
    TODO: Make sure that the dimension mappings work reliably for filterbanks and such
    """

    net_layer_vals = net.layer_vals()
    current_layer = 0
    for i, m in enumerate(modelspec):
        print(m['fn'])
        if i < len(modelspec)-1:
            next_fn = modelspec[i+1]['fn']
        else:
            next_fn = None
        if m['fn'] == 'nems.modules.nonlinearity.relu':
            pass # already handled

        elif m['fn'] in ['nems.modules.fir.basic', 'nems.modules.fir.filter_bank']:
            m['phi']['coefficients'] = np.fliplr(net_layer_vals[current_layer]['W'][:,:,0].T)
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            m['phi']['coefficients'] = net_layer_vals[current_layer]['W'][0,:,:].T
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1

        elif m['fn'] in ['nems.modules.nonlinearity.dlog']:
            modelspec[i]['phi']['offset'] = np.log10(np.exp(net_layer_vals[current_layer]['b'][0, :, :].T))
            print(net_layer_vals[current_layer])
            current_layer += 1

        elif m['fn'] in ['nems.modules.weight_channels.gaussian']:
            modelspec[i]['phi']['mean'] = net_layer_vals[current_layer]['m'][0, 0, :].T
            modelspec[i]['phi']['sd'] = net_layer_vals[current_layer]['s'][0, 0, :].T / 10
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = -net_layer_vals[current_layer]['b'][0,:,:].T
            print(net_layer_vals[current_layer])
            current_layer += 1
        else:
            raise ValueError("fn %s not supported", m['fn'])

    return modelspec


def fit_tf(est=None, modelspec=None,
           use_modelspec_init=True, init_count=1,
           optimizer='Adam', max_iter=1000, cost_function='mse',
           **context):
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

    if (modelspec is None) or (est is None):
        raise ValueError("Parameters modelspec and est required")
    if 'mask' in est.signals.keys():
        est = est.apply_mask()

    modelspec = modelspec.copy()

    sr_Hz = est['resp'].fs
    time_win_sec = 0.1

    n_feats = est['stim'].shape[0]
    e = est['stim'].get_epoch_indices('REFERENCE')
    n_tps_per_stim = e[0][1]-e[0][0]
    # before: hard coded as n_tps_per_stim = 550

    n_stim = int(est['stim'].shape[1] / n_tps_per_stim)
    n_resp = 1
    feat_dims = [n_stim, n_tps_per_stim, n_feats]
    data_dims = [n_stim, n_tps_per_stim, n_resp]
    net1_seed = 50
    print('feat_dims ', feat_dims)
    print('data_dims ', data_dims)

    # extract stimulus matrix
    F = np.reshape(est['stim'].as_continuous().copy().T, feat_dims)
    D = np.reshape(est['resp'].as_continuous().copy().T, data_dims)

    # SKIP ? normalize to mean 0, variannce 1
    # should be taken care of in earlier normalization of stimulus
    #m_stim = np.mean(F, axis=(0, 1), keepdims=True)
    #s_stim = np.std(F, axis=(0, 1), keepdims=True)
    #F -= m_stim
    #F /= s_stim

    new_est = est.tile_views(init_count)
    r_fit = np.zeros(init_count, dtype=np.float)
    print('fitting from {} initial conditions'.format(init_count))
    modelspec0 = copy.deepcopy(modelspec)

    for i in range(init_count):
        if i > 0:
            # add a new set of fit parameters
            modelspec.raw.append(copy.deepcopy(modelspec.raw[0]))
            new_est = new_est.set_view(i)

        modelspec.set_fit(i)

        layers = modelspec2cnn(modelspec, n_inputs=n_feats, fs=est['resp'].fs, use_modelspec_init=use_modelspec_init)
        # layers = [{'act': 'identity', 'n_kern': 1,
        #  'time_win_sec': 0.01, 'type': 'reweight-positive'},
        # {'act': 'relu', 'n_kern': 1, 'rank': None,
        #  'time_win_sec': 0.15, 'type': 'conv'}]
        tf.reset_default_graph()

        seed = net1_seed + i
        net2 = cnn.Net(data_dims, n_feats, sr_Hz, layers, seed=seed, log_dir=modelspec.meta['modelpath'])
        net2.optimizer = optimizer

        net2.build()
        # net2_layer_init = net2.layer_vals()
        # print(net2_layer_init)

        train_val_test = np.zeros(data_dims[0])
        val_n = int(0.9 * data_dims[0])
        train_val_test[val_n:] = 1
        train_val_test = np.roll(train_val_test, int(data_dims[0]/init_count*i))

        #import pdb
        #pdb.set_trace()

        net2.train(F, D, max_iter=max_iter, train_val_test=train_val_test, learning_rate=0.01)
        modelspec = cnn2modelspec(net2, modelspec)

        new_est = modelspec.evaluate(new_est)
        r_fit[i], se_fit = nmet.j_corrcoef(new_est, 'pred', 'resp')

        nems.utils.progress_fun()

    # figure out which modelspec does best on fit data
    print('r_fit range: ', r_fit)
    ibest = np.argmax(r_fit)
    print("Best fit_index is {} r_fit={:.3f}".format(ibest, r_fit[ibest]))
    modelspec = modelspec.copy(fit_index=ibest)

    elapsed_time = (time.time() - start_time)
    modelspec.meta['fitter'] = 'fit_tf'
    modelspec.meta['fit_time'] = elapsed_time
    # ms.set_modelspec_metadata(modelspec, 'n_parms',
    #                           len(improved_sigma))

    # import pdb
    # pdb.set_trace()

    return {'modelspec': modelspec, 'new_est': new_est}
