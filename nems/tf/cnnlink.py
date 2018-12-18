"""
Tools for mapping NEMS modelspecs to and from Tensorflow CNNs
Uses Sam Norman-Haignere's CNN library as a front end for TF

"""
import tensorflow as tf
import numpy as np

import nems
import nems.modelspec as ms
import nems.tf.cnn as cnn

modelspecs_dir = nems.get_setting('NEMS_RESULTS_DIR')

def modelspec2cnn(modelspec, data_dims=1, n_inputs=18, fs=100, net_seed=1):
    """convert NEMS modelspec to TF network.
    Initialize with existing phi?
    Translations:
        wc -> reweight, identity (require lvl?)
        fir+lvl -> conv, indentity
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
        elif m['fn'] in ['nems.modules.fir.basic', 'nems.modules.fir.filter_bank']:
            layer = {}
            layer['type'] = 'conv'
            layer['time_win_sec'] = m['prior']['coefficients'][1]['mean'].shape[1] / fs
            if next_fn == 'nems.modules.nonlinearity.relu':
                layer['act'] = 'relu'
            else:
                layer['act'] = 'identity'
            layer['n_kern'] = 1 # m['prior']['coefficients'][1]['mean'].shape[0]
            layer['rank'] = None  # P['rank']
            layers.append(layer)

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            layer = {}
            layer['type'] = 'reweight-positive'
            layer['time_win_sec'] = 1 / fs
            if next_fn == 'nems.modules.nonlinearity.relu':
                layer['act'] = 'relu'
            else:
                layer['act'] = 'identity'
            layer['n_kern'] = m['prior']['coefficients'][1]['mean'].shape[0]
            #layer['rank'] = None  # P['rank']
            layers.append(layer)

        else:
            raise ValueError("fn %s not supported", m['fn'])

    # NOT CLEAR IF THIS IS BEST DONE IN
    #net1 = cnn.Net(data_dims, n_inputs, fs, layers, seed=net_seed, log_dir=modelspecs_dir)
    #net1.build()

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
                modelspec[i+1]['phi']['offset'] = net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1

        elif m['fn'] in ['nems.modules.weight_channels.basic']:
            m['phi']['coefficients'] = net_layer_vals[current_layer]['W'][0,:,:].T
            if next_fn == 'nems.modules.nonlinearity.relu':
                modelspec[i+1]['phi']['offset'] = net_layer_vals[current_layer]['b'][0,:,:].T
            current_layer += 1
        else:
            raise ValueError("fn %s not supported", m['fn'])

    return modelspec