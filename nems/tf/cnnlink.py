
import nems
import nems.modelspec as ms
import nems.tf.cnn as cnn
import tensorflow as tf

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
            layer['type'] = 'reweight'
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


def cnn2modelspec(Net, modelspec=None):
    """pass TF network fit back into modelspec phi.
    Generate new modelspec if not provided"""

    pass