import numpy as np
import pickle

def demo_loader(datafile=None, **context):

    # load stim and resp matrices from pkl files
    if datafile is None:
        datafile = nems.get_setting('NEMS_RECORDINGS_DIR') + '/TAR010c-18-1.pkl'

    with open(datafile, 'rb') as f:  # Python 3: open(..., 'rb')
        cellid, recname, fs, X, Y, epochs = pickle.load(f)

    res = {}
    res['fs'] = fs
    res['resp'] = Y
    res['stim'] = X
    res['epochs'] = epochs

    res['resp_labels'] = [cellid]
    stim_labels = np.round(np.exp(np.linspace(np.log(2), np.log(20), res['stim'].shape[0])), 1)
    res['stim_labels'] = [str(s) for s in stim_labels]

    return res


def load_polley_data(respfile=None, stimfile=None, exptid="RECORDING", channel_num=0, **context):

    # 2p data from Polley Lab at EPL
    if respfile is None:
        respfile = '/Users/svd/data/data_nems_2p/neurons.csv'
    if stimfile is None:
        stimfile = '/Users/svd/data/data_nems_2p/stim_spectrogram.csv'

    res = {}
    res['fs'] = 30
    res['resp'] = np.genfromtxt(respfile, delimiter=',')
    res['stim'] = np.genfromtxt(stimfile, delimiter=',')

    neuron_count, T = res['resp'].shape
    stim_labels = np.exp(np.linspace(np.log(4), np.log(64), res['stim'].shape[0]))
    res['stim_labels'] = [str(s) for s in stim_labels]
    res['resp_labels'] = ["{}-{:03d}".format(exptid, i) for i in range(neuron_count)]

    return res
