import pytest

import numpy as np
import pandas as pd

from nems0.signal import RasterizedSignal
from nems0.recording import Recording
from nems0.preprocessing import average_away_epoch_occurrences


def make_signal(signal_name='dummy_signal_1', recording_name='dummy_recording',
            fs=50, nchans=3, ntimes=300):
    '''
    Generates a dummy signal with a predictable structure (every element
    increases by 1) that is useful for testing.
    '''
    # Generate a numpy array that's incrementially increasing across channels,
    # then across timepoints, by 1.
    c = np.arange(nchans, dtype=np.float)
    t = np.arange(ntimes, dtype=np.float)
    data = c[..., np.newaxis] + t*nchans

    epochs = pd.DataFrame({
        'start': [0,  50, 100, 250],
        'end':   [49, 99, 149, 299],
        'name': ['stim1', 'stim2', 'stim1', 'stim2']
        })
    epochs['start'] /= fs
    epochs['end'] /= fs
    kwargs = {
        'data': data,
        'name': signal_name,
        'recording': recording_name,
        'chans': ['chan' + str(n) for n in range(nchans)],
        'epochs': epochs,
        'fs': fs,
        'meta': {
            'for_testing': True,
            'date': "2018-01-10",
            'animal': "Donkey Hotey",
            'windmills': 'tilting'
        },
    }
    return RasterizedSignal(**kwargs)


@pytest.fixture()
def signal1():
    return make_signal(signal_name='resp',
                       recording_name='dummy_recording', fs=50, nchans=3,
                       ntimes=300)


@pytest.fixture()
def signal2():
    return make_signal(signal_name='stim',
                       recording_name='dummy_recording', fs=50, nchans=3,
                       ntimes=300)


@pytest.fixture()
def recording(signal1, signal2):
    signals = {signal1.name: signal1,
               signal2.name: signal2}
    return Recording(signals)


def test_average_away_epoch_occurrences(recording):
    averaged_recording = average_away_epoch_occurrences(recording, '^stim')
    as1 = averaged_recording['stim'].extract_epoch('stim1')
    as2 = averaged_recording['stim'].extract_epoch('stim2')
    s1 = recording['stim'].extract_epoch('stim1')
    s2 = recording['stim'].extract_epoch('stim2')

    assert as1.shape == (1, 3, 49)
    assert s1.shape == (2, 3, 49)
    assert np.all(as1[0] == np.mean(s1, axis=0))
    assert np.all(as2[0] == np.mean(s2, axis=0))

    epochs = averaged_recording['stim'].epochs[['name', 'start', 'end']]
    assert epochs.iat[0, 0] == 'stim1'
    assert epochs.iat[1, 0] == 'stim2'
    assert epochs.iat[0, 2] == 0.98
    assert epochs.iat[1, 2] == 1.96
