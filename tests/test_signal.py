import os
import json
import filecmp
import pytest
import numpy as np
from numpy import nan
import pandas as pd
import nems0.signal
from nems0.signal import RasterizedSignal, merge_selections, _string_syntax_valid


@pytest.fixture()
def signal(signal_name='dummy_signal', recording_name='dummy_recording', fs=50,
           nchans=3, ntimes=200):
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
        'start': [3, 15, 150],
        'end': [200, 60, 190],
        'name': ['trial', 'pupil_closed', 'pupil_closed']
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
def signal_tmpdir(tmpdir_factory):
    '''
    Test that signals object load/save methods work as intended, and
    return an example signal object for other tests.
    '''
    return tmpdir_factory.mktemp(__name__ + '_signal')


def test_signal_save_load(signal, signal_tmpdir):
    '''
    Test that signals save and load properly
    '''
#    if not os.path.exists(signal_tmpdir):
#        os.mkdir(signal_tmpdir)
    signal.save(str(signal_tmpdir), fmt='%1.3e')

    signals_found = RasterizedSignal.list_signals(str(signal_tmpdir))
    assert len(signals_found) == 1

    save_directory = os.path.join(str(signal_tmpdir), signals_found[0])
    signal_loaded = RasterizedSignal.load(save_directory)

    assert np.all(signal.as_continuous() == signal_loaded.as_continuous())

    # TODO: add a test for the various signal attributes


def test_epoch_save_load(signal, signal_tmpdir):
    '''
    Test that epochs save and load properly
    '''

    before = signal.epochs

    signal.save(str(signal_tmpdir), fmt='%1.3e')
    signals_found = RasterizedSignal.list_signals(str(signal_tmpdir))
    save_directory = os.path.join(str(signal_tmpdir), signals_found[0])
    signal_loaded = RasterizedSignal.load(save_directory)

    after = signal_loaded.epochs
    print("Dataframes equal?\n"
          "Before:\n{0}\n"
          "After:\n{1}\n"
          .format(before, after))
    assert before.equals(after)


def test_as_continuous(signal):
    assert signal.as_continuous().shape == (3, 200)


def test_extract_epoch(signal):
    result = signal.extract_epoch('pupil_closed')
    assert result.shape == (2, 3, 45)


# TODO: Why is this breaking now? Related to SignalBase
#       changes?
def test_as_trials(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=10)
    result = signal.extract_epoch('trial')
    assert result.shape == (10, 3, 20)

    with pytest.raises(ValueError):
        signal.epochs = signal.trial_epochs_from_occurrences(occurrences=11)


def test_as_average_trial(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=10)
    result = signal.average_epoch('trial')
    assert result.shape == (3, 20)


def test_normalized_by_mean(signal):
    normalized_signal = signal.normalized_by_mean()
    data = normalized_signal.as_continuous()
    assert np.all(np.mean(data, axis=-1) == 0.0)
    assert np.allclose(np.std(data, axis=-1), 1.0)


def test_normalized_by_bounds(signal):
    normalized_signal = signal.normalized_by_bounds()
    data = normalized_signal.as_continuous()
    assert np.all(np.max(data, axis=-1) == 1)
    assert np.all(np.min(data, axis=-1) == -1)


def test_split_at_time(signal):
    l, r = signal.split_at_time(0.81)
    print(signal.as_continuous().shape)
    assert l.as_continuous().shape == (3, 162)
    assert r.as_continuous().shape == (3, 38)


def test_jackknife_by_epoch(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=50)
    s1 = signal.jackknife_by_epoch(10, 0, 'trial', tiled=False, invert=True)

    epoch_indices = signal.get_epoch_bounds('trial')
    subset_sig = signal.select_times(epoch_indices[:10])
    jack1 = subset_sig.jackknife_by_epoch(10, 0, 'trial', tiled=False, invert=False)
    assert s1.as_continuous().shape == (3, 200)  # shape shouldn't change
    assert(1770.0 == np.nansum(s1.as_continuous()[:]))
    # Should nan 10% of subsetted data - not 10% of non-subset data
    assert(sum(~np.isnan(jack1.as_continuous().flatten()))/jack1.as_continuous().size ==0.9)

def test_jackknife_by_time(signal):
    jsig = signal.jackknife_by_time(20, 2)
    isig = signal.jackknife_by_time(20, 2, invert=True)

    jdata = jsig.as_continuous()
    idata = isig.as_continuous()
    assert jdata.shape == (3, 200)
    assert idata.shape == (3, 200)

    assert np.sum(np.isnan(jdata)) == 30
    assert np.sum(np.isnan(idata)) == 570


def test_concatenate_time(signal):
    sig1 = signal
    sig2 = sig1.jackknife_by_time(20, 2)
    sig3 = RasterizedSignal.concatenate_time([sig1, sig2])
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (3, 400)


def test_concatenate_channels(signal):
    sig1 = signal
    sig2 = sig1.jackknife_by_time(20, 2)
    sig3 = RasterizedSignal.concatenate_channels([sig1, sig2])
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (6, 200)


def test_add_epoch(signal):
    epoch = np.array([[0, 200]])
    signal.add_epoch('experiment', epoch)
    assert len(signal.epochs) == 4
    assert np.all(signal.get_epoch_bounds('experiment') == epoch)


def test_merge_selections(signal):
    signals = []
    for i in range(5):
        jk = signal.jackknife_by_time(5, i, invert=True)
        signals.append(jk)

    merged = merge_selections(signals)

    # merged and signal should be identical
    assert np.sum(np.isnan(merged.as_continuous())) == 0
    assert np.array_equal(signal.as_continuous(), merged.as_continuous())
    assert signal.epochs.equals(merged.epochs)

    # This should not throw an exception
    merge_selections([signal, signal, signal])

    normalized = signal.normalized_by_mean()

    # This SHOULD throw an exception because they totally overlap
    with pytest.raises(ValueError):
        merge_selections([signal, normalized])

    jk2 = normalized.jackknife_by_time(10, 2, invert=True)
    jk3 = signal.jackknife_by_time(10, 3, invert=True)
    jk4 = signal.jackknife_by_time(10, 4, invert=True)

    # This will NOT throw an exception because they don't overlap
    merged = merge_selections([jk2, jk3])
    merged = merge_selections([jk2, jk4])

    # This SHOULD throw an exception
    with pytest.raises(ValueError):
        merged = merge_selections([signal, jk2])


def test_extract_channels(signal):
    two_sig = signal.extract_channels(['chan0', 'chan1'])
    assert two_sig.shape == (2, 200)
    one_sig = signal.extract_channels(['chan2'])
    assert one_sig.shape == (1, 200)
    recombined = RasterizedSignal.concatenate_channels([two_sig, one_sig])
    before = signal.as_continuous()
    after = recombined.as_continuous()
    assert np.array_equal(before, after)


def test_string_syntax_valid(signal):
    assert(_string_syntax_valid('this_is_fine'))
    assert(_string_syntax_valid('THIS_IS_FINE_TOO'))
    assert(not _string_syntax_valid('#But this is not ok'))


def test_jackknifes_by_epoch(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=50)
    for est, val in signal.jackknifes_by_epoch(10, 'trial'):
        print(np.nansum(est.as_continuous()[:]),
              np.nansum(val.as_continuous()[:]),)
    # This is not much of a test -- I'm just running the generator fn!
    assert(True)


@pytest.mark.skip
def test_iloc(signal):
    s = signal.iloc[:1, :10]
    assert s.as_continuous().shape == (1, 10)
    assert s.chans == ['chan0']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:1, :10])

    s = signal.iloc[:, :10]
    assert s.as_continuous().shape == (3, 10)
    assert s.chans == ['chan0', 'chan1', 'chan2']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:, :10])

    s = signal.iloc[1:, :100]
    assert s.as_continuous().shape == (2, 100)
    assert s.chans == ['chan1', 'chan2']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[1:, :100])

    s = signal.iloc[1]
    assert s.as_continuous().shape == (1, 200)
    assert s.chans == ['chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[1][np.newaxis])

    # Test some special-case indexing
    s = signal.iloc[:, 1]
    assert s.as_continuous().shape == (3, 1)
    assert s.chans == ['chan0', 'chan1', 'chan2']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:, 1][:, np.newaxis])

    s = signal.iloc[1, :]
    assert s.as_continuous().shape == (1, 200)
    assert s.chans == ['chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[1, :][np.newaxis])

    s = signal.iloc[1, 1]
    assert s.as_continuous().shape == (1, 1)
    assert s.chans == ['chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[1, 1])

    s = signal.iloc[[0, 2], 1]
    assert s.as_continuous().shape == (2, 1)
    assert s.chans == ['chan0', 'chan2']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[[0, 2], 1][:, np.newaxis])

    with pytest.raises(IndexError):
        assert signal.iloc[None, 1, 1]

    with pytest.raises(IndexError):
        assert signal.iloc[:, [1, 2]]


@pytest.mark.skip
def test_loc(signal):
    s = signal.loc['chan1']
    assert s.as_continuous().shape == (1, 200)
    assert s.chans == ['chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[1])

    s = signal.loc[['chan1', 'chan2']]
    assert s.as_continuous().shape == (2, 200)
    assert s.chans == ['chan1', 'chan2']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[[1, 2]])

    s = signal.loc[:, 1.5:2]
    assert s.as_continuous().shape == (3, 25)
    assert s.chans == ['chan0', 'chan1', 'chan2']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:, 75:100])

    s = signal.loc['chan1', 1.5:2]
    assert s.as_continuous().shape == (1, 25)
    assert s.chans == ['chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[1, 75:100])

    s = signal.loc[:'chan1', 1.5:2]
    assert s.as_continuous().shape == (2, 25)
    assert s.chans == ['chan0', 'chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:2, 75:100])

    s = signal.loc[:'chan1', 1.5]
    assert s.as_continuous().shape == (2, 1)
    assert s.chans == ['chan0', 'chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:2, 75][:, np.newaxis])

    s = signal.loc[:'chan1', :1.5]
    assert s.as_continuous().shape == (2, 75)
    assert s.chans == ['chan0', 'chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:2, :75])

    s = signal.loc[:'chan1', 2.5:]
    assert s.as_continuous().shape == (2, 75)
    assert s.chans == ['chan0', 'chan1']
    assert np.allclose(s.as_continuous(),
                       signal.as_continuous()[:2, 125:])

    with pytest.raises(IndexError):
        signal.loc[:, :, :]

    with pytest.raises(IndexError):
        signal.loc[:'chan1', [1.5, 2]]

    s = signal.loc['chan1', 0.732:]
    s_epochs = s.extract_epoch('pupil_closed')
    assert s_epochs.shape == (1, 1, 40)
    signal_epochs = signal.loc['chan1'].extract_epoch('pupil_closed')
    assert np.allclose(s_epochs, signal_epochs[1, :, :40])


def test_rasterized_signal_subset(signal):
    subset = signal.select_times([(0, 0.2), (0.3, 2)])
    assert subset.as_continuous().shape == (3, 95)
    epoch_subset = subset.extract_epoch('pupil_closed')
    assert epoch_subset.shape == (1, 3, 45)
    assert np.all(epoch_subset[0] == signal.extract_epoch('pupil_closed')[0])
    with pytest.raises(IndexError):
        epoch_subset = subset.extract_epoch('trial')
    assert subset.average_epoch('pupil_closed').shape == (3, 45)


def test_epoch_to_signal(signal):
    s = signal.epoch_to_signal('pupil_closed')
    assert s.as_continuous().shape == (1, 200)
    assert s.as_continuous().sum() == 85
