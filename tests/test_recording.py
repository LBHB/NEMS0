import numpy as np
import pandas as pd
import pytest
import unittest
from nems.recording import Recording
from nems.signal import RasterizedSignal

@pytest.fixture()
def signal1(signal_name='dummy_signal_1', recording_name='dummy_recording', fs=50,
           nchans=3, ntimes=250):
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
        'start': [3, 15, 150, 200],
        'end': [200, 60, 190, 250],
        'name': ['trial', 'pupil_closed', 'pupil_closed', 'trial2']
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
def signal2(signal_name='dummy_signal_2', recording_name='dummy_recording', fs=50,
           nchans=3, ntimes=250):
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
        'start': [3, 15, 150, 200],
        'end': [200, 60, 190, 250],
        'name': ['trial', 'pupil_closed', 'pupil_closed', 'trial2']
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
def recording(signal1, signal2):
    signals = {signal1.name: signal1,
               signal2.name: signal2}
    return Recording(signals)


def test_select_times(recording):
    '''
    Test that we can pull out select times in a given recording
    '''
    bounds = recording['dummy_signal_1'].get_epoch_bounds('trial2')
    newrec = recording.select_times(bounds)
    
    # assert that pulls the correct length of data
    assert newrec['dummy_signal_1'].as_continuous().shape == (3, 50) 
    
    # assert that epochs outside of this time window no longer exist
    with pytest.raises(IndexError):
        newrec['dummy_signal_1'].extract_epoch('pupil_closed')

    
def test_recording_loading():
    '''
    Test the loading and saving of files to various HTTP/S3/File routes.
    '''
    # NOTE: FOR THIS TEST TO SUCCEED, /auto/data/tmp/recordings/ must not have
    # blah.tar.gz or TAR010c-18-1.tar.gz in it.

    # Local filesystem
    # rec0 = Recording.load("/home/ivar/git/nems/signals/TAR010c-18-1.tar.gz")
    rec0 = Recording.load("/auto/data/nems_db/recordings/eno052d-a1.tgz")
    rec2 = Recording.load("file:///auto/data/nems_db/recordings/eno052d-a1.tgz")

    # HTTP
    rec3 = Recording.load("http://hyrax.ohsu.edu:3000/recordings/eno052d-a1.tgz")
    rec4 = Recording.load("http://hyrax.ohsu.edu:3000/baphy/294/eno052d-a1?stim=0&pupil=0")

    # S3
    # Direct access (would need AWS CLI lib? Maybe not best idea!)
    # TODO: Requires s3 credentials in environment. Probably best if on
    #       server only; put in nems_db?
    #rec5 = Recording.load('s3://mybucket/myfile.tar.gz')

    # Indirect access via http:
    rec6 = Recording.load("https://s3-us-west-2.amazonaws.com/nemspublic/"
                          "sample_data/eno052d-a1.tgz")

    # Save to a specific tar.gz file
    rec0.save('/tmp/tmp.tar.gz')

    # Save in a newly created directory under /tmp
    rec0.save('/tmp/', uncompressed=True)

    # Save ina  newly created tar.gz under /home/ivar/tmp
    rec0.save('file:///tmp/')

    # TODO: these will fail if test has already been run
    #if not rec0.save('http://hyrax:3000/recordings/blah.tar.gz'):
    #    print('Error saving to explicit file URI')
    #    assert 0

    #if not rec0.save('http://hyrax:3000/recordings/'):
    #    print('Error saving to a directory URI')
    #    assert 0

def test_recording_from_arrays():
    # need a list of array-like data structures
    x = np.random.rand(3, 200)
    y = np.random.rand(1, 200)
    z = np.random.rand(5, 200)
    arrays = [x, y, z]
    # a name for the recording that will hold the signals
    rec_name = 'testing123'
    # the sampling rate for the signals, or a list of
    # individual sampling rates (if different)
    fs = [100, 100, 200]
    # a list of signal names (optional, but preferred)
    names = ['stim', 'resp', 'reference']
    # a list of keyword arguments for each signal,
    # such as channel names or epochs (also optional)
    kwargs = [{'chans': ['2kHz', '4kHz', '8kHz']},
              {'chans': ['spike_rate']},
              {'meta': {'experiment': 'oddball_2'},
               'chans': ['One', 'Two', 'Three', 'Four', 'Five']}]
    rec = Recording.load_from_arrays(arrays, rec_name, fs, sig_names=names,
                                     signal_kwargs = kwargs)
    # should also work with integer fs instead of list
    rec = Recording.load_from_arrays(arrays, rec_name, 100, sig_names=names,
                                     signal_kwargs = kwargs)

    # All signal names should be present in recording signals dict
    contains = [(n in rec.signals.keys()) for n in names]
    assert not (False in contains)

    bad_names = ['stim']
    # should get an error now since len(names)
    # doesn't match up with len(arrays)
    with pytest.raises(ValueError):
        rec = Recording.load_from_arrays(arrays, rec_name, fs,
                                         sig_names=bad_names,
                                         signal_kwargs=kwargs)