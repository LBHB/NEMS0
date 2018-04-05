import numpy as np
import pytest
from nems.recording import Recording


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