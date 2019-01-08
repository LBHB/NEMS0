import time
import hashlib
import json
import os
from collections import Sequence

import numpy as np
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)


def iso8601_datestring():
    '''
    Returns a string containing the present date as a string.
    '''
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


def recording_filename_hash(name, meta, uri_path='', uncompressed=False):
    """
    name : string
    meta : dictionary (string keys only?)
    uri_path : base path
    uncompressed : boolean
        if True, return .tgz file
        if False, return path to folder

    returns
       guessed_filename : string
        <uri_path>/<batch>/<name>_<meta hash>.tgz

    hashing function to generate recording filenames
    JSON encode meta, then append hash to name
    """
    meta_hash = hashlib.sha1(json.dumps(meta, sort_keys=True).encode('utf-8')).hexdigest()

    if uncompressed:
        guessed_filename = name + "_" + meta_hash + os.sep
    else:
        guessed_filename = name + "_" + meta_hash + '.tgz'

    batch = meta.get('batch', None)
    if batch is not None:
        guessed_filename = os.path.join(str(batch), guessed_filename)

    if uri_path is not None and uri_path != '':
        guessed_filename = os.path.join(uri_path, guessed_filename)

    return guessed_filename


def one_zz(zerocount=1):
    """ vector of 1 followed by zerocount 0s """
    return np.concatenate((np.ones(1), np.zeros(zerocount)))


def split_to_api_and_fn(mystring):
    '''
    Returns (api, fn_name) given a string that would be used to import
    a function from a package.
    '''
    matches = mystring.split(sep='.')
    api = '.'.join(matches[:-1])
    fn_name = matches[-1]
    return api, fn_name


def shrinkage(mH, eH, sigrat=1, thresh=0):
    """
    apply shrinkage transformation to estimated mean value (mH),
    based on the relative size of the standard error (eH)
    """

    smd = np.abs(mH) / (eH + np.finfo(float).eps * (eH == 0)) / sigrat

    if thresh:
        hf = mH * (smd > 1)
    else:
        smd = 1 - np.power(smd, -2)
        smd = smd * (smd > 0)
        # smd[np.isnan(smd)]=0
        hf = mH * smd

    return hf


def progress_fun():
    """
    This function can be redirected to a function that tracks progress
    externallly, eg, in a queueing system
    """
    pass


def find_module(query, modelspec, find_all_matches=False, key='fn'):
    """
    name : string
    modelspec : NEMS modelspec (list of dictionaries)

    find_all_matches : boolean
      if True:
          returns first index where modelspec[]['fn'] contains name
          returns None if no match
      if False
          returns list of index(es) where modelspec[]['fn'] contains name
          returns empty list [] if no match
    """
    if find_all_matches:
        target_i = []
    else:
        target_i = None
    for i, m in enumerate(modelspec):
        if query in m[key]:
            if find_all_matches:
                target_i.append(i)
            else:
                target_i = i
                break

    if not target_i:
        log.debug("target_module: %s not found in modelspec.", query)
    else:
        log.debug("target_module: %s found at modelspec[%d]",
                  query, target_i)

    return target_i


def escaped_split(string, delimiter):
    '''
    Allows escaping of characters when splitting a string to a list,
    useful for some arguments in keyword strings that need to use
    underscores, decimals, hyphens, or other characters parsed by
    the keyword system.
    '''
    x = 'EXTREMELYUNLIKELYTOEVERENCOUNTERTHISEXACTSTRINGANYWHEREELSE'
    match = "\%s" % delimiter
    temp = string.replace(match, x)
    temp_split = temp.split(delimiter)
    final_split = [s.replace(x, match) for s in temp_split]

    return final_split


def escaped_join(list, delimiter):
    '''
    Allows escaping of characters when joining a list of strings,
    useful for some arguments in keyword strings that need to use
    underscores, decimals, hyphens, or other characters parsed by
    the keyword system.
    '''
    x = 'EXTREMELYUNLIKELYTOEVERENCOUNTERTHISEXACTSTRINGANYWHEREELSE'
    match = "\%s" % delimiter
    temp = [s.replace(match, x) for s in list]
    temp_join = delimiter.join(temp)
    final_join = temp_join.replace(x, match)

    return final_join


def get_channel_number(sig, channel=None):
    """
    find number of channel in signal sig that matches channel name or number
      specified in channel. default return 0
    """
    if channel is None:
        chanidx = 0
    elif sig.chans is None:
        chanidx = 0
    elif type(channel) is str:
        try:
            chanidx = sig.chans.index(channel)
        except ValueError:
            raise ValueError('channel name not in list')

    elif type(channel) is int:
        chanidx = channel
    else:
        raise ValueError('channel not integer or string')

    if chanidx >= sig.shape[0]:
        raise ValueError('channel number not valid')

    return chanidx


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def ax_remove_box(ax=None):
    """
    remove right and top lines from plot border
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def depth(s):
    '''
    Count depth of nesting of a sequence. E.g. [[[1]]] has a depth of 3.
    '''
    i = 0
    x = s
    while isinstance(x, Sequence):
        i += 1
        if x:
            x = x[0]
        else:
            x = None
    return i

