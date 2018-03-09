"""
modules/signal_mod.py

functions for modifying signals (merging, averaging, subsampling)
"""

import numpy as np
import nems.signal as signal
import nems.preprocessing as preproc


def make_state_signal(rec, signals_in=['pupil'], signals_permute=[], o='state'):
    
    x = np.ones([1,rec[signals_in[0]]._matrix.shape[1]])  # Much faster; TODO: Test if throws warnings
    ones_sig = rec[signals_in[0]]._modified_copy(x)
    ones_sig.name="baseline"
    
    state=signal.Signal.concatenate_channels([ones_sig]+[rec[x] for x in signals_in])
    
    # TODO support for signals_permute
    if len(signals_permute):
        raise ValueError("singals_permute not yet supported") 

    state.name=o

    return [state]


def average_sig(rec, i='resp', o='resp'):

    return [preproc.generate_average_sig(rec[i], new_signalname=o, 
                                 epoch_regex='^STIM_')]