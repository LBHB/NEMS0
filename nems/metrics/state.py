import copy
import numpy as np
import nems.modelspec as ms
from nems.utils import find_module
import logging

log = logging.getLogger(__name__)


def state_mod_index(rec, epoch='REFERENCE', psth_name='resp',
                    state_sig='state', state_chan='pupil'):

    if type(state_chan) is list:
        if len(state_chan) == 0:
            state_chan = rec[state_sig].chans

        mod_list = [state_mod_index(rec, epoch=epoch,
                                    psth_name=psth_name,
                                    state_sig=state_sig,
                                    state_chan=s)
                    for s in state_chan]
        return mod_list

    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)

    full_var = rec[state_sig].loc[state_chan]
    folded_var = np.squeeze(full_var.extract_epoch(epoch))

    # compute the mean state for each occurrence
    g = (np.sum(np.isfinite(folded_var), axis=1) > 0)
    m = np.zeros_like(g, dtype=float)
    m[g] = np.nanmean(folded_var[g, :], axis=1)

    # compute the mean state across all occurrences
    mean = np.nanmean(m)
    gtidx = (m >= mean) & g
    ltidx = np.logical_not(gtidx) & g

    if (np.sum(ltidx) == 0) or (np.sum(gtidx) == 0):
        return np.nan

    # low = response on epochs when state less than mean
    low = np.nanmean(folded_psth[ltidx, :, :], axis=0).T

    # high = response on epochs when state greater than or equal to mean
    high = np.nanmean(folded_psth[gtidx, :, :], axis=0).T

    mod = np.sum(high - low) / np.sum(high + low)

    return mod


def single_state_mod_index(rec, modelspec, epoch='REFERENCE', psth_name='pred',
                           state_sig='state', state_chan='pupil'):

    if type(state_chan) is list:
        if len(state_chan) == 0:
            state_chan = rec[state_sig].chans

        mod_list = [single_state_mod_index(rec, modelspec, epoch=epoch,
                                           psth_name=psth_name,
                                           state_sig=state_sig,
                                           state_chan=s)
                    for s in state_chan]
        return mod_list

    sidx = find_module('state', modelspec)
    if sidx is None:
        raise ValueError("no state signal found")

    modelspec = copy.deepcopy(modelspec)

    state_chan_idx = rec[state_sig].chans.index(state_chan)
    k = np.ones(rec[state_sig].shape[0], dtype=bool)
    k[0] = False
    k[state_chan_idx] = False
    modelspec[sidx]['phi']['d'][:,k] = 0
    modelspec[sidx]['phi']['g'][:,k] = 0

    newrec = ms.evaluate(rec, modelspec)

    return state_mod_index(newrec, epoch=epoch, psth_name=psth_name,
                           state_sig=state_sig, state_chan=state_chan)

