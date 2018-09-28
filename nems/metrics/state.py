import copy
import numpy as np
import nems.modelspec as ms
from nems.utils import find_module, get_channel_number
import logging

log = logging.getLogger(__name__)


def state_mod_split(rec, epoch='REFERENCE', psth_name='pred', channel=None,
                    state_sig='state_raw', state_chan='pupil'):
    if 'mask' in rec.signals.keys():
        rec = rec.apply_mask()

    fs = rec[psth_name].fs

    # will default to 0 if None
    chanidx = get_channel_number(rec[psth_name], channel)
    #c = rec[psth_name].chans[chanidx]
    #full_psth = rec[psth_name].loc[c]
    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)[:, [chanidx], :] * fs

    full_var = rec[state_sig].loc[state_chan]
    folded_var = np.squeeze(full_var.extract_epoch(epoch)) * fs

    # compute the mean state for each occurrence
    g = (np.sum(np.isfinite(folded_var), axis=1) > 0)
    m = np.zeros_like(g, dtype=float)
    m[g] = np.nanmean(folded_var[g, :], axis=1)

    # compute the mean state across all occurrences
    mean = np.nanmean(m)
    gtidx = (m >= mean) & g
    if state_chan.startswith('FILE'):
        #log.info('state_chan: %s', state_chan)

        m0 = np.zeros_like(m)
        for s in rec[state_sig].chans:
            if s.startswith('FILE') and s != state_chan:
                full_var = rec[state_sig].loc[s]
                folded_var = np.squeeze(full_var.extract_epoch(epoch))
                g = (np.sum(np.isfinite(folded_var), axis=1) > 0)
                m0[g] += np.nanmean(folded_var[g, :], axis=1)

        ltidx = np.logical_not(gtidx) & np.logical_not(m0) & g
    else:
        ltidx = np.logical_not(gtidx) & g

    # low = response on epochs when state less than mean
    if (np.sum(ltidx) == 0):
        low = np.zeros_like(folded_psth[0, :, :].T) * np.nan
    else:
        low = np.nanmean(folded_psth[ltidx, :, :], axis=0).T

    # high = response on epochs when state greater than or equal to mean
    if (np.sum(gtidx) == 0):
        high = np.zeros_like(folded_psth[0, :, :].T) * np.nan
    else:
        high = np.nanmean(folded_psth[gtidx, :, :], axis=0).T

    return low, high


def state_mod_index(rec, epoch='REFERENCE', psth_name='pred', divisor=None,
                    state_sig='state_raw', state_chan='pupil'):
    """
    compute modulation index (MI) by splitting trials with state_chan into
    high and low groups (> or < median) and measuring the normalized diff
    between the two PSTHs:
       mod = (high-low) / (high+low)
    high and low PSTHs computed with state_mod_split
    """

    if type(state_chan) is list:
        if len(state_chan) == 0:
            state_chan = rec[state_sig].chans

        mod_list = [state_mod_index(rec, epoch=epoch,
                                    psth_name=psth_name,
                                    divisor=divisor,
                                    state_sig=state_sig,
                                    state_chan=s)
                    for s in state_chan]
        return np.array(mod_list)

    low, high = state_mod_split(rec, epoch=epoch, psth_name=psth_name,
                                state_sig=state_sig, state_chan=state_chan)

    if divisor is not None:
        low_denom, high_denom = state_mod_split(rec, epoch=epoch,
                                                psth_name=divisor,
                                                state_sig=state_sig,
                                                state_chan=state_chan)
        mod = np.sum(high-low) / np.sum(high_denom + low_denom)

    else:
        mod = np.sum(high - low) / np.sum(high + low)

    return mod


def j_state_mod_index(rec, epoch='REFERENCE', psth_name='pred', divisor=None,
                      state_sig='state_raw', state_chan='pupil', njacks=20):
    """
    Break into njacks jackknife sets and compute state_mod_index for each.
    Use new mask on each jackknife to pass into state_mod_index
    """

    channel_count = len(rec['resp'].chans)

    if (type(state_chan) == list) & (len(state_chan) == 0):
        state_chans = len(rec[state_sig].chans)
    else:
        state_chans = 1

    mi = np.zeros((channel_count, state_chans))
    ee = np.zeros((channel_count, state_chans))

    for i in range(channel_count):
        ff = rec['mask'].as_continuous()

        if (np.sum(ff) == 0):
            mi[i] = 0
            ee[i] = 0
        else:
            length = rec.apply_mask()['resp'].as_continuous().shape[-1]
            chunksize = int(np.ceil(length / njacks / 10))
            chunkcount = int(np.ceil(length / chunksize / njacks))
            idx = np.zeros((chunkcount, njacks, chunksize))
            for jj in range(njacks):
                idx[:, jj, :] = jj
            idx = np.reshape(idx, [-1])[:length]

            j_mi = np.zeros((njacks, state_chans))

            for jj in range(njacks):
                ff = (idx != jj)
                new_mask = rec.apply_mask()['mask']._modified_copy(ff[np.newaxis, :])
                new_rec = rec.apply_mask().copy()
                new_rec.add_signal(new_mask)
                new_rec = new_rec.apply_mask()

                j_mi[jj, :] = state_mod_index(new_rec, epoch=epoch,
                    psth_name=psth_name, divisor=divisor, state_sig=state_sig,
                    state_chan=state_chan)

            mi[i, :] = np.nanmean(j_mi, axis=0)
            ee[i, :] = np.nanstd(j_mi, axis=0) * np.sqrt(njacks-1)

    return mi, ee


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

    return np.array(state_mod_index(newrec, epoch=epoch, psth_name=psth_name,
                           state_sig=state_sig, state_chan=state_chan))

