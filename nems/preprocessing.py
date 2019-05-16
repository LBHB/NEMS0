import warnings
import copy
import logging

import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from scipy.integrate import cumtrapz
import scipy.signal as ss

import nems.epoch as ep
import nems.signal
from nems.recording import Recording

log = logging.getLogger(__name__)


def generate_average_sig(signal_to_average,
                         new_signalname='respavg', epoch_regex='^STIM_'):
    '''
    Returns a signal with a new signal created by replacing every epoch
    matched in "epoch_regex" with the average of every occurrence in that
    epoch. This is often used to make a response average signal that
    is the same length as the original signal_to_average, usually for plotting.

    Optional arguments:
       signal_to_average   The signal from which you want to create an
                           average signal. It will not be modified.
       new_signalname      The name of the new, average signal.
       epoch_regex         A regex to match which epochs to average across.
    '''

    # 1. Fold matrix over all stimuli, returning a dict where keys are stimuli
    #    and each value in the dictionary is (reps X cell X bins)
    epochs_to_extract = ep.epoch_names_matching(signal_to_average.epochs,
                                                epoch_regex)
    folded_matrices = signal_to_average.extract_epochs(epochs_to_extract)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg = signal_to_average.replace_epochs(per_stim_psth)
    respavg.name = new_signalname

    return respavg


def add_average_sig(rec, signal_to_average='resp',
                    new_signalname='respavg', epoch_regex='^STIM_'):
    '''
    Returns a recording with a new signal created by replacing every epoch
    matched in "epoch_regex" with the average of every occurrence in that
    epoch. This is often used to make a response average signal that
    is the same length as the original signal_to_average, usually for plotting.

    Optional arguments:
       signal_to_average   The signal from which you want to create an
                           average signal. It will not be modified.
       new_signalname      The name of the new, average signal.
       epoch_regex         A regex to match which epochs to average across.
    '''

    # generate the new signal by averaging epochs of the input singals
    respavg = generate_average_sig(rec[signal_to_average],
                                   new_signalname, epoch_regex)

    # Add the signal to the recording
    newrec = rec.copy()
    newrec.add_signal(respavg)

    return newrec


def average_away_epoch_occurrences(recording, epoch_regex='^STIM_'):
    '''
    Returns a recording with _all_ signals averaged across epochs that
    match epoch_regex, shortening them so that each epoch occurs only
    once in the new signals. i.e. unlike 'add_average_sig', the new
    recording will have signals 3x shorter if there are 3 occurrences of
    every epoch.

    This has advantages:
    1. Averaging the value of a signal (such as a response) in different
       occurrences will make it behave more like a linear variable with
       gaussian noise, which is advantageous in many circumstances.
    2. There will be less computation needed because the signal is shorter.

    It also has disadvantages:
    1. Stateful filters (FIR, IIR) will be subtly wrong near epoch boundaries
    2. Any ordering of epochs is essentially lost, unless all epochs appear
       in a perfectly repeated order.

    To avoid accidentally averaging away differences in responses to stimuli
    that are based on behavioral state, you may need to create new epochs
    (based on stimulus and behaviorial state, for example) and then match
    the epoch_regex to those.
    '''
    epochs = recording['resp'].epochs
    epoch_names = sorted(set(ep.epoch_names_matching(epochs, epoch_regex)))

    offset = 0
    new_epochs = [] # pd.DataFrame()
    fs = recording[list(recording.signals.keys())[0]].fs
    d = int(np.ceil(np.log10(fs))+1)
    for epoch_name in epoch_names:
        common_epochs = ep.find_common_epochs(epochs, epoch_name, d=d)
        common_epochs = common_epochs[common_epochs['name']!='TRIAL']
        query = 'name == "{}"'.format(epoch_name)
        #import pdb
        #pdb.set_trace()
        end = common_epochs.query(query).iloc[0]['end']
        common_epochs[['start', 'end']] += offset
        offset += end
        new_epochs.append(common_epochs)

    new_epochs = pd.concat(new_epochs, ignore_index=True)

    #averaged_recording = recording.copy()
    averaged_signals = {}
    for signal_name, signal in recording.signals.items():
        # TODO: this may be better done as a method in signal subclasses since
        # some subclasses may have more efficient approaches (e.g.,
        # TiledSignal)

        # Extract all occurances of each epoch, returning a dict where keys are
        # stimuli and each value in the dictionary is (reps X cell X bins)
        epoch_data = signal.rasterize().extract_epochs(epoch_names)

        # Average over all occurrences of each epoch
        data = []
        for epoch_name in epoch_names:
            epoch = epoch_data[epoch_name]

            # TODO: fix empty matrix error. do epochs align properly?
            if np.sum(np.isfinite(epoch)):
                epoch = np.nanmean(epoch, axis=0)
            else:
                epoch = epoch[0,...]

            mask = new_epochs['name'] == epoch_name
            bounds = new_epochs.loc[mask, ['start', 'end']].values
            bounds = np.round(bounds.astype(float) * signal.fs).astype(int)
            elen = bounds[0,1] - bounds[0, 0]
            if epoch.shape[-1] > elen:
                log.info('truncating epoch_data for epoch %s', epoch_name)
                epoch = epoch[..., :elen]
            elif epoch.shape[-1]<elen:
                pad = np.zeros((epoch.shape[0], elen-epoch.shape[1])) * np.nan
                epoch = np.concatenate((epoch, pad), axis=1)
                log.info('padding epoch_data for epoch %s with nan', epoch_name)

            data.append(epoch)

        data = np.concatenate(data, axis=-1)
        if data.shape[-1] != round(signal.fs * offset):
            raise ValueError('Misalignment issue in averaging signal')

        averaged_signal = signal._modified_copy(data, epochs=new_epochs)
        averaged_signals[signal_name] = averaged_signal

#        # TODO: Eventually need a smarter check for this in case it's named
#        #       something else. Basically just want to preserve spike data.
#        if signal.name == 'resp':
#            spikes = signal.copy()
#            spikes.name = signal.name + ' spikes'
#            averaged_recording.add_signal(spikes)
    averaged_recording = Recording(averaged_signals,
                                   meta=recording.meta,
                                   name=recording.name)
    return averaged_recording


def remove_invalid_segments(rec):
    """
    Currently a specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Migrate to nems_lbhb or make a more generic version
    """

    # First, select the appropriate subset of data
    rec['resp'] = rec['resp'].rasterize()
    if 'stim' in rec.signals.keys():
        rec['stim'] = rec['stim'].rasterize()

    sig = rec['resp']

    # get list of start and stop indices (epoch bounds)
    epoch_indices = np.vstack((
            ep.epoch_intersection(sig.get_epoch_indices('REFERENCE'),
                                  sig.get_epoch_indices('HIT_TRIAL')),
            ep.epoch_intersection(sig.get_epoch_indices('REFERENCE'),
                                  sig.get_epoch_indices('PASSIVE_EXPERIMENT'))))

    # Only takes the first of any conflicts (don't think I actually need this)
    epoch_indices = ep.remove_overlap(epoch_indices)

    # merge any epochs that are directly adjacent
    epoch_indices2 = epoch_indices[0:1]
    for i in range(1, epoch_indices.shape[0]):
        if epoch_indices[i, 0] == epoch_indices2[-1, 1]:
            epoch_indices2[-1, 1] = epoch_indices[i, 1]
        else:
            #epoch_indices2 = np.concatenate(
             #       (epoch_indices2, epoch_indices[i:(i + 1), :]), axis=0)
            epoch_indices2=np.append(epoch_indices2,epoch_indices[i:(i+1)], axis=0)

    # convert back to times
    epoch_times = epoch_indices2 / sig.fs

    # add adjusted signals to the recording
    newrec = rec.select_times(epoch_times)

    return newrec


def mask_all_but_correct_references(rec, balance_rep_count=False,
                                    include_incorrect=False, generate_evoked_mask=False):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """
    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()
    resp = newrec['resp']

    if balance_rep_count:

        epoch_regex = "^STIM_"
        epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
        p=resp.get_epoch_indices("PASSIVE_EXPERIMENT")
        a=resp.get_epoch_indices("HIT_TRIAL")

        epoch_list=[]
        for s in epochs_to_extract:
            e = resp.get_epoch_indices(s)
            pe = ep.epoch_intersection(e, p)
            ae = ep.epoch_intersection(e, a)
            if len(pe)>len(ae):
                epoch_list.extend(ae)
                subset=np.round(np.linspace(0,len(pe),len(ae)+1)).astype(int)
                for i in subset[:-1]:
                    epoch_list.append(pe[i])
            else:
                subset=np.round(np.linspace(0,len(ae),len(pe)+1)).astype(int)
                for i in subset[:-1]:
                    epoch_list.append(ae[i])
                epoch_list.extend(pe)

        newrec = newrec.create_mask(epoch_list)

    elif include_incorrect:
        log.info('INCLUDING ALL TRIALS (CORRECT AND INCORRECT)')
        newrec = newrec.and_mask(['REFERENCE'])

    else:
        newrec = newrec.and_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL'])
        newrec = newrec.and_mask(['REFERENCE'])

    # figure out if some actives should be masked out
#    t = ep.epoch_names_matching(resp.epochs, "^TAR_")
#    tm = [tt[:-2] for tt in t]  # trim last digits
#    active_epochs = resp.get_epoch_indices("ACTIVE_EXPERIMENT")
#    if len(set(tm)) > 1 and len(active_epochs) > 1:
#        print('Multiple targets: ', tm)
#        files = ep.epoch_names_matching(resp.epochs, "^FILE_")
#        keep_files = files
#        e = active_epochs[1]
#        for i,f in enumerate(files):
#            fi = resp.get_epoch_indices(f)
#            if any(ep.epoch_contains([e], fi, 'both')):
#                keep_files = files[:i]
#
#        print('Print keeping files: ', keep_files)
#        newrec = newrec.and_mask(keep_files)

    if 'state' in newrec.signals:
        b_states = ['far', 'hit', 'lick',
                    'puretone_trials', 'easy_trials', 'hard_trials']
        trec = newrec.copy()
        trec = trec.and_mask(['ACTIVE_EXPERIMENT'])
        st = trec['state'].as_continuous().copy()
        str = trec['state_raw'].as_continuous().copy()
        mask = trec['mask'].as_continuous()[0, :]
        for s in trec['state'].chans:
            if s in b_states:
                i = trec['state'].chans.index(s)
                m = np.nanmean(st[i, mask])
                sd = np.nanstd(st[i, mask])
                # print("{} {}: m={}, std={}".format(s, i, m, sd))
                # print(np.sum(mask))
                st[i, mask] -= m
                st[i, mask] /= sd
                str[i, mask] -= m
                str[i, mask] /= sd
        newrec['state'] = newrec['state']._modified_copy(st)
        newrec['state_raw'] = newrec['state_raw']._modified_copy(str)

    if generate_evoked_mask:
        mask = newrec['mask'].as_continuous().copy()
        padbins=int(np.round(newrec['resp'].fs * 0.1))

        preidx = resp.get_epoch_indices('PreStimSilence', mask=newrec['mask'])
        posidx = resp.get_epoch_indices('PostStimSilence', mask=newrec['mask'])
        for i,p in enumerate(posidx):
            posidx[i]=(p[0]+padbins, p[1])

        post_mask = newrec['resp'].epoch_to_signal(indices=posidx)
        pre_mask = newrec['resp'].epoch_to_signal(indices=preidx)
        #mask[post_mask.as_continuous()] = False
        ev_mask = mask.copy()
        ev_mask[pre_mask.as_continuous()] = False
        ev_mask[post_mask.as_continuous()] = False
        newrec['sp_mask'] = newrec['mask']._modified_copy(data=mask)
        newrec['ev_mask'] = newrec['mask']._modified_copy(data=ev_mask)

    return newrec


def mask_keep_passive(rec):
    """
    Mask out all times that don't fall in PASSIVE_EXPERIMENT epochs.

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """

    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    newrec = newrec.and_mask(['PASSIVE_EXPERIMENT'])

    return newrec


def mask_all_but_targets(rec):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """

    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    newrec = newrec.or_mask(['TARGET'])

    return newrec


def nan_invalid_segments(rec):
    """
    Currently a specialized signal for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Complete this function, replicate remove_invalid_segments logic
          Or delete ME
    TODO: Migrate to nems_lbhb or make a more generic version
    """

    # First, select the appropriate subset of data
    rec['resp'] = rec['resp'].rasterize()
    sig = rec['resp']

    # get list of start and stop times (epoch bounds)
    epoch_indices = np.vstack((
            ep.epoch_intersection(sig.get_epoch_bounds('HIT_TRIAL'),
                                  sig.get_epoch_bounds('REFERENCE')),
            ep.epoch_intersection(sig.get_epoch_bounds('REFERENCE'),
                                  sig.get_epoch_bounds('PASSIVE_EXPERIMENT'))))

    # Only takes the first of any conflicts (don't think I actually need this)
    epoch_indices = ep.remove_overlap(epoch_indices)

    epoch_indices2 = epoch_indices[0:1, :]
    for i in range(1, epoch_indices.shape[0]):
        if epoch_indices[i, 0] == epoch_indices2[-1, 1]:
            epoch_indices2[-1, 1] = epoch_indices[i, 0]
        else:
            epoch_indices2 = np.concatenate(
                    (epoch_indices2, epoch_indices[i: (i+1), :]), axis=0
                    )

    # add adjusted signals to the recording
    newrec = rec.nan_times(epoch_indices2)

    return newrec


def generate_stim_from_epochs(rec, new_signal_name='stim',
                              epoch_regex='^STIM_', epoch_shift=0,
                              epoch2_regex=None, epoch2_shift=0,
                              epoch2_shuffle=False,
                              onsets_only=True):

    rec = rec.copy()
    resp = rec['resp'].rasterize()

    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    sigs = []
    for e in epochs_to_extract:
        log.info('Adding to %s: %s with shift = %d',
                 new_signal_name, e, epoch_shift)
        s = resp.epoch_to_signal(e, onsets_only=onsets_only, shift=epoch_shift)
        if epoch_shift:
            s.chans[0] = "{}{:+d}".format(s.chans[0], epoch_shift)
        sigs.append(s)

    if epoch2_regex is not None:
        epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch2_regex)
        for e in epochs_to_extract:
            log.info('Adding to %s: %s with shift = %d',
                     new_signal_name, e, epoch2_shift)
            s = resp.epoch_to_signal(e, onsets_only=onsets_only,
                                     shift=epoch2_shift)
            if epoch2_shuffle:
                log.info('Shuffling %s', e)
                s = s.shuffle_time()
                s.chans[0] = "{}_shf".format(s.chans[0])
            if epoch_shift:
                s.chans[0] = "{}{:+d}".format(s.chans[0], epoch2_shift)
            sigs.append(s)

    stim = sigs[0].concatenate_channels(sigs)
    stim.name = new_signal_name

    # add_signal operates in place
    rec.add_signal(stim)

    return rec


def integrate_signal_per_epoch(rec, sig='stim', sig_out='stim_int', epoch_regex='^STIM_'):
    '''
    Calculates integral for each epoch of a signal

    if rec['mask'] exists, uses rec['mask'] == True to determine valid epochs
    '''

    newrec = rec.copy()
    sig = newrec[sig].rasterize()

    # compute PSTH response during valid trials
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(sig.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(sig.epochs, epoch_regex)

    folded_matrices = sig.extract_epochs(epochs_to_extract,
                                         mask=newrec['mask'])

    # 2. Average across reps and integrate each stim
    for k, v in folded_matrices.items():
        v = np.nanmean(v, axis=0)
        v = cumtrapz(v, dx=1/sig.fs, initial=0)
        folded_matrices[k] = v

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    new_sig = sig._modified_copy(data=np.zeros_like(sig._data))
    new_sig = new_sig.replace_epochs(folded_matrices)
    new_sig.name = sig_out

    # add the new signals to the recording
    newrec.add_signal(new_sig)

    return newrec


def generate_psth_from_resp(rec, resp_sig='resp', epoch_regex='^STIM_', smooth_resp=False):
    '''
    Estimates a PSTH from all responses to each regex match in a recording

    subtract spont rate based on pre-stim silence for ALL estimation data.

    if rec['mask'] exists, uses rec['mask'] == True to determine valid epochs
    '''

    newrec = rec.copy()
    resp = newrec[resp_sig].rasterize()

    # compute spont rate during valid (non-masked) trials
    if 'mask' in newrec.signals.keys():
        mask = newrec['mask']
    else:
        mask = None

    prestimsilence = resp.extract_epoch('PreStimSilence', mask=mask)

    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    preidx = resp.get_epoch_indices('PreStimSilence', mask=mask)
    dpre=preidx[:,1]-preidx[:,0]
    minpre=np.min(dpre)
    prebins = preidx[0][1] - preidx[0][0]
    posidx = resp.get_epoch_indices('PostStimSilence', mask=mask)
    dpos=posidx[:,1]-posidx[:,0]
    minpos=np.min(dpre)
    postbins = posidx[0][1] - posidx[0][0]
    #refidx = resp.get_epoch_indices('REFERENCE')

    # compute PSTH response during valid trials
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(resp.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)

    #import pdb
    #pdb.set_trace()
    for ename in epochs_to_extract:
        ematch = np.argwhere(resp.epochs['name']==ename)
        ff = resp.get_epoch_indices(ename, mask=mask)
        for i,fe in enumerate(ff):
            re = ((resp.epochs['name']=='REFERENCE') &
                  (resp.epochs['start']==fe[0]/resp.fs))
            pe = ep.epoch_contained(preidx, [fe])
            thispdur = np.diff(preidx[pe])

            #import pdb
            #pdb.set_trace()

            if np.sum(pe)==1 and thispdur>minpre:
                print('adjust {} to {}'.format(thispdur, minpre))
                print(resp.epochs.loc[ematch[i]])
                resp.epochs.loc[ematch[i],'start'] += (thispdur[0,0]-minpre)/resp.fs
                resp.epochs.loc[re,'start'] += (thispdur[0,0]-minpre)/resp.fs
                print(resp.epochs.loc[ematch[i]])

            pe = ep.epoch_contained(posidx, [fe])
            thispdur = np.diff(posidx[pe])
            if thispdur.shape and thispdur>minpos:
                print('adjust {} to {}'.format(thispdur, minpos))
                print(resp.epochs.loc[ematch[i]])
                resp.epochs.loc[ematch[i],'end'] -= (thispdur[0,0]-minpos)/resp.fs
                resp.epochs.loc[re,'end'] -= (thispdur[0,0]-minpos)/resp.fs
                print(resp.epochs.loc[ematch[i]])
    newrec['resp'].epochs = resp.epochs.copy()

    if 'mask' in newrec.signals.keys():
        folded_matrices = resp.extract_epochs(epochs_to_extract,
                                              mask=newrec['mask'])
    else:
        folded_matrices = resp.extract_epochs(epochs_to_extract)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    per_stim_psth_spont = dict()
    for k, v in folded_matrices.items():
        if smooth_resp:
            # replace each epoch (pre, during, post) with average
            v[:, :, :prebins] = np.nanmean(v[:, :, :prebins],
                                           axis=2, keepdims=True)
            v[:, :, prebins:(prebins+2)] = np.nanmean(v[:, :, prebins:(prebins+2)],
                                                      axis=2, keepdims=True)
            v[:, :, (prebins+2):-postbins] = np.nanmean(v[:, :, (prebins+2):-postbins],
                                                        axis=2, keepdims=True)
            v[:, :, -postbins:(-postbins+2)] = np.nanmean(v[:, :, -postbins:(-postbins+2)],
                                                          axis=2, keepdims=True)
            v[:, :, (-postbins+2):] = np.nanmean(v[:, :, (-postbins+2):],
                                                 axis=2, keepdims=True)

        per_stim_psth[k] = np.nanmean(v, axis=0) - spont_rate[:, np.newaxis]
        per_stim_psth_spont[k] = np.nanmean(v, axis=0)
        folded_matrices[k] = v

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg = resp.replace_epochs(per_stim_psth)
    respavg_with_spont = resp.replace_epochs(per_stim_psth_spont)
    respavg.name = 'psth'
    respavg_with_spont.name = 'psth_sp'

    # Fill in a all non-masked periods with 0 (presumably, these are spont
    # periods not contained within stimulus epochs), or spont rate (for the signal
    # containing spont rate)
    respavg_data = respavg.as_continuous().copy()
    respavg_spont_data = respavg_with_spont.as_continuous().copy()

    if 'mask' in newrec.signals.keys():
        mask_data = newrec['mask']._data
    else:
        mask_data = np.ones(respavg_data.shape).astype(np.bool)

    spont_periods = ((np.isnan(respavg_data)) & (mask_data==True))

    respavg_data[:, spont_periods[0,:]] = 0
    # respavg_spont_data[:, spont_periods[0,:]] = spont_rate[:, np.newaxis]

    respavg = respavg._modified_copy(respavg_data)
    respavg_with_spont = respavg_with_spont._modified_copy(respavg_spont_data)

    # add the new signals to the recording
    newrec.add_signal(respavg)
    newrec.add_signal(respavg_with_spont)
    if 'stim' in newrec.signals.keys():
        # add as channel to stim signal if it exists
        newrec = concatenate_state_channel(newrec, respavg, 'stim')
        newrec['stim'].chans[-1] = 'psth'

    if smooth_resp:
        log.info('Replacing resp with smoothed resp')
        resp = resp.replace_epochs(folded_matrices, mask=newrec['mask'])
        newrec.add_signal(resp)

    return newrec


def smooth_signal_epochs(rec, signal='resp', epoch_regex='^STIM_',
                         **context):
    """
    xforms-compatible wrapper for smooth_epoch_segments
    """

    newrec = rec.copy()

    smoothed_sig, respavg, respavg_with_spont = smooth_epoch_segments(
        newrec[signal], epoch_regex=epoch_regex, mask=newrec['mask'])

    newrec.add_signal(smoothed_sig)

    return {'rec': newrec}


def smooth_epoch_segments(sig, epoch_regex='^STIM_', mask=None):
    """
    wonky function that "smooths" signals by computing the mean of the
    pre-stim silence, onset, sustained, and post-stim silence
    Used in PSTH-based models. Duration of onset hard-coded to 2 bins
    :return: (smoothed_sig, respavg, respavg_with_spont)
    smoothed_sig - smoothed signal
    respavg - smoothed signal, averaged across all reps of matching epochs
    """

    # compute spont rate during valid (non-masked) trials
    prestimsilence = sig.extract_epoch('PreStimSilence', mask=mask)

    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    preidx = sig.get_epoch_indices('PreStimSilence', mask=mask)
    dpre=preidx[:,1]-preidx[:,0]
    minpre=np.min(dpre)
    prebins = preidx[0][1] - preidx[0][0]
    posidx = sig.get_epoch_indices('PostStimSilence', mask=mask)
    dpos=posidx[:,1]-posidx[:,0]
    minpos=np.min(dpre)
    postbins = posidx[0][1] - posidx[0][0]
    #refidx = sig.get_epoch_indices('REFERENCE')

    # compute PSTH response during valid trials
    if type(epoch_regex) == list:
        epochs_to_extract = []
        for rx in epoch_regex:
            eps = ep.epoch_names_matching(resp.epochs, rx)
            epochs_to_extract += eps

    elif type(epoch_regex) == str:
        epochs_to_extract = ep.epoch_names_matching(sig.epochs, epoch_regex)
    else:
        raise ValueError("invalid epoch_regex")

    #import pdb
    #pdb.set_trace()
    for ename in epochs_to_extract:
        ematch = np.argwhere(sig.epochs['name']==ename)
        ff = sig.get_epoch_indices(ename, mask=mask)
        for i,fe in enumerate(ff):
            re = ((sig.epochs['name'] == 'REFERENCE') &
                  (sig.epochs['start'] == fe[0]/sig.fs))
            pe = ep.epoch_contained(preidx, [fe])
            thispdur = np.diff(preidx[pe])

            if np.sum(pe)==1 and thispdur>minpre:
                print('adjust {} to {}'.format(thispdur, minpre))
                print(sig.epochs.loc[ematch[i]])
                sig.epochs.loc[ematch[i],'start'] += (thispdur[0,0]-minpre)/resp.fs
                sig.epochs.loc[re,'start'] += (thispdur[0,0]-minpre)/resp.fs
                print(sig.epochs.loc[ematch[i]])

            pe = ep.epoch_contained(posidx, [fe])
            thispdur = np.diff(posidx[pe])
            if thispdur.shape and thispdur>minpos:
                print('adjust {} to {}'.format(thispdur, minpos))
                print(sig.epochs.loc[ematch[i]])
                sig.epochs.loc[ematch[i],'end'] -= (thispdur[0,0]-minpos)/resp.fs
                sig.epochs.loc[re,'end'] -= (thispdur[0,0]-minpos)/resp.fs
                print(resp.epochs.loc[ematch[i]])

    smoothed_sig = sig.copy()
    smoothed_sig.epochs = smoothed_sig.epochs.copy()

    folded_matrices = smoothed_sig.extract_epochs(epochs_to_extract, mask=mask)

    # 2. Average over all reps of each epoch and save into dict called psth.
    per_stim_psth = dict()
    per_stim_psth_spont = dict()
    for k, v in folded_matrices.items():
        # replace each epoch (pre, during, post) with average
        v[:, :, :prebins] = np.nanmean(v[:, :, :prebins],
                                       axis=2, keepdims=True)
        v[:, :, prebins:(prebins+2)] = np.nanmean(v[:, :, prebins:(prebins+2)],
                                                  axis=2, keepdims=True)
        v[:, :, (prebins+2):-postbins] = np.nanmean(v[:, :, (prebins+2):-postbins],
                                                    axis=2, keepdims=True)
        v[:, :, -postbins:(-postbins+2)] = np.nanmean(v[:, :, -postbins:(-postbins+2)],
                                                      axis=2, keepdims=True)
        v[:, :, (-postbins+2):] = np.nanmean(v[:, :, (-postbins+2):],
                                             axis=2, keepdims=True)

        per_stim_psth[k] = np.nanmean(v, axis=0) - spont_rate[:, np.newaxis]
        per_stim_psth_spont[k] = np.nanmean(v, axis=0)
        folded_matrices[k] = v

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    log.info('Replacing resp with smoothed resp')
    smoothed_sig = smoothed_sig.replace_epochs(folded_matrices, mask=mask)

    respavg = smoothed_sig.replace_epochs(per_stim_psth)
    respavg.name = 'psth'
    respavg_with_spont = smoothed_sig.replace_epochs(per_stim_psth_spont)
    respavg_with_spont.name = 'psth_sp'

    # Fill in a all non-masked periods with 0 (presumably, these are spont
    # periods not contained within stimulus epochs), or spont rate (for the signal
    # containing spont rate)
    respavg_data = respavg.as_continuous().copy()
    respavg_spont_data = respavg_with_spont.as_continuous().copy()

    if mask is not None:
        mask_data = mask._data
    else:
        mask_data = np.ones(respavg_data.shape).astype(np.bool)

    spont_periods = ((np.isnan(respavg_data)) & (mask_data==True))

    respavg_data[:, spont_periods[0,:]] = 0
    # respavg_spont_data[:, spont_periods[0,:]] = spont_rate[:, np.newaxis]

    respavg = respavg._modified_copy(respavg_data)
    respavg_with_spont = respavg_with_spont._modified_copy(respavg_spont_data)

    return smoothed_sig, respavg, respavg_with_spont


def generate_psth_from_est_for_both_est_and_val(est, val,
                                                epoch_regex='^STIM_'):
    '''
    Estimates a PSTH from the EST set, and returns two signals based on the
    est and val, in which each repetition of a stim uses the EST PSTH?

    subtract spont rate based on pre-stim silence for ALL estimation data.
    '''

    resp_est = est['resp'].rasterize()
    resp_val = val['resp'].rasterize()

    # compute PSTH response and spont rate during those valid trials
    prestimsilence = resp_est.extract_epoch('PreStimSilence')
    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
    folded_matrices = resp_est.extract_epochs(epochs_to_extract,
                                              mask=est['mask'])

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0) - \
            spont_rate[:, np.newaxis]

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg_est = resp_est.replace_epochs(per_stim_psth)
    respavg_est.name = 'psth'

    # add signal to the est recording
    est.add_signal(respavg_est)

    respavg_val = resp_val.replace_epochs(per_stim_psth)
    respavg_val.name = 'psth'

    # add signal to the val recording
    val.add_signal(respavg_val)

    return est, val


def generate_psth_from_est_for_both_est_and_val_nfold(ests, vals,
                                                      epoch_regex='^STIM_'):
    '''
    call generate_psth_from_est_for_both_est_and_val for each e,v
    pair in ests,vals
    '''
    for e, v in zip(ests, vals):
        e, v = generate_psth_from_est_for_both_est_and_val(e, v, epoch_regex)

    return ests, vals


def resp_to_pc(rec, pc_idx=[0], resp_sig='resp', pc_sig='pca',
               pc_count=None, pc_source='all', overwrite_resp=True,
               whiten=True, **context):
    """
    generate pca signal, replace (multichannel) reference with a single
    pc channel

    """
    rec0 = rec.copy()
    if type(pc_idx) is not list:
        pc_idx = [pc_idx]
    resp = rec0[resp_sig]

    # compute duration of spont period
    d = resp.get_epoch_bounds('PreStimSilence')
    if len(d):
        PreStimSilence = np.mean(np.diff(d))
    else:
        PreStimSilence = 0
    prestimbins = int(PreStimSilence * resp.fs)

    # compute PCs only on valid (masked) times
    rec0[resp_sig] = rec0[resp_sig].rasterize()
    if 'mask' in rec0.signals:
        rec_masked = rec0.apply_mask()
    else:
        rec_masked = rec0

    rec_masked = generate_psth_from_resp(rec_masked, resp_sig=resp_sig)

    if pc_source=='all':
        D_ref = rec_masked[resp_sig].as_continuous().T
    elif pc_source=='psth':
        D_ref = rec_masked['psth'].as_continuous().T
    elif pc_source=='noise':
        D_psth = rec_masked['psth'].as_continuous().T
        D_raw = rec_masked[resp_sig].as_continuous().T
        D_ref = D_raw - D_psth
    else:
        raise ValueError('pc_source {} not supported'.format(pc_source))

    # project full response dataset to preserve time
    D = rec0[resp_sig].as_continuous().T

    if pc_count is None:
        pc_count = D_ref.shape[1]

    if False:
        # use sklearn. maybe someday
        pca = PCA(n_components=pc_count)
        pca.fit(D_ref)

        X = pca.transform(D)
        rec0[pc_sig] = rec0[resp_sig]._modified_copy(X.T)
    else:
        # each row(??) of v is a PC --weights to project into PC domain
        m = np.nanmean(D_ref, axis=0, keepdims=True)
        if whiten:
            sd = np.nanstd(D_ref, axis=0, keepdims=True)
        else:
            sd = np.ones(m.shape)

        u, s, v = np.linalg.svd((D_ref-m)/sd, full_matrices=False)
        X = (D-m) / sd @ v.T

        rec0[pc_sig] = rec0[resp_sig]._modified_copy(X.T)

        r = rec0[pc_sig].extract_epoch('REFERENCE', mask=rec0['mask'])
        mr = np.mean(r, axis=0)
        spont = np.mean(mr[:,:prestimbins],axis=1,keepdims=True)
        mr -= spont
        vs = np.sign(np.sum(mr[:, prestimbins:(prestimbins+10)], axis=1, keepdims=True))
        v *= vs
        X = (D-m) / sd @ v.T

        rec0[pc_sig] = rec0[resp_sig]._modified_copy(X.T)

#    r = rec0[pc_sig].extract_epoch('REFERENCE')
#    mr=np.mean(r,axis=0)
#    spont=np.mean(mr[:,:50],axis=1,keepdims=True)
#    mr-=spont
#    plt.figure()
#    plt.plot(mr[:5,:].T)
#    plt.legend(('1','2','3','4','5'))

    rec0.meta['pc_weights'] = v
    rec0.meta['pc_mag'] = s
    if overwrite_resp:
        rec0[resp_sig] = rec0[resp_sig]._modified_copy(X[:, pc_idx].T)
        rec0.meta['pc_idx'] = pc_idx

    return {'rec': rec0}


def make_state_signal(rec, state_signals=['pupil'], permute_signals=[],
                      new_signalname='state'):
    """
    generate state signal for stategain.S/sdexp.S models

    valid state signals include (incomplete list):
        pupil, pupil_ev, pupil_bs, pupil_psd
        active, each_file, each_passive, each_half
        far, hit, lick, p_x_a

    TODO: Migrate to nems_lbhb or make a more generic version
    """

    newrec = rec.copy()
    resp = newrec['resp'].rasterize()

    # normalize mean/std of pupil trace if being used
    if ('pupil' in state_signals) or ('pupil2' in state_signals) or \
        ('pupil_ev' in state_signals) or ('pupil_bs' in state_signals) or \
        ('pupil_stim' in state_signals) or ('pupil_x_population' in state_signals):
        # save raw pupil trace
        newrec["pupil_raw"] = newrec["pupil"].copy()
        # normalize min-max
        p = newrec["pupil"].as_continuous().copy()
        # p[p < np.nanmax(p)/5] = np.nanmax(p)/5
        p -= np.nanmean(p)
        p /= np.nanstd(p)
        newrec["pupil"] = newrec["pupil"]._modified_copy(p)

        if ('pupil2') in state_signals:
            newrec["pupil2"] = newrec["pupil"]._modified_copy(p ** 2)
            newrec["pupil2"].chans = ['pupil2']

    if ('pupil_psd') in state_signals:
        pup = newrec['pupil'].as_continuous().copy()
        fs = newrec['pupil'].fs
        # get spectrogram of pupil
        nperseg = int(60*fs)
        noverlap = nperseg-1
        f, time, Sxx = ss.spectrogram(pup.squeeze(), fs=fs, nperseg=nperseg,
                                      noverlap=noverlap)
        max_chan = 4 # (np.abs(f - 0.1)).argmin()
        # Keep only first five channels of spectrogram
        #f = interpolate.interp1d(np.arange(0, Sxx.shape[1]), Sxx[:max_chan, :], axis=1)
        #newspec = f(np.linspace(0, Sxx.shape[-1]-1, pup.shape[-1]))
        pad1 = np.ones((max_chan,int(nperseg/2)))*Sxx[:max_chan,[0]]
        pad2 = np.ones((max_chan,int(nperseg/2-1)))*Sxx[:max_chan,[-1]]
        newspec = np.concatenate((pad1,Sxx[:max_chan, :],pad2), axis=1)

        # = np.concatenate((Sxx[:max_chan, :], np.tile(Sxx[:max_chan,-1][:, np.newaxis], [1, noverlap])), axis=1)
        newspec -= np.nanmean(newspec, axis=1, keepdims=True)
        newspec /= np.nanstd(newspec, axis=1, keepdims=True)

        spec_signal = newrec['pupil']._modified_copy(newspec)
        spec_signal.name = 'pupil_psd'
        chan_names = []
        for chan in range(0, newspec.shape[0]):
            chan_names.append('puppsd{0}'.format(chan))
        spec_signal.chans = chan_names

        newrec.add_signal(spec_signal)

    if ('pupil_ev' in state_signals) or ('pupil_bs' in state_signals):
        # generate separate pupil baseline and evoked signals

        prestimsilence = newrec["pupil"].extract_epoch('PreStimSilence')
        spont_bins = prestimsilence.shape[2]
        pupil_trial = newrec["pupil"].extract_epoch('TRIAL')

        pupil_bs = np.zeros(pupil_trial.shape)
        for ii in range(pupil_trial.shape[0]):
            pupil_bs[ii, :, :] = np.mean(
                    pupil_trial[ii, :, :spont_bins])
        pupil_ev = pupil_trial - pupil_bs

        newrec['pupil_ev'] = newrec["pupil"].replace_epoch('TRIAL', pupil_ev)
        newrec['pupil_ev'].chans=['pupil_ev']
        newrec['pupil_bs'] = newrec["pupil"].replace_epoch('TRIAL', pupil_bs)
        newrec['pupil_bs'].chans=['pupil_bs']

    if ('each_passive' in state_signals):
        file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
        pset = []
        found_passive1 = False
        for f in file_epochs:
            # test if passive expt
            epoch_indices = ep.epoch_intersection(
                    resp.get_epoch_indices(f),
                    resp.get_epoch_indices('PASSIVE_EXPERIMENT'))
            if epoch_indices.size:
                if not(found_passive1):
                    # skip first passive
                    found_passive1 = True
                else:
                    pset.append(f)
                    newrec[f] = resp.epoch_to_signal(f)
        state_signals.remove('each_passive')
        state_signals.extend(pset)
        if 'each_passive' in permute_signals:
            permute_signals.remove('each_passive')
            permute_signals.extend(pset)

    if ('each_file' in state_signals):
        file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
        trial_indices = resp.get_epoch_indices('TRIAL')
        passive_indices = resp.get_epoch_indices('PASSIVE_EXPERIMENT')
        pset = []
        pcount = 0
        acount = 0
        for f in file_epochs:
            # test if passive expt
            f_indices = resp.get_epoch_indices(f)
            epoch_indices = ep.epoch_intersection(f_indices, passive_indices)

            if epoch_indices.size:
                # this is a passive file
                name1 = "PASSIVE_{}".format(pcount)
                pcount += 1
                if pcount == 1:
                    acount = 1 # reset acount for actives after first passive
                else:
                    # use first passive part A as baseline - don't model
                    pset.append(name1)
                    newrec[name1] = resp.epoch_to_signal(name1, indices=f_indices)

            else:
                name1 = "ACTIVE_{}".format(acount)
                pset.append(name1)
                newrec[name1] = resp.epoch_to_signal(name1, indices=f_indices)

                if pcount == 0:
                    acount -= 1
                else:
                    acount += 1

            # test if passive expt
#            epoch_indices = ep.epoch_intersection(
#                    resp.get_epoch_indices(f),
#                    resp.get_epoch_indices('PASSIVE_EXPERIMENT'))
#            if epoch_indices.size and not(found_passive1):
#                # skip first passive
#                found_passive1 = True
#            else:
#                pset.append(f)
#                newrec[f] = resp.epoch_to_signal(f)
        state_signals.remove('each_file')
        state_signals.extend(pset)
        if 'each_file' in permute_signals:
            permute_signals.remove('each_file')
            permute_signals.extend(pset)

    if ('each_half' in state_signals):
        file_epochs = ep.epoch_names_matching(resp.epochs, "^FILE_")
        trial_indices = resp.get_epoch_indices('TRIAL')
        passive_indices = resp.get_epoch_indices('PASSIVE_EXPERIMENT')
        pset = []
        pcount = 0
        acount = 0
        for f in file_epochs:
            # test if passive expt
            f_indices = resp.get_epoch_indices(f)
            epoch_indices = ep.epoch_intersection(f_indices, passive_indices)
            trial_intersect = ep.epoch_intersection(f_indices, trial_indices)
            #trial_count = trial_intersect.shape[0]
            #_split = int(trial_count/2)
            _t1=trial_intersect[0,0]
            _t2=trial_intersect[-1,1]
            _split = int((_t1+_t2)/2)
            epoch1 = np.array([[_t1,_split]])
            epoch2 = np.array([[_split,_t2]])

            if epoch_indices.size:
                # this is a passive file
                name1 = "PASSIVE_{}_{}".format(pcount, 'A')
                name2 = "PASSIVE_{}_{}".format(pcount, 'B')
                pcount += 1
                if pcount == 1:
                    acount = 1 # reset acount for actives after first passive
                else:
                    # don't model PASSIVE_0 A -- baseline
                    pset.append(name1)
                    newrec[name1] = resp.epoch_to_signal(name1, indices=epoch1)

                # do include part B
                pset.append(name2)
                newrec[name2] = resp.epoch_to_signal(name2, indices=epoch2)
            else:
                name1 = "ACTIVE_{}_{}".format(acount, 'A')
                name2 = "ACTIVE_{}_{}".format(acount, 'B')
                pset.append(name1)
                newrec[name1] = resp.epoch_to_signal(name1, indices=epoch1)
                pset.append(name2)
                newrec[name2] = resp.epoch_to_signal(name2, indices=epoch2)

                if pcount == 0:
                    acount -= 1
                else:
                    acount += 1

        state_signals.remove('each_half')
        state_signals.extend(pset)
        if 'each_half' in permute_signals:
            permute_signals.remove('each_half')
            permute_signals.extend(pset)

    # generate task state signals
    if 'pas' in state_signals:
        fpre = (resp.epochs['name'] == "PRE_PASSIVE")
        fpost = (resp.epochs['name'] == "POST_PASSIVE")
        INCLUDE_PRE_POST = (np.sum(fpre) > 0) & (np.sum(fpost) > 0)
        if INCLUDE_PRE_POST:
            # only include pre-passive if post-passive also exists
            # otherwise the regression gets screwed up
            newrec['pre_passive'] = resp.epoch_to_signal('PRE_PASSIVE')
        else:
            # place-holder, all zeros
            newrec['pre_passive'] = resp.epoch_to_signal('XXX')
            newrec['pre_passive'].chans = ['PRE_PASSIVE']
    if 'puretone_trials' in state_signals:
        newrec['puretone_trials'] = resp.epoch_to_signal('PURETONE_BEHAVIOR')
        newrec['puretone_trials'].chans = ['puretone_trials']
    if 'easy_trials' in state_signals:
        newrec['easy_trials'] = resp.epoch_to_signal('EASY_BEHAVIOR')
        newrec['easy_trials'].chans = ['easy_trials']
    if 'hard_trials' in state_signals:
        newrec['hard_trials'] = resp.epoch_to_signal('HARD_BEHAVIOR')
        newrec['hard_trials'].chans = ['hard_trials']
    if ('active' in state_signals) or ('far' in state_signals):
        newrec['active'] = resp.epoch_to_signal('ACTIVE_EXPERIMENT')
        newrec['active'].chans = ['active']
    if (('hit_trials' in state_signals) or ('miss_trials' in state_signals) or
        ('far' in state_signals) or ('hit' in state_signals)):
        newrec['hit_trials'] = resp.epoch_to_signal('HIT_TRIAL')
        newrec['miss_trials'] = resp.epoch_to_signal('MISS_TRIAL')
        newrec['fa_trials'] = resp.epoch_to_signal('FA_TRIAL')

    sm_len = 180 * newrec['resp'].fs
    if 'far' in state_signals:
        a = newrec['active'].as_continuous()
        fa = newrec['fa_trials'].as_continuous().astype(float)
        #c = np.concatenate((np.zeros((1,sm_len)), np.ones((1,sm_len+1))),
        #                   axis=1)
        c = np.ones((1,sm_len))/sm_len

        fa = convolve2d(fa, c, mode='same')
        fa[a] -= 0.25 # np.nanmean(fa[a])
        fa[np.logical_not(a)] = 0

        s = newrec['fa_trials']._modified_copy(fa)
        s.chans = ['far']
        s.name='far'
        newrec.add_signal(s)

    if 'hit' in state_signals:
        a = newrec['active'].as_continuous()
        hr = newrec['hit_trials'].as_continuous().astype(float)
        ms = newrec['miss_trials'].as_continuous().astype(float)
        ht = hr-ms

        c = np.ones((1,sm_len))/sm_len

        ht = convolve2d(ht, c, mode='same')
        ht[a] -= 0.1  # np.nanmean(ht[a])
        ht[np.logical_not(a)] = 0

        s = newrec['hit_trials']._modified_copy(ht)
        s.chans = ['hit']
        s.name='hit'
        newrec.add_signal(s)

    if 'lick' in state_signals:
        newrec['lick'] = resp.epoch_to_signal('LICK')

    # pupil interactions
    if ('p_x_a' in state_signals):
        # normalize min-max
        p = newrec["pupil"].as_continuous()
        a = newrec["active"].as_continuous()
        newrec["p_x_a"] = newrec["pupil"]._modified_copy(p * a)
        newrec["p_x_a"].chans = ["p_x_a"]

    if ('pupil_x_population' in state_signals):
        # normalize min-max
        p = newrec["pupil"].as_continuous()
        a = newrec["population"].as_continuous()
        newrec["pupil_x_population"] = newrec["population"]._modified_copy(p * a)
        newrec["pupil_x_population"].chans = ["px"+c for c in newrec["pupil_x_population"].chans]

    if ('active_x_population' in state_signals):
        # normalize min-max
        a = newrec["active"].as_continuous()
        p = newrec["population"].as_continuous()
        newrec["active_x_population"] = newrec["population"]._modified_copy(p * a)
        newrec["active_x_population"].chans = ["ax"+c for c in newrec["active_x_population"].chans]

    if ('prw' in state_signals):
        # add channel two of the resp to state and delete it from resp
        if len(rec['resp'].chans) != 2:
            raise ValueError("this is for pairwise fitting")
        else:
            ch2 = rec['resp'].chans[1]
            ch1 = rec['resp'].chans[0]

        newrec['prw'] = newrec['resp'].extract_channels([ch2]).rasterize()
        newrec['resp'] = newrec['resp'].extract_channels([ch1]).rasterize()

    if ('pup_x_prw' in state_signals):
        # interaction term between pupil and the other cell
        if 'prw' not in newrec.signals.keys():
            raise ValueError("Must include prw alone before using interaction")

        else:
            pup = newrec['pupil']._data
            prw = newrec['prw']._data
            sig = newrec['pupil']._modified_copy(pup * prw)
            sig.name = 'pup_x_prw'
            sig.chans = ['pup_x_prw']
            newrec.add_signal(sig)

    for i, x in enumerate(state_signals):
        if x in permute_signals:
            # kludge: fix random seed to index of state signal in list
            # this avoids using the same seed for each shuffled signal
            # but also makes shuffling reproducible
            newrec = concatenate_state_channel(
                    newrec, newrec[x].shuffle_time(rand_seed=i,
                                  mask=newrec['mask']),
                    state_signal_name=new_signalname)
        else:
            newrec = concatenate_state_channel(
                    newrec, newrec[x], state_signal_name=new_signalname)

        newrec = concatenate_state_channel(
                newrec, newrec[x], state_signal_name=new_signalname+"_raw")

    return newrec


def concatenate_state_channel(rec, sig, state_signal_name='state'):

    newrec = rec.copy()

    if state_signal_name not in rec.signals.keys():
        # create an initial state signal of all ones for baseline
        x = np.ones([1, sig.shape[1]])
        ones_sig = sig._modified_copy(x)
        ones_sig.name = "baseline"
        ones_sig.chans = ["baseline"]

        state_sig_list = [ones_sig]
    else:
        # start with existing state signal
        state_sig_list = [rec[state_signal_name]]

    state_sig_list.append(sig)

    state = nems.signal.RasterizedSignal.concatenate_channels(state_sig_list)
    state.name = state_signal_name

    newrec.add_signal(state)

    return newrec


def concatenate_input_channels(rec, input_signals=[], input_name=None):
    newrec = rec.copy()
    input_sig_list = []
    for s in input_signals:
        input_sig_list.append(newrec[s])
    input_sig_list.append(newrec[input_name].rasterize())
    input = nems.signal.RasterizedSignal.concatenate_channels(input_sig_list)
    input.name = input_name

    newrec.add_signal(input)

    return newrec

def signal_select_channels(rec, sig_name="resp", chans=None):

    newrec = rec.copy()
    if chans is None:
        return newrec

    s = newrec[sig_name].rasterize()
    s = s.extract_channels(chans)
    newrec[sig_name] = s

    return newrec


def split_est_val_for_jackknife(rec, epoch_name='TRIAL', modelspecs=None,
                                njacks=10, IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    est = []
    val = []
    # logging.info("Generating {} jackknifes".format(njacks))

    for i in range(njacks):
        # est_out += [est.jackknife_by_time(njacks, i)]
        # val_out += [est.jackknife_by_time(njacks, i, invert=True)]
        est += [rec.jackknife_by_epoch(njacks, i, epoch_name,
                                       tiled=True)]
        val += [rec.jackknife_by_epoch(njacks, i, epoch_name,
                                       tiled=True, invert=True)]

    modelspecs_out = []
    if (not IsReload) and (modelspecs is not None):
        if len(modelspecs) == 1:
            modelspecs_out = [copy.deepcopy(modelspecs[0])
                              for i in range(njacks)]
        elif len(modelspecs) == njacks:
            # assume modelspecs already generated for njacks
            modelspecs_out = modelspecs
        else:
            raise ValueError('modelspecs must be len 1 or njacks')

    return est, val, modelspecs_out


def mask_est_val_for_jackknife(rec, epoch_name='TRIAL', modelspec=None,
                               njacks=10, IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    #est = []
    #val = []
    # logging.info("Generating {} jackknifes".format(njacks))
    if rec.get_epoch_indices(epoch_name).shape[0]:
        pass
    elif rec.get_epoch_indices('REFERENCE').shape[0]:
        log.info('Jackknifing by REFERENCE epochs')
        epoch_name = 'REFERENCE'
    elif rec.get_epoch_indices('TARGET').shape[0]:
        log.info('Jackknifing by TARGET epochs')
        epoch_name = 'TARGET'
    elif rec.get_epoch_indices('TRIAL').shape[0]:
        log.info('Jackknifing by TRIAL epochs')
        epoch_name = 'TRIAL'
    else:
        raise ValueError('No epochs matching '+epoch_name)

    #for i in range(njacks):
        # est_out += [est.jackknife_by_time(njacks, i)]
        # val_out += [est.jackknife_by_time(njacks, i, invert=True)]
        #est += [rec.jackknife_mask_by_epoch(njacks, i, epoch_name,
        #                                    tiled=True)]
        #val += [rec.jackknife_mask_by_epoch(njacks, i, epoch_name,
        #                                    tiled=True, invert=True)]
    est = rec.jackknife_masks_by_epoch(njacks, epoch_name, tiled=True)
    val = rec.jackknife_masks_by_epoch(njacks, epoch_name,
                                       tiled=True, invert=True)

    modelspec_out = []
    if (not IsReload) and (modelspec is not None):
        if modelspec.fit_count == 1:
            modelspec_out = modelspec.tile_fits(njacks)
        elif modelspec.fit_count == njacks:
            # assume modelspec already generated for njacks
            modelspec_out = modelspec
        else:
            raise ValueError('modelspec must be len 1 or njacks')

    return est, val, modelspec_out


def mask_est_val_for_jackknife_by_time(rec, modelspecs=None,
                               njacks=10, IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    #est = []
    #val = []
    #for i in range(njacks):
    #    est += [rec.jackknife_mask_by_time(njacks, i, tiled=True)]
    #    val += [rec.jackknife_mask_by_time(njacks, i, tiled=True, invert=True)]

    est = rec.jackknife_masks_by_time(njacks, tiled=True)
    val = rec.jackknife_masks_by_time(njacks, tiled=True, invert=True)

    modelspec_out = []
    if (not IsReload) and (modelspec is not None):
        if modelspec.fit_count == 1:
            modelspec_out = modelspec.tile_fits(njacks)
        elif modelspec.fit_count == njacks:
            # assume modelspec already generated for njacks
            modelspec_out = modelspec
        else:
            raise ValueError('modelspec must be len 1 or njacks')

    return est, val, modelspec_out
