import warnings
import numpy as np
import nems.epoch as ep
import pandas as pd
from nems.signal import RasterizedSignal
import copy
from scipy.signal import convolve2d

import logging
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
    epochs = recording.epochs
    epoch_names = sorted(set(ep.epoch_names_matching(epochs, epoch_regex)))

    offset = 0
    new_epochs = []
    for epoch_name in epoch_names:
        common_epochs = ep.find_common_epochs(epochs, epoch_name)
        query = 'name == "{}"'.format(epoch_name)
        end = common_epochs.query(query).iloc[0]['end']
        common_epochs[['start', 'end']] += offset
        offset += end
        new_epochs.append(common_epochs)
    new_epochs = pd.concat(new_epochs, ignore_index=True)

    averaged_recording = recording.copy()

    for signal_name, signal in recording.signals.items():
        # TODO: this may be better done as a method in signal subclasses since
        # some subclasses may have more efficient approaches (e.g.,
        # TiledSignal)

        # Extract all occurances of each epoch, returning a dict where keys are
        # stimuli and each value in the dictionary is (reps X cell X bins)
        epoch_data = signal.rasterize().extract_epochs(epoch_names)

        # Average over all occurrences of each epoch
        for epoch_name, epoch in epoch_data.items():
            epoch_data[epoch_name] = np.nanmean(epoch, axis=0)
        data = [epoch_data[epoch_name] for epoch_name in epoch_names]
        data = np.concatenate(data, axis=-1)
        if data.shape[-1] != round(signal.fs * offset):
            raise ValueError('Misalignment issue in averaging signal')

        averaged_signal = signal._modified_copy(data, epochs=new_epochs)
        averaged_recording.add_signal(averaged_signal)

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


def mask_all_but_correct_references(rec):
    """
    Specialized function for removing incorrect trials from data
    collected using baphy during behavior.

    TODO: Migrate to nems_lbhb and/or make a more generic version
    """

    newrec = rec.copy()
    newrec['resp'] = newrec['resp'].rasterize()
    if 'stim' in newrec.signals.keys():
        newrec['stim'] = newrec['stim'].rasterize()

    newrec = newrec.or_mask(['PASSIVE_EXPERIMENT', 'HIT_TRIAL'])
    newrec = newrec.and_mask(['REFERENCE'])

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

    epoch_indices2=epoch_indices[0:1,:]
    for i in range(1,epoch_indices.shape[0]):
        if epoch_indices[i,0]==epoch_indices2[-1,1]:
            epoch_indices2[-1,1]=epoch_indices[i,0]
        else:
            epoch_indices2=np.concatenate((epoch_indices2,epoch_indices[i:(i+1),:]),
                                          axis=0)

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


def generate_psth_from_resp(rec, epoch_regex='^STIM_', smooth_resp=False):
    '''
    Estimates a PSTH from all responses to each regex match in a recording

    subtract spont rate based on pre-stim silence for ALL estimation data.

    if rec['mask'] exists, uses rec['mask'] == True to determine valid epochs
    '''

    resp = rec['resp'].rasterize()
    nCells = len(resp.chans)
    # compute spont rate during valid (non-masked) trials
    prestimsilence = resp.extract_epoch('PreStimSilence')
    if 'mask' in rec.signals.keys():
        prestimmask = np.tile(rec['mask'].extract_epoch('PreStimSilence'), [1, nCells, 1])
        prestimsilence[prestimmask == False] = np.nan

    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    idx = resp.get_epoch_indices('PreStimSilence')
    prebins = idx[0][1] - idx[0][0]
    idx = resp.get_epoch_indices('PostStimSilence')
    postbins = idx[0][1] - idx[0][0]

    # compute PSTH response during valid trials
    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    folded_matrices = resp.extract_epochs(epochs_to_extract)
    if 'mask' in rec.signals.keys():
        log.info('masking out invalid time bins before PSTH calc')
        folded_mask = rec['mask'].extract_epochs(epochs_to_extract)
        for k, m in folded_mask.items():
            m = np.tile(m, [1, nCells, 1])
            folded_matrices[k][m == False] = np.nan

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
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

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg = resp.replace_epochs(per_stim_psth)
    respavg.name = 'psth'

    # add signal to the recording
    rec.add_signal(respavg)

    if smooth_resp:
        log.info('Replacing resp with smoothed resp')
        resp = resp.replace_epochs(folded_matrices)
        rec.add_signal(resp)

    return rec


def generate_psth_from_est_for_both_est_and_val(est, val,
                                                epoch_regex='^STIM_'):
    '''
    Estimates a PSTH from the EST set, and returns two signals based on the
    est and val, in which each repetition of a stim uses the EST PSTH?

    subtract spont rate based on pre-stim silence for ALL estimation data.
    '''

    resp_est = est['resp']
    resp_val = val['resp']

    # compute PSTH response and spont rate during those valid trials
    prestimsilence = resp_est.extract_epoch('PreStimSilence')
    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
    folded_matrices = resp_est.extract_epochs(epochs_to_extract)

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


def generate_psth_from_est_for_both_est_and_val_nfold(ests, vals, epoch_regex = '^STIM_'):
    '''
    call generate_psth_from_est_for_both_est_and_val for each e,v
    pair in ests,vals
    '''
    for e ,v in zip(ests, vals):
        e, v = generate_psth_from_est_for_both_est_and_val(e, v, epoch_regex)

    return ests, vals


def make_state_signal(rec, state_signals=['pupil'], permute_signals=[],
                      new_signalname='state'):
    """
    generate state signal for stategainX models

    TODO: Migrate to nems_lbhb or make a more generic version
    """

    newrec = rec.copy()
    resp = newrec['resp'].rasterize()

    # Much faster; TODO: Test if throws warnings
    x = np.ones([1, resp.shape[1]])
    ones_sig = resp._modified_copy(x)
    ones_sig.name = "baseline"
    ones_sig.chans = ["baseline"]

    # DEPRECATED, NOW THAT NORMALIZATION IS IMPLEMENTED
    if ('pupil' in state_signals) or ('pupil_ev' in state_signals) or \
       ('pupil_bs' in state_signals):
        # normalize min-max
        p = newrec["pupil"].as_continuous().copy()
        # p[p < np.nanmax(p)/5] = np.nanmax(p)/5
        p -= np.nanmean(p)
        p /= np.nanstd(p)
        newrec["pupil"] = newrec["pupil"]._modified_copy(p)

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

        newrec['pupil_ev'] = newrec["pupil"].replace_epoch(
                'TRIAL', pupil_ev)
        newrec['pupil_bs'] = newrec["pupil"].replace_epoch(
                'TRIAL', pupil_bs)

    # generate task state signals
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

    newrec['hit_trials'] = resp.epoch_to_signal('HIT_TRIAL')
    newrec['miss_trials'] = resp.epoch_to_signal('MISS_TRIAL')
    newrec['fa_trials'] = resp.epoch_to_signal('FA_TRIAL')
    newrec['puretone_trials'] = resp.epoch_to_signal('PURETONE_BEHAVIOR')
    newrec['puretone_trials'].chans = ['puretone_trials']
    newrec['easy_trials'] = resp.epoch_to_signal('EASY_BEHAVIOR')
    newrec['easy_trials'].chans = ['easy_trials']
    newrec['hard_trials'] = resp.epoch_to_signal('HARD_BEHAVIOR')
    newrec['hard_trials'].chans = ['hard_trials']
    newrec['active'] = resp.epoch_to_signal('ACTIVE_EXPERIMENT')
    newrec['active'].chans = ['active']
    if 'lick' in state_signals:
        newrec['lick'] = resp.epoch_to_signal('LICK')

    # pupil interactions
    if ('pupil' in state_signals):
        # normalize min-max
        p = newrec["pupil"].as_continuous()
        a = newrec["active"].as_continuous()
        newrec["p_x_a"] = newrec["pupil"]._modified_copy(p * a)
        newrec["p_x_a"].chans = ["p_x_a"]

    state_sig_list = [ones_sig]
    # rint(state_sig_list[-1].shape)

    for x in state_signals:
        if x in permute_signals:
            # TODO support for signals_permute
            # raise ValueError("permute_signals not yet supported")
            state_sig_list += [newrec[x].shuffle_time()]
        else:
            state_sig_list += [newrec[x]]
        # print(x)
        # print(state_sig_list[-1])
        # print(state_sig_list[-1].shape)

    state = RasterizedSignal.concatenate_channels(state_sig_list)
    state.name = new_signalname

    # scale all signals to range from 0 - 1
    # state = state.normalize(normalization='minmax')

    newrec.add_signal(state)

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


def mask_est_val_for_jackknife(rec, epoch_name='TRIAL', modelspecs=None,
                               njacks=10, IsReload=False, **context):
    """
    take a single recording (est) and define njacks est/val sets using a
    jackknife logic. returns lists est_out and val_out of corresponding
    jackknife subsamples. removed timepoints are replaced with nan
    """
    est = []
    val = []
    # logging.info("Generating {} jackknifes".format(njacks))
    if rec.get_epoch_indices(epoch_name).shape[0]:
        pass
    elif rec.get_epoch_indices('REFERENCE').shape[0]:
        log.info('jackknifing by REFERENCE epochs')
        epoch_name = 'REFERENCE'
    elif rec.get_epoch_indices('TARGET').shape[0]:
        log.info('jackknifing by TARGET epochs')
        epoch_name = 'TARGET'
    else:
        raise ValueError('No epochs matching '+epoch_name)

    for i in range(njacks):
        # est_out += [est.jackknife_by_time(njacks, i)]
        # val_out += [est.jackknife_by_time(njacks, i, invert=True)]
        est += [rec.jackknife_mask_by_epoch(njacks, i, epoch_name,
                                            tiled=True)]
        val += [rec.jackknife_mask_by_epoch(njacks, i, epoch_name,
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


def make_contrast_signal(rec, name='contrast', source_name='stim', ms=500,
                         bins=None):
    '''
    Creates a new signal whose values represent the degree of variability
    in each channel of the source signal. Each value is based on the
    previous values within a range specified by either <ms> or <bins>.
    Only supports RasterizedSignal.
    '''

    rec = rec.copy()

    source_signal = rec[source_name]

    if not isinstance(source_signal, RasterizedSignal):
        try:
            source_signal = source_signal.rasterize()
        except AttributeError:
            raise TypeError("signal with key {} was not a RasterizedSignal"
                            " and could not be converted to one."
                            .format(source_name))

    array = source_signal.as_continuous().copy()

    if ms:
        history = int((ms/1000)*source_signal.fs)
    elif bins:
        history = int(bins)
    else:
        raise ValueError("Either ms or bins parameter must be specified "
                         "and nonzero.")
    # TODO: Alternatively, base history length on some feature of signal?
    #       Like average length of some epoch ex 'TRIAL'

    array[np.isnan(array)] = 0
    filt = np.concatenate((np.zeros([1, history+1]),
                           np.ones([1, history])), axis=1)
    contrast = convolve2d(array, filt, mode='same')

    contrast_sig = source_signal._modified_copy(contrast)
    rec[name] = contrast_sig

    return rec
