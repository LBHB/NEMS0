import warnings
import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal
import copy

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


def average_away_epoch_occurrences(rec, epoch_regex='^STIM_'):
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

    # Create new recording
    newrec = rec.copy()

    counter = 0

    # iterate through each signal
    for signal_name, signal_to_average in rec.signals.items():
        # TODO: for TiledSignals, there is a much simpler way to do this!

        # 0. rasterize
        signal_to_average = signal_to_average.rasterize()

        # 1. Find matching epochs
        epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)

        # 2. Fold over all stimuli, returning a dict where keys are stimuli
        #    and each value in the dictionary is (reps X cell X bins)
        folded_matrices = signal_to_average.extract_epochs(epochs_to_extract)

        # force a standard list of sorted keys for all signals
        if counter == 0:
            sorted_keys = list(folded_matrices.keys())
            sorted_keys.sort()
        counter += 1

        # 3. Average over all occurrences of each epoch, and append to data
        fs = signal_to_average.fs
        data = np.zeros([signal_to_average.nchans, 0])
        current_time = 0
        epochs = None
        for k in sorted_keys:
            # Supress warnings about all-nan matrices
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_stim_psth = np.nanmean(folded_matrices[k], axis=0)
            data = np.concatenate((data, per_stim_psth), axis=1)
            epoch = current_time+np.array([[0, per_stim_psth.shape[1]/fs]])
            df = pd.DataFrame(np.tile(epoch, [2, 1]), columns=['start', 'end'])
            df['name'] = k
            df.at[1, 'name'] = 'TRIAL'
            if epochs is not None:
                epochs = epochs.append(df, ignore_index=True)
            else:
                epochs = df
            current_time = epoch[0, 1]
            # print("{0} epoch: {1}-{2}".format(k,epoch[0,0],epoch[0,1]))

        avg_signal = signal.RasterizedSignal(
                fs=fs, data=data,
                name=signal_to_average.name,
                recording=signal_to_average.recording,
                chans=signal_to_average.chans,
                epochs=epochs,
                meta=signal_to_average.meta)
        newrec.add_signal(avg_signal)

    return newrec


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

    # get list of start and stop times (epoch bounds)
#    epoch_indices = np.vstack((
#            ep.epoch_intersection(sig.get_epoch_bounds('HIT_TRIAL'),
#                                  sig.get_epoch_bounds('REFERENCE')),
#            ep.epoch_intersection(sig.get_epoch_bounds('REFERENCE'),
#                                  sig.get_epoch_bounds('PASSIVE_EXPERIMENT'))))
    epoch_indices = np.vstack((
            ep.epoch_intersection(sig.get_epoch_indices('HIT_TRIAL'),
                                  sig.get_epoch_indices('REFERENCE')),
            ep.epoch_intersection(sig.get_epoch_indices('REFERENCE'),
                                  sig.get_epoch_indices('PASSIVE_EXPERIMENT'))))

    # Only takes the first of any conflicts (don't think I actually need this)
    epoch_indices = ep.remove_overlap(epoch_indices)

    # merge any epochs that are directly adjacent
    epoch_indices2 = epoch_indices[0:1, :]
    for i in range(1, epoch_indices.shape[0]):
        if epoch_indices[i, 0] == epoch_indices2[-1, 1]:
            epoch_indices2[-1, 1] = epoch_indices[i, 0]
        else:
            epoch_indices2 = np.concatenate(
                    (epoch_indices2, epoch_indices[i:(i + 1), :]), axis=0)

    epoch_times = epoch_indices2 / sig.fs

    # add adjusted signals to the recording
    newrec = rec.select_times(epoch_times)

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


def generate_psth_from_resp(rec, epoch_regex='^STIM_', smooth_resp=False):
    '''
    Estimates a PSTH from all responses to each regex match in a recording

    subtract spont rate based on pre-stim silence for ALL estimation data.
    '''

    resp = rec['resp'].rasterize()

    # compute PSTH response and spont rate during those valid trials
    prestimsilence = resp.extract_epoch('PreStimSilence')
    if len(prestimsilence.shape) == 3:
        spont_rate = np.nanmean(prestimsilence, axis=(0, 2))
    else:
        spont_rate = np.nanmean(prestimsilence)

    idx = resp.get_epoch_indices('PreStimSilence')
    prebins = idx[0][1] - idx[0][0]
    idx = resp.get_epoch_indices('PostStimSilence')
    postbins = idx[0][1] - idx[0][0]

    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    folded_matrices = resp.extract_epochs(epochs_to_extract)

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
        p = newrec["pupil"].as_continuous()
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

    # generate stask tate signals
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

    state = signal.RasterizedSignal.concatenate_channels(state_sig_list)
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
