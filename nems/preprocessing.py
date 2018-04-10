import warnings
import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal
import copy

import logging
logging.basicConfig(level=logging.INFO)

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
    epochs_to_extract = ep.epoch_names_matching(signal_to_average.epochs, epoch_regex)
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

    counter=0

    # iterate through each signal
    for signal_name, signal_to_average in rec.signals.items():
        # TODO: for TiledSignals, there is a much simpler way to do this!

        # 0. rasterize
        signal_to_average=signal_to_average.rasterize()


        # 1. Find matching epochs
        epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)

        # 2. Fold over all stimuli, returning a dict where keys are stimuli
        #    and each value in the dictionary is (reps X cell X bins)
        folded_matrices = signal_to_average.extract_epochs(epochs_to_extract)

        # force a standard list of sorted keys for all signals
        if counter==0:
            sorted_keys=list(folded_matrices.keys())
            sorted_keys.sort()
        counter+=1

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
            #print("{0} epoch: {1}-{2}".format(k,epoch[0,0],epoch[0,1]))

        avg_signal = signal.RasterizedSignal(fs=fs, data=data,
                                   name=signal_to_average.name,
                                   recording=signal_to_average.recording,
                                   chans=signal_to_average.chans,
                                   epochs=epochs,
                                   meta=signal_to_average.meta)
        newrec.add_signal(avg_signal)

    return newrec

def generate_psth_from_est_for_both_est_and_val(est,val):
    '''
    Estimates a PSTH from the EST set, and returns two signals based on the
    est and val, in which each repetition of a stim uses the EST PSTH?

    subtract spont rate based on pre-stim silence for ALL estimation data.
    '''

    epoch_regex='^STIM_'
    resp_est=est['resp'].copy()
    resp_val=val['resp']

    # find all valid references in est data-- passive or correct trials
    ref_phase=resp_est.epoch_to_signal('REFERENCE')
    active_phase=resp_est.epoch_to_signal('ACTIVE_EXPERIMENT')
    correct_phase=resp_est.epoch_to_signal('HIT_TRIAL')
    valid_phase=np.logical_and(ref_phase.as_continuous(),
                               np.logical_or(np.logical_not(active_phase.as_continuous()),
                                             correct_phase.as_continuous()))
    ref_phase=ref_phase._modified_copy(valid_phase)
    resp_est=resp_est.nan_mask(ref_phase.as_continuous())

    # compute PSTH response and spont rate during those valid trials
    prestimsilence = resp_est.extract_epoch('PreStimSilence')
    if len(prestimsilence.shape)==3:
        spont_rate = np.nanmean(prestimsilence,axis=(0,2))
    else:
        spont_rate=np.nanmean(prestimsilence)

    epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
    folded_matrices = resp_est.extract_epochs(epochs_to_extract)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)-spont_rate

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg_est = resp_est.replace_epochs(per_stim_psth)
    respavg_est.name = 'stim'  # TODO: SVD suggests rename 2018-03-08

    # mark invalid phases as nan
    respavg_est=respavg_est.nan_mask(ref_phase.as_continuous())

    # add signal to the recording
    est.add_signal(respavg_est)

    respavg_val = resp_val.replace_epochs(per_stim_psth)
    respavg_val.name = 'stim' # TODO: SVD suggests rename 2018-03-08
    ref_phase=val['resp'].epoch_to_signal('REFERENCE')
    active_phase=val['resp'].epoch_to_signal('ACTIVE_EXPERIMENT')
    correct_phase=val['resp'].epoch_to_signal('HIT_TRIAL')
    valid_phase=np.logical_and(ref_phase.as_continuous(),
                               np.logical_or(np.logical_not(active_phase.as_continuous()),
                                             correct_phase.as_continuous()))
    ref_phase=ref_phase._modified_copy(valid_phase)

    respavg_val=respavg_val.nan_mask(ref_phase.as_continuous())

    # add signal to the recording
    val.add_signal(respavg_val)

    return est, val

def generate_psth_from_est_for_both_est_and_val_nfold(ests,vals):
    '''
    call generate_psth_from_est_for_both_est_and_val for each e,v
    pair in ests,vals
    '''
    for e,v in zip(ests,vals):
        e,v=generate_psth_from_est_for_both_est_and_val(e,v)

    return ests,vals

def make_state_signal(rec, state_signals=['pupil'], permute_signals=[], new_signalname='state'):
    """
    generate state signal for stategainX models
    TODO: SVD document this and/or move it out of generic nems code
    """
    x = np.ones([1,rec[state_signals[0]]._data.shape[1]])  # Much faster; TODO: Test if throws warnings
    ones_sig = rec[state_signals[0]]._modified_copy(x)
    ones_sig.name="baseline"
    ones_sig.chans=["baseline"]

    newrec = rec.copy()
    resp=newrec['resp']

    if 'pupil' in state_signals:
        # normalize by 100
        newrec["pupil"]=newrec["pupil"]._modified_copy(newrec["pupil"].as_continuous()/100)

    # generate stask tate signals
    fpre=(resp.epochs['name']=="PRE_PASSIVE")
    fpost=(resp.epochs['name']=="POST_PASSIVE")
    INCLUDE_PRE_POST=(np.sum(fpre)>0) & (np.sum(fpost)>0)
    if INCLUDE_PRE_POST:
        # only include pre-passive if post-passive also exists
        # otherwise the regression gets screwed up
        newrec['pre_passive']=resp.epoch_to_signal('PRE_PASSIVE')
    else:
        # place-holder, all zeros
        newrec['pre_passive']=resp.epoch_to_signal('XXX')
        newrec['pre_passive'].chans=['PRE_PASSIVE']

    newrec['hit_trials']=resp.epoch_to_signal('HIT_TRIAL')
    newrec['miss_trials']=resp.epoch_to_signal('MISS_TRIAL')
    newrec['fa_trials']=resp.epoch_to_signal('FA_TRIAL')
    newrec['puretone_trials']=resp.epoch_to_signal('PURETONE_BEHAVIOR')
    newrec['easy_trials']=resp.epoch_to_signal('EASY_BEHAVIOR')
    newrec['hard_trials']=resp.epoch_to_signal('HARD_BEHAVIOR')
    newrec['active']=resp.epoch_to_signal('ACTIVE_EXPERIMENT')
    newrec['active'].chans=['active']
    state_sig_list=[ones_sig]
    #print(state_sig_list[-1].shape)
    for x in state_signals:
        if x in permute_signals:
            # TODO support for signals_permute
            #raise ValueError("permute_signals not yet supported")
            state_sig_list+=[newrec[x].shuffle_time()]
            #print(state_sig_list[-1].shape)
        else:
            state_sig_list+=[newrec[x]]

    state=signal.RasterizedSignal.concatenate_channels(state_sig_list)
    state.name=new_signalname
    newrec.add_signal(state)

    return newrec

def split_est_val_for_jackknife(est, modelspecs=None, njacks=10, IsReload=False, **context):

    est_out=[]
    val_out=[]
    logging.info("Generating  {} jackknifes".format(njacks))
    for i in range(njacks):
        #est_out += [est.jackknife_by_time(njacks, i)]
        #val_out += [est.jackknife_by_time(njacks, i, invert=True)]
        est_out += [est.jackknife_by_epoch(njacks, i,
                        epoch_name='TRIAL',tiled=True)]
        val_out += [est.jackknife_by_epoch(njacks, i,
                        epoch_name='TRIAL',tiled=True,invert=True)]
    modelspecs_out=[]
    if (not IsReload) and (modelspecs is not None):
        if len(modelspecs)==1:
            modelspecs_out=[copy.deepcopy(modelspecs[0]) for i in range(njacks)]
        elif len(modelspecs)==njacks:
            # assume modelspecs already generated for njacks
            modelspecs_out=modelspecs
        else:
            raise ValueError('modelspecs must be len 1 or njacks')
    return est_out, val_out, modelspecs_out

