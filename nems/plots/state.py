import matplotlib.pyplot as plt
import numpy as np
import scipy

from .timeseries import timeseries_from_signals, timeseries_from_vectors

def state_vars_timeseries(rec, modelspec, ax=None):
    if ax is not None:
        plt.sca(ax)

    pred = rec['pred']
    resp = rec['resp']

    r1 = resp.as_continuous().T
    p1 = pred.as_continuous().T
    nnidx = np.isfinite(p1)

    r1 = scipy.signal.decimate(r1[nnidx], q=5, axis=0)
    p1 = scipy.signal.decimate(p1[nnidx], q=5, axis=0)

    plt.plot(r1)
    plt.plot(p1)
    mmax = np.nanmax(p1)

    if 'state' in rec.signals.keys():
        for m in modelspec:
            if 'state_dc_gain' in m['fn']:
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
            s = ",".join(rec["state"].chans)
            g_string = np.array2string(g, precision=3)
            d_string = np.array2string(d, precision=3)
            s += " g={} d={} ".format(g_string, d_string)

        num_vars = rec['state'].shape[0]
        for i in range(1, num_vars):
            d = rec['state'].as_continuous()[[i], :].T
            d = scipy.signal.decimate(d[nnidx], q=5, axis=0)
            d = d/np.nanmax(d)*mmax - mmax*1.1
            plt.plot(d)
        ax = plt.gca()
        plt.text(0.5, 0.9, s, transform=ax.transAxes,
                 horizontalalignment='center')
    plt.axis('tight')


def state_var_psth(rec, psth_name='stim', var_name='pupil', ax=None):
    if ax is not None:
        plt.sca(ax)

    psth = rec[psth_name]._data
    fs = rec[psth_name].fs
    var = rec['state'].loc[var_name]._data
    mean = np.nanmean(var)
    low = psth[var < mean]
    high = psth[var >= mean]

    timeseries_from_vectors([low, high], fs=fs, title=var_name, ax=ax)


def state_var_psth_from_epochs(rec, epoch, psth_name='stim', var_name='pupil',
                               occurrence=0, ax=None):
    # TODO: Does using epochs make sense for these?
    if ax is not None:
        plt.sca(ax)

    fs = rec[psth_name].fs
    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)
    psth = folded_psth[occurrence]

    full_var = rec[var_name]
    folded_var = full_var.extract_epoch(epoch)
    var = folded_var[occurrence]

    mean = np.nanmean(var)
    low = psth[var < mean]
    high = psth[var >= mean]

    timeseries_from_vectors([low, high], fs=fs, title=var_name, ax=ax)


"""
    #timeseries_from_signals(
    epoch_regex='^STIM_'
    resp_est=est['resp']
    resp_val=val['resp']

    epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
    folded_matrices = resp_est.extract_epochs(epochs_to_extract)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg_est = resp_est.replace_epochs(per_stim_psth)
    respavg_est.name = 'stim'  # TODO: SVD suggests rename 2018-03-08
    ref_phase=est['resp'].epoch_to_signal('REFERENCE')
    respavg_est=respavg_est.nan_mask(ref_phase.as_continuous())
    #hit_phase=est['resp'].epoch_to_signal('HIT_TRIAL')
    #respavg_est=respavg_est.nan_mask(hit_phase.as_continuous())
    est.add_signal(respavg_est)

    respavg_val = resp_val.replace_epochs(per_stim_psth)
    respavg_val.name = 'stim' # TODO: SVD suggests rename 2018-03-08
    ref_phase=val['resp'].epoch_to_signal('REFERENCE')
    respavg_val=respavg_val.nan_mask(ref_phase.as_continuous())
    #hit_phase=est['resp'].epoch_to_signal('HIT_TRIAL')
    #respavg_val=respavg_val.nan_mask(hit_phase.as_continuous())
    val.add_signal(respavg_val)
"""
