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
        #plt.text(0.5, 0.9, s, transform=ax.transAxes,
        #         horizontalalignment='center')
        plt.title(s)
    plt.axis('tight')


def state_var_psth(rec, psth_name='resp', var_name='pupil', ax=None):
    if ax is not None:
        plt.sca(ax)

    psth = rec[psth_name]._data
    fs = rec[psth_name].fs
    var = rec['state'].loc[var_name]._data
    mean = np.nanmean(var)
    low = psth[var < mean]
    high = psth[var >= mean]
    timeseries_from_vectors([low, high], fs=fs, title=var_name, ax=ax)


def state_var_psth_from_epoch(rec, epoch, psth_name='resp', state_sig='pupil',
                              ax=None):
    """
    Plot PSTH averaged across all occurences of epoch, grouped by
    above- and below-average values of a state signal (state_sig)
    """

    # TODO: Does using epochs make sense for these?
    if ax is not None:
        plt.sca(ax)

    fs = rec[psth_name].fs

    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)

    full_var = rec['state'].loc[state_sig]
    folded_var = np.squeeze(full_var.extract_epoch(epoch))

    # compute the mean state for each occurrence
    m = np.nanmean(folded_var, axis=1)

    # compute the mean state across all occurrences
    mean = np.nanmean(m)

    # low = response on epochs when state less than mean
    if np.sum(m < mean):
        low = np.nanmean(folded_psth[m < mean, :, :], axis=0).T
    else:
        low = np.ones(folded_psth[0, :, :].shape).T * np.nan

    # high = response on epochs when state less than mean
    high = np.nanmean(folded_psth[m >= mean, :, :], axis=0).T

    legend = ('< Mean', '>= Mean')
    title = '{} conditioned {}'.format(state_sig, epoch)

    timeseries_from_vectors([low, high], fs=fs, title=title, ax=ax,
                            legend=legend)
