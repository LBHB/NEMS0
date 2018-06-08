import matplotlib.pyplot as plt
import numpy as np
import scipy

from .timeseries import timeseries_from_signals, timeseries_from_vectors

def state_vars_timeseries(rec, modelspec, ax=None):

    if ax is not None:
        plt.sca(ax)

    pred = rec['pred']
    resp = rec['resp']

    r1 = resp.as_continuous().copy().T
    p1 = pred.as_continuous().copy().T
    nnidx = np.isfinite(p1)
    r1=r1[nnidx]
    p1=p1[nnidx]
    #r1 = scipy.signal.decimate(r1[nnidx], q=5, axis=0)
    #p1 = scipy.signal.decimate(p1[nnidx], q=5, axis=0)
    #t = np.arange(len(r1))/pred.fs*5
    t = np.arange(len(r1))/pred.fs

    plt.plot(t, r1, linewidth=1)
    plt.plot(t, p1, linewidth=1)
    mmax = np.nanmax(p1) * 0.8

    if 'state' in rec.signals.keys():
        for m in modelspec:
            if 'state_dc_gain' in m['fn']:
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
                if len(g) < 10:
                    s = ",".join(rec["state"].chans)
                    g_string = np.array2string(g, precision=3)
                    d_string = np.array2string(d, precision=3)
                    s += " g={} d={} ".format(g_string, d_string)
                else:
                    s = None
            else:
                s = None

        num_vars = rec['state'].shape[0]
        ts = rec['state'].as_continuous().copy()

        for i in range(1, num_vars):
            d = ts[[i], :].T
            d = d[nnidx]
            #d = scipy.signal.decimate(d[nnidx], q=5, axis=0)
            d = d / np.nanmax(d) * mmax - (0.1 + i) * mmax
            plt.plot(t, d, linewidth=1)
        ax = plt.gca()
        # plt.text(0.5, 0.9, s, transform=ax.transAxes,
        #         horizontalalignment='center')
        if s:
            plt.title(s)
    plt.xlabel('time (s)')
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


def state_var_psth_from_epoch(rec, epoch, psth_name='resp', psth_name2='pred',
                              state_sig='pupil', ax=None):
    """
    Plot PSTH averaged across all occurences of epoch, grouped by
    above- and below-average values of a state signal (state_sig)
    """

    # TODO: Does using epochs make sense for these?
    if ax is not None:
        plt.sca(ax)

    fs = rec[psth_name].fs

    d = rec[psth_name].get_epoch_bounds('PreStimSilence')
    PreStimSilence = np.mean(np.diff(d)) - 0.5/fs
    d = rec[psth_name].get_epoch_bounds('PostStimSilence')
    PostStimSilence = np.mean(np.diff(d)) - 0.5/fs

    full_psth = rec[psth_name]
    folded_psth = full_psth.extract_epoch(epoch)
    if psth_name2 is not None:
        full_psth2 = rec[psth_name2]
        folded_psth2 = full_psth2.extract_epoch(epoch)

    full_var = rec['state'].loc[state_sig]
    folded_var = full_var.extract_epoch(epoch)

    # remove masked out occurences if mask signal exists
    if 'mask' in rec.signals.keys():
        folded_mask = rec['mask'].extract_epoch(epoch)
        keep_occurences = folded_mask[:, 0, 0]
        folded_psth = folded_psth[keep_occurences, :, :]
        folded_psth2 = folded_psth2[keep_occurences, :, :]
        folded_var = folded_var[keep_occurences, :, :]

        # print(state_sig)
        # print(folded_var.shape)
        # print(folded_mask.shape)
        # print(np.sum(np.isfinite(folded_mask)))

    # compute the mean state for each occurrence
    m = np.nanmean(folded_var[:, 0, :], axis=1)

    # compute the mean state across all occurrences
    mean = np.nanmean(m)

    # low = response on epochs when state less than mean
    if np.sum(m < mean):
        low = np.nanmean(folded_psth[m < mean, :, :], axis=0).T
        low2 = np.nanmean(folded_psth2[m < mean, :, :], axis=0).T
    else:
        low = np.ones(folded_psth[0, :, :].shape).T * np.nan
        low2 = np.ones(folded_psth2[0, :, :].shape).T * np.nan

    # high = response on epochs when state less than mean
    title = state_sig
    high = np.nanmean(folded_psth[m >= mean, :, :], axis=0).T
    high2 = np.nanmean(folded_psth2[m >= mean, :, :], axis=0).T

    hv = np.nanmean(m[m >= mean])
    if np.sum(m < mean) > 0:
        lv = np.nanmean(m[m < mean])
        if (hv > 0.95) and (hv < 1.05) and (lv > -0.05) and (lv < 0.05):
            legend = ('Lo', 'Hi')
        else:
            legend = ('< Mean', '>= Mean')

        timeseries_from_vectors([low, high], fs=fs, title=title, ax=ax,
                                legend=legend, time_offset=PreStimSilence)
        timeseries_from_vectors([low2, high2], fs=fs, title=title, ax=ax,
                                linestyle='--', time_offset=PreStimSilence)
    else:
        timeseries_from_vectors([low, high], fs=fs, title=title, ax=ax,
                                time_offset=PreStimSilence)
        timeseries_from_vectors([low2, high2], fs=fs, title=title, ax=ax,
                                linestyle='--', time_offset=PreStimSilence)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([0, 0])+(xlim[1]-PostStimSilence), ylim, 'k--')

    if state_sig == 'baseline':
        ax.set_xlabel(epoch)


def state_gain_plot(modelspec, ax=None, clim=None, title=None):
    for m in modelspec:
        if 'state_dc_gain' in m['fn']:
            g = m['phi']['g']
            d = m['phi']['d']

    if ax is not None:
        plt.sca(ax)

    plt.plot(d)
    plt.plot(g)
    plt.xlabel('state channel')
    plt.legend(('baseline', 'gain'))
    if title:
        plt.title(title)


