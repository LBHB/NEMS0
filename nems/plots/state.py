import matplotlib.pyplot as plt
import numpy as np
import scipy

from .timeseries import (timeseries_from_signals, timeseries_from_vectors,
                         ax_remove_box)

from nems.utils import get_channel_number
from nems.metrics.state import state_mod_split
from nems.plots.utils import ax_remove_box



def state_vars_timeseries(rec, modelspec, ax=None, state_colors=None,
                          decimate_by=1, channel=None):

    if ax is not None:
        plt.sca(ax)
    pred = rec['pred']
    resp = rec['resp']
    fs = resp.fs

    chanidx = get_channel_number(resp, channel)

    r1 = resp.as_continuous()[chanidx, :].T * fs
    p1 = pred.as_continuous()[chanidx, :].T * fs
    nnidx = np.isfinite(p1)
    r1 = r1[nnidx]
    p1 = p1[nnidx]

    if decimate_by > 1:
        r1 = scipy.signal.decimate(r1, q=decimate_by, axis=0)
        p1 = scipy.signal.decimate(p1, q=decimate_by, axis=0)
        fs /= decimate_by

    t = np.arange(len(r1)) / fs

    plt.plot(t, r1, linewidth=1, color='gray')
    plt.plot(t, p1, linewidth=1, color='black')
    print(p1.shape)
    mmax = np.nanmax(p1) * 0.8

    if 'state' in rec.signals.keys():
        s = None
        g = None
        d = None
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

        num_vars = rec['state'].shape[0]
        ts = rec['state'].as_continuous().copy()
        if state_colors is None:
            state_colors = [None] * num_vars
        print(nnidx.shape)
        print(ts.shape)
        for i in range(1, num_vars):
            st = ts[i, :].T
            if decimate_by>1:
                st = scipy.signal.decimate(st[nnidx], q=decimate_by, axis=0)
            else:
                st = st[nnidx]

            st = st / np.nanmax(st) * mmax - 1.25 * i * mmax
            plt.plot(t, st, linewidth=1, color=state_colors[i-1])

            if g is not None:
                if g.ndim == 1:
                    tstr = "{} (d={:.3f},g={:.3f})".format(
                            rec['state'].chans[i], d[i], g[i])
                else:
                    tstr = "{} (d={:.3f},g={:.3f})".format(
                            rec['state'].chans[i], d[0, i], g[0, i])
            else:
                tstr = "{}".format(rec['state'].chans[i])

            plt.text(t[0], -i*mmax*1.25, tstr, fontsize=6)
        ax = plt.gca()
        # plt.text(0.5, 0.9, s, transform=ax.transAxes,
        #         horizontalalignment='center')
        # if s:
        #    plt.title(s, fontsize=8)
    plt.xlabel('time (s)')
    plt.axis('tight')

    ax_remove_box(ax)


def state_var_psth(rec, psth_name='resp', var_name='pupil', ax=None,
                   channel=None):
    if ax is not None:
        plt.sca(ax)

    chanidx = get_channel_number(rec[psth_name], channel)

    psth = rec[psth_name]._data[:, chanidx, :]
    fs = rec[psth_name].fs
    var = rec['state'].loc[var_name]._data
    mean = np.nanmean(var)
    low = psth[var < mean]
    high = psth[var >= mean]
    timeseries_from_vectors([low, high], fs=fs, title=var_name, ax=ax)


def state_var_psth_from_epoch(rec, epoch, psth_name='resp', psth_name2='pred',
                              state_sig='state_raw', state_chan='pupil', ax=None,
                              colors=None, channel=None, decimate_by=1):
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
    if d.size > 0:
        PostStimSilence = np.min(np.diff(d)) - 0.5/fs
        dd = np.diff(d)
        dd = dd[dd > 0]
    else:
        dd = np.array([])
    if dd.size > 0:
        PostStimSilence = np.min(dd) - 0.5/fs
    else:
        PostStimSilence = 0

    low, high = state_mod_split(rec, epoch=epoch, psth_name=psth_name,
                                channel=channel, state_sig=state_sig,
                                state_chan=state_chan)
    if psth_name2 is not None:
        low2, high2 = state_mod_split(rec, epoch=epoch, psth_name=psth_name2,
                                      channel=channel, state_sig=state_sig,
                                      state_chan=state_chan)

    if decimate_by > 1:
        low = scipy.signal.decimate(low, q=decimate_by, axis=1)
        high = scipy.signal.decimate(high, q=decimate_by, axis=1)
        if psth_name2 is not None:
            low2 = scipy.signal.decimate(low2, q=decimate_by, axis=1)
            high2 = scipy.signal.decimate(high2, q=decimate_by, axis=1)
        fs /= decimate_by

    title = state_chan
    if state_chan == 'baseline':
        legend = None
    else:
        legend = ('Lo', 'Hi')

    timeseries_from_vectors([low, high], fs=fs, title=title, ax=ax,
                            legend=legend, time_offset=PreStimSilence,
                            colors=colors, ylabel="sp/sec")

    if psth_name2 is not None:
        timeseries_from_vectors([low2, high2], fs=fs, title=title, ax=ax,
                                linestyle='--', time_offset=PreStimSilence,
                                colors=colors, ylabel="sp/sec")

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    ax.plot(np.array([0, 0]), ylim, 'k--')
    ax.plot(np.array([xlim[1], xlim[1]])-PostStimSilence, ylim, 'k--')

    if state_chan == 'baseline':
        ax.set_xlabel(epoch)


def state_gain_plot(modelspec, ax=None, clim=None, title=None):
    for m in modelspec:
        if ('state_dc_gain' in m['fn']):
            g = m['phi']['g'][0, :]
            d = m['phi']['d'][0, :]
        elif ('state_dexp' in m['fn']):
            # hack, sdexp currently only supports single output channel
            g = m['phi']['g']
            d = m['phi']['d']
    MI = modelspec[0]['meta']['state_mod']
    state_chans = modelspec[0]['meta']['state_chans']
    if ax is not None:
        plt.sca(ax)
    plt.plot(d)
    plt.plot(g)
    plt.plot(MI)
    plt.xticks(np.arange(len(state_chans)), state_chans, fontsize=6)
    plt.legend(('baseline', 'gain', 'MI'))
    plt.plot(np.arange(len(state_chans)),np.zeros(len(state_chans)),'k--',
             linewidth=0.5)
    if title:
        plt.title(title)

    ax_remove_box(ax)
