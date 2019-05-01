import matplotlib.pyplot as plt
import numpy as np
import scipy

from .timeseries import (timeseries_from_signals, timeseries_from_vectors,
                         ax_remove_box)
import nems.modelspec as ms
from nems.utils import get_channel_number
from nems.metrics.state import state_mod_split
from nems.plots.utils import ax_remove_box


line_colors = {'actual_psth': (0,0,0),
               'predicted_psth': 'red',
               #'passive': (255/255, 133/255, 133/255),
               'passive': (216/255, 151/255, 212/255),
               #'active': (196/255, 33/255, 43/255),
               'active': (129/255, 64/255, 138/255),
               'false_alarm': (79/255, 114/255, 184/255),
               'miss': (183/255, 196/255, 229/255),
               'hit': (36/255, 49/255, 103/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'pas1': 'green',
               'pas2': (153/255, 124/255, 248/255),
               'pas3': (173/255, 144/255, 255/255),
               'pas4': (193/255, 164/255, 255/255),
               'pas5': 'green',
               'pas6': (123/255, 104/255, 238/255),
               'hard': (196/255, 149/255, 44/255),
               'easy': (255/255, 206/255, 6/255),
               'puretone': (247/255, 223/255, 164/255),
               'large': (44/255, 125/255, 61/255),
               'small': (181/255, 211/255, 166/255)}
fill_colors = {'actual_psth': (.8,.8,.8),
               'predicted_psth': 'pink',
               #'passive': (226/255, 172/255, 185/255),
               'passive': (234/255, 176/255, 223/255),
               #'active': (244/255, 44/255, 63/255),
               'active': (163/255, 102/255, 163/255),
               'false_alarm': (107/255, 147/255, 204/255),
               'miss': (200/255, 214/255, 237/255),
               'hit': (78/255, 92/255, 135/255),
               'pre': 'green',
               'post': (123/255, 104/255, 238/255),
               'hard':  (229/255, 172/255, 57/255),
               'easy': (255/255, 225/255, 100/255),
               'puretone': (255/255, 231/255, 179/255),
               'large': (69/255, 191/255, 89/255),
               'small': (215/255, 242/255, 199/255)}

def state_vars_timeseries(rec, modelspec, ax=None, state_colors=None,
                          decimate_by=1, channel=None, **options):

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
        sp = None
        for m in modelspec:
            if ('state_dc_gain' in m['fn']):
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
            elif ('state_sp_dc_gain' in m['fn']):
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
                sp = np.array(m['phi']['sp'])
            elif ('state_gain' in m['fn']):
                g = np.array(m['phi']['g'])

        if g is not None:
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
        offset = -1.25 * mmax
        for i in range(1, num_vars):

            st = ts[i, :].T
            if (len(np.unique(st)) == 2) and num_vars > 3:
                # special, binary variable, keep in one row
                m = np.array([np.min(st)])
                st = np.concatenate((m, st, m))
                dinc = np.argwhere(np.diff(st) > 0)
                ddec = np.argwhere(np.diff(st) < 0)
                for x0, x1 in zip(dinc, ddec):
                    plt.plot([x0/fs, x1/fs], [offset, offset],
                             lw=2, color=state_colors[i-1])
                tstr = "{}".format(rec['state'].chans[i])
                plt.text(x0/fs, offset, tstr, fontsize=6)
                #print("{} {} {}".format(rec['state'].chans[i], x0/fs, offset))
            else:
                # non-binary variable, plot in own row
                # figure out gain
                if sp is not None:
                    if g.ndim == 1:
                        tstr = "{} (sp={:.3f},d={:.3f},g={:.3f})".format(
                                rec['state'].chans[i], sp[i], d[i], g[i])
                    else:
                        tstr = "{} (sp={:.3f},d={:.3f},g={:.3f})".format(
                                rec['state'].chans[i], sp[0, i], d[0, i], g[0, i])
                elif g is not None:
                    if g.ndim == 1:
                        tstr = "{} (d={:.3f},g={:.3f})".format(
                                rec['state'].chans[i], d[i], g[i])
                    else:
                        tstr = "{} (d={:.3f},g={:.3f})".format(
                                rec['state'].chans[i], d[0, i], g[0, i])
                else:
                    tstr = "{}".format(rec['state'].chans[i])
                if decimate_by > 1:
                    st = scipy.signal.decimate(st[nnidx], q=decimate_by, axis=0)
                else:
                    st = st[nnidx]

                st = st / np.nanmax(st) * mmax + offset
                plt.plot(t, st, linewidth=1, color=state_colors[i-1])
                plt.text(t[0], offset, tstr, fontsize=6)

                offset -= 1.25*mmax

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


def state_var_psth_from_epoch(rec, epoch="REFERENCE", psth_name='resp', psth_name2='pred',
                              state_sig='state_raw', state_chan='pupil', ax=None,
                              colors=None, channel=None, decimate_by=1, **options):
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


def state_vars_psth_all(rec, epoch="REFERENCE", psth_name='resp', psth_name2='pred',
                        state_sig='state_raw', ax=None,
                        colors=None, channel=None, decimate_by=1,
                        files_only=False, modelspec=None, **options):

    # TODO: Does using epochs make sense for these?
    if ax is not None:
        plt.sca(ax)

    newrec = rec.copy()
    fn = lambda x: x - newrec['pred']._data
    newrec['error'] = rec['resp'].transform(fn, 'error')

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

    state_chan_list = rec[state_sig].chans
    low = np.zeros([0,1])
    high = np.zeros([0,1])
    lowE = np.zeros([0,1])
    highE = np.zeros([0,1])
    low2 = np.zeros([0,1])
    high2 = np.zeros([0,1])
    _high2 = None
    limitset = []
    if files_only: #state_chan_list =['a','p','PASSIVE_1']
        #print(state_chan_list)
        state_chan_list = [s for s in state_chan_list
                           if (s.startswith('FILE') | s.startswith('ACTIVE') |
                               s.startswith('PASSIVE')) ]
    for state_chan in state_chan_list:

        _low, _high = state_mod_split(rec, epoch=epoch, psth_name=psth_name,
                                    channel=channel, state_sig=state_sig,
                                    state_chan=state_chan)
        _lowE, _highE = state_mod_split(newrec, epoch=epoch, psth_name='error',
                                    channel=channel, state_sig=state_sig,
                                    state_chan=state_chan, stat=scipy.stats.sem)
        if psth_name2 is not None:
            _low2, _high2 = state_mod_split(rec, epoch=epoch, psth_name=psth_name2,
                                            channel=channel, state_sig=state_sig,
                                            state_chan=state_chan)

        gapdur = _low.shape[0]/fs/10
        gap = np.ones([int(np.ceil(fs*gapdur)),1]) * np.nan
        pgap = np.ones(_low.shape) * np.nan
        if files_only:
            if state_chan == state_chan_list[0]:
                low = np.concatenate((high,gap,_low,gap), axis=0)
                lowE = np.concatenate((highE,gap,_lowE,gap), axis=0)
                high = np.concatenate((high,gap,pgap,gap), axis=0)
                highE = np.concatenate((highE,gap,pgap,gap), axis=0)
                if psth_name2 is not None:
                    low2 = np.concatenate((low2,gap,_low2,gap), axis=0)
                    high2 = np.concatenate((high2,gap,pgap,gap), axis=0)
            current_start = high.shape[0]/fs + gapdur
            if state_chan.startswith('ACTIVE'):
                _low = pgap
                _low2 = pgap
            else:
                _low = _high.copy()
                _low2 = _high2.copy()
                _high = pgap
                _high2 = pgap

        else:
            current_start = low.shape[0]/fs + gapdur

        low = np.concatenate((low,gap,_low,gap), axis=0)
        high = np.concatenate((high,gap,_high,gap), axis=0)
        lowE = np.concatenate((lowE,gap,_lowE,gap), axis=0)
        highE = np.concatenate((highE,gap,_highE,gap), axis=0)

        if psth_name2 is not None:
            low2 = np.concatenate((low2,gap,_low2,gap), axis=0)
            high2 = np.concatenate((high2,gap,_high2,gap), axis=0)

        limitset += [[current_start + PreStimSilence,
                     current_start + _low.shape[0]/fs - PostStimSilence]]

    if decimate_by > 1:
        low = scipy.signal.decimate(low, q=decimate_by, axis=1)
        high = scipy.signal.decimate(high, q=decimate_by, axis=1)
        if psth_name2 is not None:
            low2 = scipy.signal.decimate(low2, q=decimate_by, axis=1)
            high2 = scipy.signal.decimate(high2, q=decimate_by, axis=1)
        fs /= decimate_by

    tt = np.arange(high.shape[0])/fs
    ax.fill_between(tt, low[:,0]-lowE[:,0], low[:,0]+lowE[:,0], color=fill_colors['passive'])
    l1, = ax.plot(tt, low, ls='-', lw=1, color=line_colors['passive'])
    ax.fill_between(tt, high[:,0]-highE[:,0], high[:,0]+highE[:,0], color=fill_colors['active'])
    l2, = ax.plot(tt, high, ls='-', lw=1, color=line_colors['active'])
    if psth_name2 is not None:
        ax.plot(tt, low2, ls='--', lw=1, color=line_colors['passive'])
        ax.plot(tt, high2, ls='--', lw=1, color=line_colors['active'])

    if not files_only:
        plt.legend((l1,l2), ('Lo','Hi'))

    ax.set_ylabel('sp/sec')
    ylim = ax.get_ylim()

    for ls, s in zip(limitset, state_chan_list):
        try:
            sc = modelspec[0]['meta']['state_chans']
            mi = modelspec[0]['meta']['state_mod']
            sn = "{} ({:.2f})".format(s,mi[sc.index(s)])
        except:
            sn = s
        ax.plot(ls, [ylim[1], ylim[1]], 'k-', linewidth=2)
        lc = np.mean(ls)
        ax.text(lc, ylim[1], sn, ha='center', va='bottom', fontsize=6)
    ax.set_ylim([ylim[0], ylim[1]*1.1])
    ax_remove_box(ax)


def state_gain_plot(modelspec, ax=None, colors=None, clim=None, title=None, **options):
    for m in modelspec:
        if ('state' in m['fn']):
            g = m['phi']['g']
            d = m['phi']['d']

    MI = modelspec[0]['meta']['state_mod']
    state_chans = modelspec[0]['meta']['state_chans']
    if ax is not None:
        plt.sca(ax)
    opt={}
    for cc in range(d.shape[1]):
        if colors is not None:
            opt = {'color': colors[cc]}
        plt.plot(d[:,cc],'--', **opt)
        plt.plot(g[:,cc], **opt)
    #plt.plot(MI)
    #plt.xticks(np.arange(len(state_chans)), state_chans, fontsize=6)
    plt.legend(('baseline', 'gain'))
    plt.plot(np.arange(len(state_chans)),np.zeros(len(state_chans)),'k--',
             linewidth=0.5)
    if title:
        plt.title(title)

    ax_remove_box(ax)


def model_per_time(ctx, fit_idx=0):
    """
    state_colors : N x 2 list
       color spec for high/low lines in each of the N states
    """
    if type(ctx['val']) is list:
        rec = ctx['val'][0].apply_mask()
    else:
        rec = ctx['val'].apply_mask()

    modelspec = ctx['modelspec']
    modelspec.fit_idx = 0

    epoch="REFERENCE"
    rec = ms.evaluate(rec, modelspec)

    plt.figure()
    ax = plt.subplot(2, 1, 1)
    state_vars_timeseries(rec, modelspec, ax=ax)

    ax = plt.subplot(2, 1, 2)
    state_vars_psth_all(rec, epoch, psth_name='resp',
                        psth_name2='pred', state_sig='state_raw',
                        colors=None, channel=None, decimate_by=1,
                        ax=ax, files_only=True, modelspec=modelspec)

