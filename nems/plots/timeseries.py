import numpy as np
import matplotlib.pyplot as plt

from nems.plots.assemble import pad_to_signals
import nems.modelspec as ms
import nems.signal as signal
import nems.recording as recording
import nems.modules.stp as stp


def plot_timeseries(times, values, xlabel='Time', ylabel='Value', legend=None,
                    linestyle='-', linewidth=1,
                    ax=None, title=None):
    '''
    Plots a simple timeseries with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : list of vectors
    values : list of vectors
    xlabel : str
    ylabel : str
    legend : list of strings
    linestyle, linewidth : pass-through options to plt.plot()

    TODO: expand this doc  -jacob 2-17-18
    '''
    if ax is not None:
        plt.sca(ax)

    for t, v in zip(times, values):
        plt.plot(t, v, linestyle=linestyle, linewidth=linewidth)

    plt.margins(x=0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)
    if title:
        plt.title(title)


def timeseries_from_vectors(vectors, xlabel='Time', ylabel='Value', fs=None,
                            linestyle='-', linewidth=1, legend=None,
                            ax=None, title=None, time_offset=0):
    """TODO: doc"""
    times = []
    values = []
    for v in vectors:
        values.append(v)
        if fs is None:
            times.append(np.arange(0, len(v)))
        else:
            times.append(np.arange(0, len(v))/fs)
    plot_timeseries(times - time_offset, values, xlabel, ylabel,
                    legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


def timeseries_from_signals(signals, channels=0, xlabel='Time', ylabel='Value',
                            linestyle='-', linewidth=1,
                            ax=None, title=None):
    """TODO: doc"""
    channels = pad_to_signals(signals, channels)

    times = []
    values = []
    legend = []
    for s, c in zip(signals, channels):
        # Get values from specified channel
        value_vector = s.as_continuous()[c]
        # Convert indices to absolute time based on sampling frequency
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
        legend.append(s.name+' '+s.chans[c])

    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


def timeseries_from_epoch(signals, epoch, occurrences=0, channels=0,
                          xlabel='Time', ylabel='Value',
                          linestyle='-', linewidth=1,
                          ax=None, title=None):
    """TODO: doc"""
    if occurrences is None:
        return
    occurrences = pad_to_signals(signals, occurrences)
    channels = pad_to_signals(signals, channels)

    legend = [s.name for s in signals]
    times = []
    values = []
    for s, o, c in zip(signals, occurrences, channels):
        # Get occurrences x chans x time
        extracted = s.extract_epoch(epoch)
        # Get values from specified occurrence and channel
        value_vector = extracted[o][c]
        # Convert bins to time (relative to start of epoch)
        # TODO: want this to be absolute time relative to start of data?
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
    plot_timeseries(times, values, xlabel, ylabel, legend=legend,
                    linestyle=linestyle, linewidth=linewidth,
                    ax=ax, title=title)


def before_and_after_stp(modelspec, sig_name='pred', ax=None, title=None,
                         channels=0, xlabel='Time', ylabel='Value', fs=100):
    '''
    Plots a timeseries of specified signal just before and just after
    the transformation performed at some step in the modelspec.

    Arguments:
    ----------
    rec : recording object
        really only used to get the sampling rate, since we're using
        a cartoon stimulus

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    Returns:
    --------
    None
    '''

    for m in modelspec:
        if 'stp' in m['fn']:
            break
    c = len(m['phi']['tau'])
    seg = np.int(fs * 0.05)
    A=0.5
    pred = np.concatenate([np.zeros([c, seg * 2]), np.ones([c, seg * 4]) * A,
                           np.zeros([c, seg * 4]), np.ones([c, seg]) * A,
                           np.zeros([c, seg]), np.ones([c, seg]) * A,
                           np.zeros([c, seg]), np.ones([c, seg]) * A,
                           np.zeros([c, seg * 2])], axis=1)

    kwargs = {
        'data': pred,
        'name': 'pred',
        'recording': 'rec',
        'chans': ['chan' + str(n) for n in range(c)],
        'fs': fs,
        'meta': {},
    }
    pred = signal.RasterizedSignal(**kwargs)
    r = recording.Recording({'pred': pred})

    u = m['phi']['u']
    tau = m['phi']['tau']

    r = stp.short_term_plasticity(r, 'pred', 'pred_out', u=u, tau=tau)
    pred_out = r[0]

    pred.name = 'before'
    pred_out.name = 'after'
    signals = [pred]
    channels = [0]
    for i in range(c):
        signals.append(pred_out)
        channels.append(i)

    timeseries_from_signals(signals, channels=channels,
                            xlabel=xlabel, ylabel=ylabel, ax=ax,
                            title=title)


def before_and_after(rec, modelspec, sig_name, ax=None, title=None,
                     channels=0, xlabel='Time', ylabel='Value'):
    '''
    Plots a timeseries of specified signal just before and just after
    the transformation performed at some step in the modelspec.

    Arguments:
    ----------
    rec : recording object
        The dataset to use. See nems/recording.py.

    modelspec : list of dicts
        The transformations to perform. See nems/modelspec.py.

    sig_name : str
        Specifies the signal in 'rec' to be examined.

    idx : int
        An index into the modelspec. rec[sig_name] will be plotted
        as it exists after step idx-1 and after step idx.

    Returns:
    --------
    None
    '''
    # HACK: shouldn't hardcode 'stim', might be named something else
    #       or not present at all. Need to figure out a better solution
    #       for special case of idx = 0
    if idx == 0:
        before = rec['stim'].copy()
        before.name += ' before**'
    else:
        before = ms.evaluate(rec.copy(), modelspec, start=None, stop=idx)[sig_name]
        before.name += ' before'

    after = ms.evaluate(rec, modelspec, start=idx, stop=idx+1)[sig_name].copy()
    after.name += ' after'
    timeseries_from_signals([before, after], channels=channels,
                            xlabel=xlabel, ylabel=ylabel, ax=ax,
                            title=title)
