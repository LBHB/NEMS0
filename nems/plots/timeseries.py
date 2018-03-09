import numpy as np
import matplotlib.pyplot as plt
from nems.signal import Signal
from nems.plots.assemble import pad_to_signals


def plot_timeseries(times, values, xlabel='Time', ylabel='Value',
                    legend=None, ax=None):
    '''
    Plots a simple timeseries with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : list of vectors
    values : list of vectors
    xlabel : str
    ylabel : str
    legend : list of strings
    TODO: expand this doc  -jacob 2-17-18
    '''
    for t, v in zip(times, values):
        ax.plot(t, v)

    ax.margins(x=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(legend)


def timeseries_from_signals(signals, channels=0, xlabel='Time', ylabel='Value',
                            ax=None):
    """TODO: doc"""
    channels = pad_to_signals(signals, channels)

    legend = [s.name for s in signals]
    times = []
    values = []
    for s, c in zip(signals, channels):
        # Get values from specified channel
        value_vector = s.as_continuous()[c]
        # Convert indices to absolute time based on sampling frequency
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
    plot_timeseries(times, values, xlabel, ylabel, legend, ax=ax)


def timeseries_from_epoch(signals, epoch, occurrences=0, channels=0,
                          xlabel='Time', ylabel='Value', ax=None):
    """TODO: doc"""
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
    plot_timeseries(times, values, xlabel, ylabel, legend, ax=ax)
