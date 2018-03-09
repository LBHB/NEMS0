import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(sig1, sig2, ax=None, title=None, smoothing_bins=False,
                 xlabel=None, ylabel=None, legend=True):
    '''
    Uses the channels of sig1 to place points along the x axis, and channels of
    sig2 for distances along the y axis. If sig1 has one channel but sig2 has
    multiple channels, then all of sig2's channels will be plotted against the
    values from sig1. If sig1 has more than 1 channel, then sig2 must have the
    same number of channels, because XY coordinates will be determined from
    the same channel of both sig1 and sig2.

    Optional arguments:
        ax
        smoothing_bins: int
        xlabel
        ylabel
        legend
    '''
    if sig1.nchans > 1 and sig1.nchans != sig2.nchans:
        m = 'sig1 and sig2 must have same # of chans if sig1 is multichannel'
        raise ValueError(m)

    if ax:
        plt.sca(ax)

    m1 = sig1.as_continuous()
    m2 = sig2.as_continuous()
    
    # remove NaNs
    keepidx = np.isfinite(m1[0,:]) * np.isfinite(m2[0,:])
    m1 = m1[0:1, keepidx]
    m2 = m2[0:1, keepidx]

    for i in range(sig2.nchans):
        if sig1.nchans > 1:
            x = m1[[i], :]
        else:
            x = m1[[0], :]
        y = m2[[i], :]

        if smoothing_bins:
            
            # Concatenate and sort
            s2 = np.append(x, y, 0)
            s2 = s2[:, s2[0, :].argsort()]
            # ????
            bincount = np.min([smoothing_bins, s2.shape[1]])
            T = np.int(np.floor(s2.shape[1] / bincount))
            s2 = s2[:, 0:(T * bincount)]
            s2 = np.reshape(s2, [2, bincount, T])
            s2 = np.mean(s2, 2)
            s2 = np.squeeze(s2)
            x = s2[0, :]
            y = s2[1, :]

        chan_name = 'Channel {}'.format(i) if not sig2.chans else sig2.chans[i]
        plt.scatter(x, y, label=chan_name)

    xlabel = xlabel if xlabel else sig1.name
    plt.xlabel(xlabel)

    ylabel = ylabel if ylabel else sig2.name
    plt.ylabel(ylabel)

    if legend and sig2.nchans > 1:
        plt.legend(loc='best')

    if title:
        plt.title(title)
