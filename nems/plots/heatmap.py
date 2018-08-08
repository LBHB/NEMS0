import logging
import matplotlib.pyplot as plt
import numpy as np
import nems.modelspec as ms

from nems.plots.timeseries import plot_timeseries

log = logging.getLogger(__name__)


def plot_heatmap(array, xlabel='Time', ylabel='Channel',
                 ax=None, cmap=None, clim=None, skip=0, title=None, fs=None):
    '''
    A wrapper for matplotlib's plt.imshow() to ensure consistent formatting.
    '''
    if ax is not None:
        plt.sca(ax)

    # Make sure array is converted to ndarray if passed as list
    array = np.array(array)

    if clim is None:
        mmax = np.nanmax(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    if fs is not None:
        extent = [0.5/fs, (array.shape[1]+0.5)/fs, 0.5, array.shape[0]+0.5]
    else:
        extent = None

    plt.imshow(array, aspect='auto', origin='lower',
               cmap=plt.get_cmap('jet'), clim=clim,
               interpolation='none', extent=extent)

    # Force integer tick labels, skipping gaps
    #y, x = array.shape

    #plt.xticks(np.arange(skip, x), np.arange(0, x-skip))
    #plt.xticklabels(np.arange(0, x-skip))
    #plt.yticks(np.arange(skip, y), np.arange(0, y-skip))
    #plt.yticklabels(np.arange(0, y-skip))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set the color bar
    # cbar = ax.colorbar()
    # cbar.set_label('Gain')
    if title is not None:
        plt.title(title)


def _get_wc_coefficients(modelspec, idx=0):
    i = 0
    for m in modelspec:
        if 'weight_channels' in m['fn']:
            if 'fn_coefficients' in m.keys():
                if i == idx:
                    fn = ms._lookup_fn_at(m['fn_coefficients'])
                    kwargs = {**m['fn_kwargs'], **m['phi']}  # Merges dicts
                    return fn(**kwargs)
                else:
                    i += 1
            else:
                return m['phi']['coefficients']
    return None


def _get_fir_coefficients(modelspec, idx=0):
    i = 0
    for m in modelspec:
        if 'fir' in m['fn']:
            if i == idx:
                return m['phi']['coefficients']
            else:
                i += 1
    return None


def weight_channels_heatmap(modelspec, ax=None, clim=None, title=None,
                            chans=None, wc_idx=0):
    coefficients = _get_wc_coefficients(modelspec, idx=wc_idx)
    plot_heatmap(coefficients, xlabel='Channel In', ylabel='Channel Out',
                 ax=ax, clim=clim, title=title)
    if chans is not None:
        for i, c in enumerate(chans):
            plt.text(i, 0, c)


def fir_heatmap(modelspec, ax=None, clim=None, title=None, chans=None,
                fir_idx=0):
    coefficients = _get_fir_coefficients(modelspec, idx=fir_idx)
    plot_heatmap(coefficients, xlabel='Time Bin', ylabel='Channel In',
                 ax=ax, clim=clim, title=title)
    if chans is not None:
        for i, c in enumerate(chans):
            plt.text(-0.4, i, c, verticalalignment='center')

def strf_heatmap(modelspec, ax=None, clim=None, show_factorized=True,
                 title=None, fs=None, chans=None, wc_idx=0, fir_idx=0):
    """
    chans: list
       if not None, label each row of the strf with the corresponding
       channel name
    """
    wcc = _get_wc_coefficients(modelspec, idx=wc_idx)
    firc = _get_fir_coefficients(modelspec, idx=fir_idx)
    if wcc is None and firc is None:
        log.warn('Unable to generate STRF.')
        return
    elif wcc is None and firc is not None:
        strf = np.array(firc)
        show_factorized = False
    elif wcc is not None and firc is None:
        strf = np.array(wcc).T
        show_factorized = False
    else:
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
        if wc_coefs.shape[1] == fir_coefs.shape[0]:
            strf = wc_coefs @ fir_coefs
        else:
            strf = fir_coefs
            show_factorized = False

    if not clim:
        cscale = np.nanmax(np.abs(strf.reshape(-1)))
        clim = [-cscale, cscale]
    else:
        cscale = np.max(np.abs(clim))

    if show_factorized:
        # Never rescale the STRF or CLIM!
        # The STRF should be the final word and respect input colormap and clim
        # However: rescaling WC and FIR coefs to make them more visible is ok
        wc_max = np.nanmax(np.abs(wc_coefs[:]))
        fir_max = np.nanmax(np.abs(fir_coefs[:]))
        wc_coefs = wc_coefs * (cscale / wc_max)
        fir_coefs = fir_coefs * (cscale / fir_max)

        n_inputs, _ = wc_coefs.shape
        nchans, ntimes = fir_coefs.shape
        gap = np.full([nchans + 1, nchans + 1], np.nan)
        horz_space = np.full([1, ntimes], np.nan)
        vert_space = np.full([n_inputs, 1], np.nan)
        top_right = np.concatenate([fir_coefs, horz_space], axis=0)
        top_left = np.concatenate([wc_coefs, vert_space], axis=1)
        bot = np.concatenate([top_left, strf], axis=1)
        top = np.concatenate([gap, top_right], axis=1)
        everything = np.concatenate([top, bot], axis=0)
        skip = nchans + 1
    else:
        everything = strf
        skip = 0

    plot_heatmap(everything, xlabel='Lag (s)',
                 ylabel='Channel In', ax=ax, skip=skip, title=title, fs=fs)
    if chans is not None:
        for i, c in enumerate(chans):
            plt.text(0, i + nchans + 1, c, verticalalignment='center')


def strf_timeseries(modelspec, ax=None, clim=None, show_factorized=True,
                    show_fir_only=True,
                    title=None, fs=1, chans=None):
    """
    chans: list
       if not None, label each row of the strf with the corresponding
       channel name
    """

    wcc = _get_wc_coefficients(modelspec)
    firc = _get_fir_coefficients(modelspec)
    if wcc is None and firc is None:
        log.warn('Unable to generate STRF.')
        return
    elif show_fir_only or (wcc is None):
        strf = np.array(firc)
    elif wcc is not None and firc is None:
        strf = np.array(wcc).T
    else:
        wc_coefs = np.array(wcc).T
        fir_coefs = np.array(firc)
        if wc_coefs.shape[1] == fir_coefs.shape[0]:
            strf = wc_coefs @ fir_coefs
        else:
            strf = fir_coefs

    times=np.arange(strf.shape[1])/fs
    plot_timeseries([times], [strf.T], xlabel='Time lag', ylabel='Gain',
                    legend=chans, linestyle='-', linewidth=1,
                    ax=ax, title=title)
