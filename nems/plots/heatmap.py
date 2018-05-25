import logging
import matplotlib.pyplot as plt
import numpy as np
import nems.modelspec as ms

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


def _get_wc_coefficients(modelspec):
    # TODO: what about modelspecs with multiple weight_channesl?
    for m in modelspec:
        if 'weight_channels' in m['fn']:
            if 'fn_coefficients' in m.keys():
                fn = ms._lookup_fn_at(m['fn_coefficients'])
                kwargs = {**m['fn_kwargs'], **m['phi']}  # Merges dicts
                return fn(**kwargs)
            else:
                return m['phi']['coefficients']
    return None


def _get_fir_coefficients(modelspec):
    # TODO: what about modelspecs with multiple fir filters?
    for m in modelspec:
        if 'fir' in m['fn']:
            return m['phi']['coefficients']
    return None


def weight_channels_heatmap(modelspec, ax=None, clim=None, title=None):
    coefficients = _get_wc_coefficients(modelspec)
    plot_heatmap(coefficients, xlabel='Channel In', ylabel='Channel Out',
                 ax=ax, clim=clim, title=title)


def fir_heatmap(modelspec, ax=None, clim=None, title=None):
    coefficients = _get_fir_coefficients(modelspec)
    plot_heatmap(coefficients, xlabel='Time Bin', ylabel='Channel In',
                 ax=ax, clim=clim, title=title)


def strf_heatmap(modelspec, ax=None, clim=None, show_factorized=True,
                 title=None, fs=None):
    wcc = _get_wc_coefficients(modelspec)
    firc = _get_fir_coefficients(modelspec)
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
        strf = wc_coefs @ fir_coefs

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
