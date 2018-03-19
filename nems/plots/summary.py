from functools import partial
import logging
import numpy as np

from nems.plots.assemble import plot_layout
from nems.plots.heatmap import (weight_channels_heatmap, fir_heatmap,
                                strf_heatmap)
from nems.plots.scatter import plot_scatter
from nems.plots.spectrogram import spectrogram_from_epoch
from nems.plots.timeseries import timeseries_from_epoch
from nems.plots.histogram import pred_error_hist
import nems.modelspec as ms

log = logging.getLogger(__name__)


def plot_summary(rec, modelspecs):
    '''
    Plots a summary of the modelspecs and their performance predicting on rec.
    '''
    if not modelspecs:
        raise ValueError('No modelspecs defined')

    if type(rec) is list:
        rec=rec[0]

    stim = rec['stim']
    resp = rec['respavg'] if 'respavg' in rec.signals else rec['resp']

    # Make predictions on the data set using the modelspecs
    pred = [ms.evaluate(rec, m)['pred'] for m in modelspecs]

    sigs = [resp]
    sigs.extend(pred)

    # Example of how to plot a complicated thing:
    extracted = resp.extract_epoch('TRIAL')
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    occurrence = occurrences[0]

    def my_scatter_raw(idx, ax):
        plot_scatter(pred[idx], resp, ax=ax, title=rec.name)

    def my_scatter(idx, ax):
        plot_scatter(pred[idx], resp, ax=ax, title=rec.name,
                     smoothing_bins=100)

    def my_spectro(ax):
        spectrogram_from_epoch(stim, 'TRIAL', ax=ax, occurrence=occurrence)

    def my_timeseries(ax):
        timeseries_from_epoch(sigs, 'TRIAL', ax=ax, occurrences=occurrence)

    def my_strf(idx, ax):
        strf_heatmap(modelspecs[idx], ax=ax)

    def my_wc(idx, ax):
        weight_channels_heatmap(modelspecs[idx], ax=ax)

    def my_fir(idx, ax):
        fir_heatmap(modelspecs[idx], ax=ax)

    def my_hist(idx, ax):
        pred_error_hist(resp, pred[idx])

    def make_partials(fn, items):
        partials = [partial(fn, i) for i in range(len(items))]
        return partials

    if len(modelspecs) <= 10:
        fig = plot_layout([[my_spectro],
                           #make_partials(my_wc, modelspecs),
                           #make_partials(my_fir, modelspecs),
                           make_partials(my_strf, modelspecs),
                           [my_timeseries],
                           make_partials(my_scatter, modelspecs),
                           make_partials(my_hist, modelspecs)])
    else:
        # Don't plot the scatters/strfs when you have more than 10
        fig = plot_layout([[my_spectro],
                           [my_timeseries]])

    fig.tight_layout()
    return fig
