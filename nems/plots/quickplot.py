import importlib
import logging
import copy
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import nems.utils
import nems.modelspec as ms

# Better way to do this than to copy all of .api's imports?
# Can't use api b/c get circular import issue
from .scatter import plot_scatter
from .spectrogram import (plot_spectrogram, spectrogram_from_signal,
                          spectrogram_from_epoch)
from .timeseries import timeseries_from_signals, timeseries_from_epoch
from .heatmap import weight_channels_heatmap, fir_heatmap, strf_heatmap
from .file import save_figure, load_figure_img, load_figure_bytes, fig2BytesIO
from .histogram import pred_error_hist

log = logging.getLogger(__name__)

# NOTE: Currently the quickplot function is set up to hard-code LBHB-specific
#       conventions (like signal names 'resp', 'pred', and 'stim')
#       so it may not be useable as-is by other groups. Making it 100%
#       flexible and agnostic was just unfeasible. However, none of it
#       depends on anything outside of the main nems repository so it should
#       serve as a good model for a plot script that others can adapt to their
#       needs.

# NOTE: To LBHB team or other contributors:
#       To support additional modules with quickplot, their behavior
#       should be specified within the if/else loop in _get_plot_fns,
#       which follows the format (within each iteration):
#
#       if 'module_name' matched:
#           if 'function_name' matched:
#               <append a partial plot for this>
#           elif 'other_function_name' matched:
#               <append a partial plot for this instead>
#           else:
#               <don't plot anything>
#
#       The 'partial plot' that gets appended should be a tuple of lists,
#       of the form:
#           ([plot_fn1, fn2, ...], [col_span1, span2, ...])
#       Or if only one plot is needed, a special shorthand: (plot_fn, 1)
#       (See _get_plot_fns for examples)
#                                                           -jacob 3/31/2018


# TODO: _get_plot_fns has a lot of extra spacing in an attempt to make
#      the module names stand out for easy scanning. I felt this was
#      necessary since it is meant to be edited frequently as modules
#      are added or changed. If others find this
#      annoying or have a better solution, feel free to re-organize.
#      -jacob 3/31/2018

def quickplot(rec, modelspec, occurrence=0, figsize=None, height_mult=2):
    plot_fns = _get_plot_fns(rec, modelspec, occurrence=occurrence)

    # Need to know how many total plots for outer gridspec (n).
    # +2 is to account for module-independent scatter at end
    # and spectrogram at beginning.
    # If other independent plots are added, will need to
    # adjust this calculation.
    n = len(plot_fns)+2
    if figsize is None:
        fig = plt.figure()
    else:
        fig = plt.figure(figsize=(9, n*height_mult))
    gs_outer = gridspec.GridSpec(n, 1)

    # Each plot will be represented as a nested gridspec.
    # That way, plots have control over how many subplots
    # they use etc. Only restriction is that they get
    # one row (but the if/else flow control above could
    # add more than one plot for a module if multiple
    # rows are needed).

    def _plot_axes(col_spans, fns, outer_index):
        """Expects col_spans and fns to be lists, outer_index integer."""
        # Special check to allow shorthand for single column and fn
        if isinstance(col_spans, int):
            if not isinstance(fns, list):
                fns = [fns]
                col_spans = [col_spans]
            else:
                raise ValueError("col_spans and fns must either both be"
                                 "lists or both be singular, got:\n {}\n{}"
                                 .format(col_spans, fns))

        n_cols = sum(col_spans)
        g = gridspec.GridSpecFromSubplotSpec(
                1, n_cols, subplot_spec=gs_outer[outer_index]
                )
        i = 0
        for j, span in enumerate(col_spans):
            ax = plt.Subplot(fig, g[0, i:i+span])
            fig.add_subplot(ax)
            fns[j](ax=ax)
            i += span


    ### Special plots that go *BEFORE* iterated modules

    # Stimulus Spectrogram
    fn_spectro = partial(
            spectrogram_from_epoch, rec['stim'], 'TRIAL',
            occurrence=occurrence, title='Stimulus Spectrogram'
            )
    _plot_axes([1], [fn_spectro], 0)


    ### Iterated module plots (defined in _get_plot_fns)
    for i, (fns, col_spans) in enumerate(plot_fns):
        # +1 because we did spectrogram above. Adjust as necessary.
        j = i+1
        _plot_axes(col_spans, fns, j)


    ### Special plots that go *AFTER* iterated modules

    # Pred v Act Scatter Smoothed
    r_test = modelspec[0]['meta']['r_test']
    pred = rec['pred']
    resp = rec['resp']
    title = "{0} r_test={1:.3f}".format(rec.name, r_test)
    smoothed = partial(
            plot_scatter, pred, resp, title=title, smoothing_bins=100
            )
    not_smoothed = partial(
            plot_scatter, pred, resp, title=title, smoothing_bins=False
            )
    _plot_axes([1, 1], [smoothed, not_smoothed], -1)

    fig.tight_layout(pad=1.5, w_pad=1.0, h_pad=2.5)
    return fig


def _get_plot_fns(rec, modelspec, occurrence=0):
    plot_fns = []

    # TODO: This feels a bit hacky, likely needs review.
    modules = [m['fn'] for m in modelspec]
    check_wc = [('weight_channels' in mod) for mod in modules]
    check_fir = [('fir' in mod) for mod in modules]
    if (sum(check_wc) > 0) and (sum(check_fir) > 0):
        # if both weight channels and fir are present, do STRF heatmap
        # instead of either of the individual heatmaps for those modules
        do_strf = True
    else:
        do_strf = False
    # Only do STRF once
    strf_done = False

    for m in modelspec:
        fn = m['fn']

        # STRF is a special case that relies on multiple modules, so
        # the dependent modules are wrapped here
        # in a separate logic heirarchy.
        if not do_strf:
            if 'weight_channels' in fn:

                if 'weight_channels.basic' in fn:
                    fn = partial(weight_channels_heatmap, modelspec)
                    plot = (fn, 1)
                    plot_fns.append(plot)

                elif 'weight_channels.gaussian' in fn:
                    fn = partial(weight_channels_heatmap, modelspec)
                    plot = (fn, 1)
                    plot_fns.append(plot)

                else:
                    # Don't plot anything
                    pass

            elif 'fir' in fn:

                if 'fir.basic' in fn:
                    fn = partial(fir_heatmap, modelspec)
                    plot = (fn, 1)
                    plot_fns.append(plot)
                else:
                    pass
        # do strf
        else:
            if not strf_done:
                fn = partial(strf_heatmap, modelspec)
                plot = (fn, 1)
                plot_fns.append(plot)
                strf_done = True
                continue
            else:
                pass


        if 'levelshift' in fn:
            if 'levelshift.levelshift' in fn:
                # TODO
                pass
            else:
                pass


        elif 'nonlinearity' in fn:

            if 'nonlinearity.double_exponential' in fn:
                pass
            elif 'nonlinearity.quick_sigmoid' in fn:
                pass
            elif 'nonlinearity.logistic_sigmoid' in fn:
                pass
            elif 'nonlinearity.tanh' in fn:
                pass
            elif 'nonlinearity.dlog' in fn:
                pass
            else:
                pass

        elif 'signal_mod' in fn:
            if 'signal_mod.make_state_signal' in fn:
                pass
            elif 'signal_mod.average_sig' in fn:
                pass
            else:
                pass

        elif 'state' in fn:
            if 'state.state_dc_gain' in fn:
                pass
            else:
                pass

    return plot_fns


#######################      QUICKPLOT WRAPPERS      ##########################
# NOTE: Plot functions referenced by defaults should have
#       rec and modelspec as their first two positional arguments.
#       This will likely require defining a wrapper function for an
#       existing plot function that computes the necessary arguments.
# TODO: Is this a good idiom? Simplest standardization I coudl think of
#       that ensures plot functions still get the information they
#       are dependent on. Alternative solutions seem equally awkward,
#       maybe a question for bburan.


def before_and_after(rec, modelspec, sig_name, idx, ax=None, title=None,
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
        before = ms.evaluate(rec, modelspec, start=None, stop=idx)[sig_name]
        before.name += ' before'

    after = ms.evaluate(rec, modelspec, start=idx, stop=idx+1)[sig_name].copy()
    after.name += ' after'
    timeseries_from_signals([before, after], channels=channels,
                            xlabel=xlabel, ylabel=ylabel, ax=ax,
                            title=title)


def fir_heatmap_quick(rec, modelspec, idx=None, ax=None, title=None,
                      clim=None):
    fir_heatmap(modelspec, ax=ax, clim=clim, title=title)


def wc_heatmap_quick(rec, modelspec, idx=None, ax=None, title=None,
                     clim=None):
    weight_channels_heatmap(modelspec, ax=ax, clim=clim, title=title)


def _get_pred_resp(rec, modelspec, pred_name, resp_name):
    """Evaluates rec using a fitted modelspec, then returns pred and resp
    signals."""
    new_rec = ms.evaluate(rec, modelspec)
    return [new_rec[pred_name], new_rec[resp_name]]


def pred_resp_scatter(rec, modelspec, pred_name='pred', resp_name='resp',
                      idx=None, ax=None, title=None, smoothing_bins=False):
    pred, resp = _get_pred_resp(rec, modelspec, pred_name, resp_name)
    plot_scatter(pred, resp, ax=ax, title=title,
                 smoothing_bins=smoothing_bins, xlabel='Time',
                 ylabel='Firing Rate')


def error_hist(rec, modelspec, pred_name='pred', resp_name='resp', ax=None,
               channel=0, bins=None, bin_mult=5.0, xlabel='|Resp - Pred|',
               ylabel='Count', title='Prediction Error', idx=None):
    pred, resp = _get_pred_resp(rec, modelspec, pred_name, resp_name)
    pred_error_hist(resp, pred, ax=ax, channel=channel, bins=bins,
                    bin_mult=bin_mult, xlabel=xlabel, ylabel=ylabel,
                    title=title)
