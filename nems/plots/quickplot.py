import logging
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import copy

import nems.modelspec as ms
import nems.metrics.api as nm

# Better way to do this than to copy all of .api's imports?
# Can't use api b/c get circular import issue
from nems.plots.scatter import plot_scatter
from nems.plots.spectrogram import (plot_spectrogram, spectrogram_from_signal,
                          spectrogram_from_epoch)
from nems.plots.timeseries import timeseries_from_signals, timeseries_from_epoch
from nems.plots.heatmap import weight_channels_heatmap, fir_heatmap, strf_heatmap
from nems.plots.histogram import pred_error_hist
from nems.plots.state import (state_vars_timeseries, state_var_psth_from_epoch,
                    state_var_psth)

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
#           if 'module_name.function_name' matched:
#               <append a partial plot for this>
#           elif 'module_name.other_function_name' matched:
#               <append a partial plot for this instead>
#           else:
#               <don't plot anything>
#
#       The 'partial plot' that gets appended should be a tuple of lists,
#       of the form:
#           ([plot_fn1, fn2, ...], [col_span1, span2, ...])
#
#       Or if only one plot is needed, a special shorthand: (plot_fn, 1)
#       (See _get_plot_fns for examples)
#                                                           -jacob 3/31/2018


# TODO: _get_plot_fns has a lot of extra spacing in an attempt to make
#      the module names stand out for easy scanning. I felt this was
#      necessary since it is meant to be edited frequently as modules
#      are added or changed. If others find this
#      annoying or have a better solution, feel free to re-organize.
#      -jacob 3/31/2018


def quickplot(ctx, default='val', epoch=None, occurrence=None, figsize=None,
              height_mult=3.0, width_mult=1.0, m_idx=0, r_idx=0):
    """Expects an *evaluated* context dictionary ('ctx') returned by xforms."""
    # TODO: Or do we want 'est' by default?
    #       Could also just
    #       ditch default altogether and force plots to explicityly state which
    #       dataset they're plotting, but then we lose the ability to quickly
    #       choose between plotting est or val.

    # Most plots will just use the default (typically 'val' for LBHB),
    # but some plots might want to plot est vs val or need access to the
    # full recording. Keeping the full ctx reference lets those plots
    # use ctx['est'], ctx['rec'], etc.
    rec = ctx[default][r_idx]
    modelspec = ctx['modelspecs'][m_idx]
    if not epoch:
        if rec['resp'].count_epoch('REFERENCE'):
            epoch = 'REFERENCE'
        else:
            epoch = 'TRIAL'

    extracted = rec['resp'].extract_epoch(epoch)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    if occurrence is None or occurrence>len(occurrence):
        if len(occurrences)==0:
            occurrence=None
        else:
            occurrence = occurrences[0]
    else:
        occurrence=occurrences[occurrence]

    # determine if 'stim' signal exists
    show_spectrogram = ('stim' in rec.signals.keys())

    plot_fns = _get_plot_fns(ctx, default=default, occurrence=occurrence,
                             epoch=epoch, m_idx=m_idx)

    # Need to know how many total plots for outer gridspec (n).
    # +3 is to account for module-independent plots at end
    # and spectrogram at beginning.
    # If other independent plots are added, will need to
    # adjust this calculation.
    if show_spectrogram:
        n = len(plot_fns)+3
    else:
        n = len(plot_fns)+2

    if figsize is None:
        fig = plt.figure(figsize=(10*width_mult, n*height_mult))
    else:
        fig = plt.figure(figsize=figsize)

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

    # TODO: Move pre- and post- plots to separate subfunctions?
    #       Not too awful at the moment but if we add more will
    #       get pretty crowded here

    ### Special plots that go *BEFORE* iterated modules

    # Stimulus Spectrogram
    # TODO: This is a bit screwy for state_gain model, do we want
    if show_spectrogram:
        fn_spectro = partial(
                spectrogram_from_epoch, rec['stim'], epoch,
                occurrence=occurrence, title='Stimulus Spectrogram'
                )
        _plot_axes([1], [fn_spectro], 0)


    ### Iterated module plots (defined in _get_plot_fns)
    for i, (fns, col_spans) in enumerate(plot_fns):
        # +1 because we did spectrogram above. Adjust as necessary.
        j = i + (1 if show_spectrogram else 0)
        _plot_axes(col_spans, fns, j)

    ### Special plots that go *AFTER* iterated modules

    # Pred v Resp Timeseries
    sigs = [rec['resp'], rec['pred']]
    title = 'Final Prediction vs Response, {} #{}'.format(epoch, occurrence)
    timeseries = partial(timeseries_from_epoch, sigs, epoch, title=title,
                         occurrences=occurrence)
    _plot_axes(1, timeseries, -2)

    # Pred v Resp Scatter Smoothed
    r_test = modelspec[0]['meta']['r_test']
    r_fit = modelspec[0]['meta']['r_fit']
    pred = rec['pred']
    resp = rec['resp']
    text = 'r_test: {0:.3f}\nr_fit: {1:.3f}'.format(r_test, r_fit)
    smoothed = partial(
            plot_scatter, pred, resp, text=text, smoothing_bins=100,
            title='Smoothed, bins={}'.format(100), force_square=False
            )
    not_smoothed = partial(
            plot_scatter, pred, resp, text=text, smoothing_bins=False,
            title='Unsmoothed', force_square=False,
            )
    _plot_axes([1, 1], [smoothed, not_smoothed], -1)

    # TODO: Pred Error histogram too? Or was that not useful?

    # Whole-figure title
    try:
        cellid = modelspec[0]['meta']['cellid']
    except KeyError:
        cellid = "UNKNOWN"
    try:
        modelname = modelspec[0]['meta']['modelname']
    except KeyError:
        modelname = "UNKNOWN"
    try:
        batch = modelspec[0]['meta']['batch']
    except KeyError:
        batch = 0
    fig.suptitle('Cell: {}, from Batch: {},\nUsing model: {},\n {} #{}'
                 .format(cellid, batch, modelname, epoch, occurrence))

    # Space subplots appropriately
    # TODO: More dynamic way to determine the y-max for suptitle?
    #y_max = 1.00 - (height_mult+1)/100
    y_max=0.955
    gs_outer.tight_layout(fig, rect=[0, 0, 1, y_max],
                          pad=1.5, w_pad=1.0, h_pad=2.5)
    return fig


def _get_plot_fns(ctx, default='val', epoch='TRIAL', occurrence=0, m_idx=0,
                  r_idx=0):
    rec = ctx[default][r_idx]
    modelspec = ctx['modelspecs'][m_idx]

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

    for idx, m in enumerate(modelspec):
        fname = m['fn']

        # STRF is a special case that relies on multiple modules, so
        # the dependent modules are wrapped here
        # in a separate logic heirarchy.
        if not do_strf:
            if 'weight_channels' in fname:

                if 'weight_channels.basic' in fname:
                    fn = partial(weight_channels_heatmap, modelspec)
                    plot = (fn, 1)
                    plot_fns.append(plot)

                elif 'weight_channels.gaussian' in fname:
                    fn = partial(weight_channels_heatmap, modelspec)
                    plot = (fn, 1)
                    plot_fns.append(plot)

                else:
                    # Don't plot anything
                    pass

            elif 'fir' in fname:

                if 'fir.basic' in fname:
                    fn = partial(fir_heatmap, modelspec)
                    plot = (fn, 1)
                    plot_fns.append(plot)
                else:
                    pass
        # do strf
        else:
            if not strf_done:
                fn = partial(strf_heatmap, modelspec, title='STRF')
                plot = (fn, 1)
                plot_fns.append(plot)
                strf_done = True
                continue
            else:
                pass

        if 'levelshift' in fname:
            if 'levelshift.levelshift' in fname:
                # TODO - Should levelshift plot anything?
                pass
            else:
                pass

        elif 'stp' in fname:
            # channels = np.arange(m['phi']['u'].size)
            channels = 0
            fn = before_and_after_psth(rec, modelspec, idx, sig_name='pred',
                                       epoch=epoch, occurrences=occurrence,
                                       channels=channels, mod_name='STP')
            plot = (fn, 1)
            plot_fns.append(plot)

        elif 'nonlinearity' in fname:

            if 'nonlinearity.double_exponential' in fname:
                fn1, fn2 = before_and_after_scatter(
                        rec, modelspec, idx, smoothing_bins=200,
                        mod_name='dexp'
                        )
                plots = ([fn1, fn2], [1, 1])
                plot_fns.append(plots)

            elif 'nonlinearity.quick_sigmoid' in fname:
                fn1, fn2 = before_and_after_scatter(
                        rec, modelspec, idx, smoothing_bins=200,
                        mod_name='quick_sig'
                        )
                plots = ([fn1, fn2], [1, 1])
                plot_fns.append(plots)

            elif 'nonlinearity.logistic_sigmoid' in fname:
                fn1, fn2 = before_and_after_scatter(
                        rec, modelspec, idx, smoothing_bins=200,
                        mod_name='log_sig'
                        )
                plots = ([fn1, fn2], [1, 1])
                plot_fns.append(plots)

            elif 'nonlinearity.tanh' in fname:
                fn1, fn2 = before_and_after_scatter(
                        rec, modelspec, idx, smoothing_bins=200,
                        mod_name='tanh'
                        )
                plots = ([fn1, fn2], [1, 1])
                plot_fns.append(plots)

            elif 'nonlinearity.dlog' in fname:
                # SVD removed plotting here. breaks if applied to multi-
                # channel input, as when used for compression of spectrogram
                pass

            else:
                # Unrecognized nonlinearity
                pass

        elif 'signal_mod' in fname:
            if 'signal_mod.make_state_signal' in fname:
                # TODO
                pass
            elif 'signal_mod.average_sig' in fname:
                pass
            else:
                pass

        elif 'state' in fname:
            if 'state.state_dc_gain' in fname:
                fn1 = partial(state_vars_timeseries, rec, modelspec)
                fns = get_state_vars_psths(rec, epoch, psth_name='resp',
                                           occurrence=occurrence)
                plot1 = (fn1, 1)
                plot2 = (fns, [1]*len(fns))
                plot_fns.extend([plot1, plot2])
            else:
                pass

    return plot_fns


def quickplot_no_xforms(rec, est, val, modelspecs, default='val', occurrence=0,
                        epoch='TRIAL', figsize=None, height_mult=3.0, m_idx=0):
    """Compatibility wrapper for quickplot."""
    ctx = {'rec': rec, 'est': est, 'val': val, 'modelspecs': modelspecs}
    return quickplot(ctx, default=default, epoch=epoch, occurrence=occurrence,
                     figsize=figsize, height_mult=height_mult, m_idx=m_idx)


# TODO: maybe a better place to put these? but the functionality of returning
#       partial plots is pretty specific to quickplot/summary

def before_and_after_signal(rec, modelspec, idx, sig_name='pred'):
    # HACK: shouldn't hardcode 'stim', might be named something else
    #       or not present at all. Need to figure out a better solution
    #       for special case of idx = 0
    if idx == 0:
        # Can't have anything before index 0, so use input stimulus
        before = rec
        before_sig = copy.deepcopy(rec['stim'])
    else:
        before = ms.evaluate(rec, modelspec, start=None, stop=idx)
        before_sig = copy.deepcopy(before[sig_name])

    before_sig.name = 'before'

    # now evaluate next module step
    after = ms.evaluate(before.copy(), modelspec, start=idx, stop=idx+1)
    after_sig = copy.deepcopy(after[sig_name])
    after_sig.name = 'after'

    return before_sig, after_sig


def before_and_after_psth(rec, modelspec, idx, sig_name='pred',
                          epoch='TRIAL', occurrences=0, channels=0,
                          mod_name='Unknown'):

    before_sig, after_sig = before_and_after_signal(rec, modelspec, idx,
                                                    sig_name)
    signals = [before_sig, after_sig]
    fn = partial(timeseries_from_epoch, signals, epoch,
                 occurrences=occurrences, channels=channels, xlabel='Time',
                 ylabel='Value', title=mod_name)
    return fn


def before_and_after_scatter(rec, modelspec, idx, sig_name='pred',
                             compare='resp', smoothing_bins=False,
                             mod_name='Unknown', xlabel1=None, xlabel2=None,
                             ylabel1=None, ylabel2=None):

    # HACK: shouldn't hardcode 'stim', might be named something else
    #       or not present at all. Need to figure out a better solution
    #       for special case of idx = 0
    if idx == 0:
        # Can't have anything before index 0, so use input stimulus
        before = rec
        before_sig = rec['stim']
        before.name = '**stim'
    else:
        before = ms.evaluate(rec, modelspec, start=None, stop=idx)
        before_sig = before[sig_name]

    # compute correlation for pre-module before it's over-written
    corr1 = nm.corrcoef(before, pred_name=sig_name, resp_name=compare)

    # now evaluate next module step
    after = ms.evaluate(before.copy(), modelspec, start=idx, stop=idx+1)
    after_sig = after[sig_name]
    corr2 = nm.corrcoef(after, pred_name=sig_name, resp_name=compare)

    compare_to = rec[compare]
    title1 = '{} vs {} before {}'.format(sig_name, compare, mod_name)
    title2 = '{} vs {} after {}'.format(sig_name, compare, mod_name)
    # TODO: These are coming out the same, but that seems unlikely
    text1 = "r = {0:.5f}".format(corr1)
    text2 = "r = {0:.5f}".format(corr2)

    fn1 = partial(plot_scatter, before_sig, compare_to, title=title1,
                  smoothing_bins=smoothing_bins, xlabel=xlabel1,
                  ylabel=ylabel1, text=text1)
    fn2 = partial(plot_scatter, after_sig, compare_to, title=title2,
                  smoothing_bins=smoothing_bins, xlabel=xlabel2,
                  ylabel=ylabel2, text=text2)

    return fn1, fn2


def get_state_vars_psths(rec, epoch, psth_name='resp', occurrence=0):
    var_list = rec['state'].chans
    psth_list = [
            partial(state_var_psth_from_epoch, rec, epoch, psth_name=psth_name,
                    var_name=var, occurrence=occurrence)
            for var in var_list
            ]
    return psth_list
