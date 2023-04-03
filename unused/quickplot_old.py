import importlib
import logging
import copy

import matplotlib.pyplot as plt

import nems0.utils
import nems0.modelspec as ms
from nems0.plots.heatmap import (weight_channels_heatmap, fir_heatmap,
                                strf_heatmap)
from nems0.plots.scatter import plot_scatter
from nems0.plots.spectrogram import spectrogram_from_epoch
from nems0.plots.timeseries import (timeseries_from_epoch,
                                   timeseries_from_signals)
from nems0.plots.histogram import pred_error_hist

log = logging.getLogger(__name__)

# TODO: work in progress

# TODO: 'pred' and 'resp' are hard-coded into a lot of plot stuff,
#       which isn't great. should modelspec[0]['meta'] specify
#       pred and resp name? i and o solved a similar problem
#       for modules so right answer is likely somethign along
#       the same lines.

# NOTE: If you are adding a new module to the plot defaults,
#       and are unsure what plot to use, the value can be
#       left as None. In this case, the quick_plot function
#       will use whichever plot-type is currently chosen
#       as the fallback (currently timeseries.before_and_after)
#       -jacob 3/16/2018

_FALLBACK = ['nems.plots.quickplot.before_and_after',
             {'sig_name': 'pred', 'xlabel': 'Time', 'ylabel': 'Firing Rate',
              'channels': 0}]

# Entries in defaults should have the form:
#     'module_fn_name': ['plot_fn_name', {arg1=value1, arg2=value2,...}]
_DEFAULTS = {
        'nems.modules.fir.basic': [
                'nems.plots.quickplot.fir_heatmap_quick',
                {'clim': None, 'title': 'FIR heatmap'}
                ],
        'nems.modules.weight_channels.gaussian': [
                'nems.plots.quickplot.wc_heatmap_quick',
                {'clim': None, 'title': 'Weight Channels Gaussian heatmap'}
                ],
        'nems.modules.weight_channels.basic': [
                'nems.plots.quickplot.wc_heatmap_quick',
                {'clim': None, 'title': 'Weight Channels Basic heatmap'}
                ],
        'nems.modules.levelshift.levelshift': None,
        'nems.modules.nonlinearity.double_exponential': None,
        'nems.modules.nonlinearity.quick_sigmoid': None,
        'nems.modules.nonlinearity.logistic_sigmoid': None,
        'nems.modules.nonlinearity.tanh': None,
        'nems.modules.nonlinearity.dlog': None,
        'nems.modules.signal_mod.make_state_signal': None,
        'nems.modules.state.state_dc_gain': None,
        'nems.modules.signal_mod.average_sig': None
        }

# These will all be included after the plots specified by _DEFAULTS,
# regardless of what's included in the modelspec.
# Make sure these are going to work with any kind of model!

# TODO: Current pred vs act is good for LBHB but others might not
#       use pred and resp signal names.
_PERFORMANCE = [
        ['nems.plots.quickplot.pred_resp_scatter',
         {'smoothing_bins': False, 'title': 'Prediction versus Response',
          'pred_name': 'pred', 'resp_name': 'resp'}],
        ['nems.plots.quickplot.error_hist', {}],
        ]

# copied from nems/modelspec for now
lookup_table = {}  # TODO: Replace with real memoization/joblib later


def _lookup_fn_at(fn_path):
    '''
    Private function that returns a function handle found at a
    given module. Basically, a way to import a single function.
    e.g.
        myfn = _lookup_fn_at('nems.modules.fir.fir_filter')
        myfn(data)
        ...
    '''
    if fn_path in lookup_table:
        fn = lookup_table[fn_path]
    else:
        api, fn_name = nems.utils.split_to_api_and_fn(fn_path)
        api_obj = importlib.import_module(api)
        fn = getattr(api_obj, fn_name)
        lookup_table[fn_path] = fn
    return fn


def quickplot(rec, modelspec):
    '''
    Generates plots on a per-module basis as specified
    by 'defaults,' using only plot functions that take
    rec and modelspecs as the first two positional arguments.

    'idx' and 'ax' will additionally be passed to each plot
    function as keyword arguments.
    '''
    # TODO: add in special checks for plots like strf_heatmap
    #       that depend on multiple modules?
    n = len(modelspec)
    o = len(_PERFORMANCE)
    # expand rows and height based on number of modules
    fig, axes = plt.subplots(n+o, 1, figsize=(12, (n+o)*4))
    for idx, m, ax in zip(list(range(n)), modelspec, axes[:n]):
        if m['fn'] not in _DEFAULTS.keys():
            log.warn("No default plot type set for: {}\n"
                     "Using fallback plot: {}"
                     .format(m['fn'], _FALLBACK))
            _set_to_fallback(m['fn'])
        if not _verify_default(m['fn']):
            log.warn("Invalid plot specified for: {}\n"
                     "Got: {}\n Changed to fallback: {}"
                     .format(m['fn'], _DEFAULTS[m['fn']], _FALLBACK))
            _set_to_fallback(m['fn'])
        plot_fn_def, plot_args = _DEFAULTS[m['fn']]
        plot_args['ax'] = ax
        plot_args['idx'] = idx
        plot_fn = _lookup_fn_at(plot_fn_def)
        plt.sca(ax)
        plot_fn(rec, modelspec, **plot_args)

    for idx, p, ax in zip(list(range(o)), _PERFORMANCE, axes[n:]):
        plot_fn_def, plot_args = p
        plot_args['ax'] = ax
        plot_args['idx'] = idx
        plot_fn = _lookup_fn_at(plot_fn_def)
        plt.sca(ax)
        plot_fn(rec, modelspec, **plot_args)

    fig.tight_layout(pad=1.5, w_pad=1.0, h_pad=2.5)
    return fig


def _verify_default(key):
    '''
    Checks if the key and its corresponding entry in _DEFAULTS
    conforms to expectations. Mostly just to provide more
    helpful information for debugging.
    '''
    try:
        if key not in _DEFAULTS.keys():
            log.warn("Key not in _DEFAULTS")
            return False
        entry = _DEFAULTS[key]
        if not entry:
            log.warn("Entry was empty for key: {}".format(key))
            return False
        if len(entry) != 2:
            log.warn("Key is not length 2, got: {}".format(len(entry)))
            return False
        if not isinstance(entry[0], str):
            log.warn("First entry was not a string, got: {}".format(entry[0]))
            return False
        if not isinstance(entry[1], dict):
            log.warn("Second entry was not a dict, got: {}".format(entry[1]))
            return False
    except Exception as e:
        log.warn("Uncaught exception while verifying default plot for: {}"
                 .format(key))
        log.exception(e)
        return False

    return True


def _set_to_fallback(key):
    # TODO: take out title stuff? adds a somewhat-hidden
    #       additional kwarg that could cause problems if
    #       plot function doens't allow title argument.
    #       Alternatively, require plot functions to allow
    #       title argument - not totally unreasonable?
    entry = copy.deepcopy(_FALLBACK)
    if 'title' not in entry[1].keys():
        entry[1]['title'] = ('Plot: {} for Module: {}'
                             .format(entry[0], key))
    _DEFAULTS[key] = entry


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
