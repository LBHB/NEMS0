import re
import os
import copy
import json
import importlib
import logging
import numpy as np
import scipy.stats as st
import nems.utils
import nems.uri
from nems.fitters.mappers import simple_vector
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)

# Functions for saving, loading, and evaluating modelspecs

# TODO: In retrospect, this should have been a class, just like Recording.
#       Refactoring would not be too hard and would shorten many of these
#       function names. If you do so, see /docs/planning/models.py and
#       bring the ideas into this file, then delete it from docs/planning.

class ModelSpec:
    """
    Defines a model based on a NEMS modelspec.

    Attributes
    ----------
    raw : nested list of dictionaries
        Equivalent of old NEMS modelspec.
        TODO check if/how multiple cells supported
        TODO convert raw to numpy array of dicts
        The first level is a list of cells, each of which is a list of
        lists. (IS THIS SUPPORTED?)
        The second level is a list of fits, each of which is a list of
        dictionaries.
        The third level is a list of jackknifes, each of which is a list of
        dictionaries.
        Each dictionary specifies a module, or one step in the model.
        For example,
    fit_index : int
        Integer index of which fit to reference when multiple are present,
        default 0, when jacknifing et cetera.
    cell_index : int
        Integer index of which "cell" to reference when multiple are present,
        default 0.
    fast_eval : testing (default false)

    """

    def __init__(self, raw=None, phis=None, fit_index=0, cell_index=0,
                 jack_index=0, recording=None):

        if raw is None:
            # one model, no modules
            raw = np.full((1, 1, 1), None)
            raw[0, 0, 0] = []
        elif type(raw) is list:
            if type(raw[0]) is not list:
                raw=[raw]
            # compatible with load_modelspec -- read in list of lists
            # make raw (1, fit_count, 1) array of lists
            # TODO default is to make single list into jack_counts!
            r = np.full((1,1,len(raw)), None)
            for i, _r in enumerate(raw):
                r[0,0,i]=_r
            raw = r

        # otherwise, assume raw is a properly formatted 3D array (cell X fit X jack)
        self.raw = raw
        self.phis = [] if phis is None else phis

        # a Model can have multiple fits, each of which contains a different
        # set of phi values. fit_index re
        # references the default phi
        self.fit_index = fit_index
        self.jack_index = jack_index
        self.cell_index = cell_index
        self.mod_index = 0
        self.plot_epoch = 'REFERENCE'
        self.plot_occurrence = 0
        self.plot_channel = 0   # channel of pred/resp to plot in timeseries (for population models)
        self.recording = recording  # default recording for evaluation & plotting
        self.fast_eval = False
        self.fast_eval_start = 0
        self.freeze_rec = None

    #
    # overloaded methods
    #
    def __getitem__(self, key):
        try:
            return self.get_module(key)
        except:
            if type(key) is slice:
                return [self.raw[self.cell_index, self.fit_index, self.jack_index][ii]
                        for ii in range(*key.indices(len(self)))]
            elif key == 'meta':
                return self.meta
            elif key == 'phi':
                return self.phi()
            else:
                raise ValueError('key {} not supported'.format(key))

    def __setitem__(self, key, val):
        try:
            # Try converting types like np.int64 instead of just
            # throwing an error.
            self.raw[self.cell_index, self.fit_index, self.jack_index][int(key)] = val
        except ValueError:
            raise ValueError('key {} not supported'.format(key))
        return self

    def __iter__(self):
        self.mod_index = -1
        return self

    # TODO: Something funny is going on when iterating over modules directly
    #       using these methods. The last couple modules were being excluded.
    #       Temp fix: use .modules instead and then iterate over the list
    #       returned by that.
    def __next__(self):
        if self.mod_index < len(self.raw[self.cell_index, self.fit_index, self.jack_index])-1:
            self.mod_index += 1
            return self.get_module(self.mod_index)
        else:
            raise StopIteration

    def __repr__(self):
        return repr(self.raw)

    def __str__(self):
        x = [m['fn'] for m in self.raw[self.cell_index, self.fit_index, self.jack_index]]
        return "\n".join(x)

    def __len__(self):
        return len(self.raw[self.cell_index, self.fit_index, self.jack_index])

    def copy(self, lb=None, ub=None, fit_index=None, jack_index=None):
        """
        :param lb: start module (default 0) -- doesn't work
        :param ub: stop module (default -1) -- doesn't work
        :return: A deep copy of the modelspec (subset of modules if specified)
        """
        raw = copy.deepcopy(self.raw)
        meta_save = copy.deepcopy(self.meta)

        if fit_index is not None:
            raw = raw[:, fit_index:(fit_index+1), :]
            _f = 0
        else:
            _f = self.fit_index
        if jack_index is not None:
            raw = raw[:, :, jack_index:(jack_index + 1)]
            _j = 0
        else:
            _j = self.jack_index

        for r in self.raw.flatten():
            r[0]['meta'] = meta_save

        m = ModelSpec(raw, fit_index=_f, cell_index=self.cell_index,
                      jack_index=_j)
        return m

    #
    # module control/listing -- act like a list of dictionaries
    #
    def get_module(self, mod_index=None):
        """
        :param mod_index: index of module to return
        :return: single module from current fit_index (doesn't create a copy!)
        """
        if mod_index is None:
            mod_index = self.mod_index
        return self.raw[self.cell_index, self.fit_index, self.jack_index][mod_index]

    def drop_module(self, mod_index=None, in_place=False):
        if mod_index is None:
            mod_index = self.mod_index

        if in_place:
            for fit in np.flatten(self.raw):
                del fit[mod_index]
            return None

        else:
            raw_copy = copy.deepcopy(self.raw)
            for fit in np.flatten(raw_copy):
                del fit[mod_index]
            new_spec = ModelSpec(raw_copy)
            new_spec.recording = self.recording
            return new_spec

    @property
    def modules(self):
        return self.raw[self.cell_index, self.fit_index, self.jack_index]

    #
    # fit/jackknife control
    #

    # TODO support for multiple recording views/modelspec jackknifes (jack_count>0)
    #  and multiple fits (fit_count>0)

    def tile_fits(self, fit_count=0):
        """
        create <fit_count> sets of fit parameters to allow for multiple fits,
        useful for n-fold cross validation or starting from multiple intial
        conditions. values of each phi are copied from the existing first
        value.
        Applied in-place.
        """
        meta_save = self.meta

        fits = [copy.deepcopy(self.raw[:, 0:1, :]) for i in range(fit_count)]
        self.raw = np.concatenate(fits, axis=1)

        for r in self.raw.flatten():
            r[0]['meta'] = meta_save

        return self

    def tile_jacks(self, jack_count=0):
        """
        create <jack_count> sets of fit parameters to allow for multiple jackknifes,
        useful for n-fold cross validation or starting from multiple intial
        conditions. values of each phi are copied from the existing first
        value.
        Applied in-place.
        """
        meta_save = self.meta

        jacks = [copy.deepcopy(self.raw[:, :, 0:1]) for i in range(jack_count)]
        self.raw = np.concatenate(jacks, axis=2)

        for r in self.raw.flatten():
            r[0]['meta'] = meta_save

        return self

    @property
    def cell_count(self):
        """Number of cells (sets of phi values) in this modelspec"""
        return self.raw.shape[0]

    @property
    def fit_count(self):
        """Number of fits (sets of phi values) in this modelspec"""
        return self.raw.shape[1]

    @property
    def jack_count(self):
        """Number of jackknifes (sets of phi values) in this modelspec"""
        return self.raw.shape[2]

    def set_cell(self, cell_index=None):
        """return self with cell_index set to specified value"""
        if cell_index is not None:
            self.cell_index = cell_index

    def set_fit(self, fit_index):
        """return self with fit_index set to specified value"""
        if fit_index is None:
            pass
        elif fit_index > (self.fit_count - 1):
            raise ValueError('fit_index greater than fit_count-1')
        elif fit_index < self.fit_count*-1:
            raise ValueError('negative fit_index smaller than fit_count')
        else:
            self.fit_index = fit_index

        return self

    def set_jack(self, jack_index=None):
        """return self with jack_index (current jackknife #) set to specified value"""
        if jack_index is not None:
            self.jack_index = jack_index

        return self

    def fits(self):
        """List of modelspecs, one for each fit, for compatibility with some
           old functions"""
        return [ModelSpec(self.raw, jack_index=f)
                for f in range(self.jack_count)]

    #
    # metadata
    #
    @property
    def meta(self):
        if self.raw[0,0,0][0].get('meta') is None:
            self.raw[0,0,0][0]['meta'] = {}
        return self.raw[0,0,0][0]['meta']

    @property
    def modelspecname(self):
        return '-'.join([m.get('id', 'BLANKID') for m in self.modules])

    def fn(self):
        return [m['fn'] for m in self.raw[self.cell_index, self.fit_index, self.jack_index]]

    #
    # parameter info
    #
    @property
    def phi(self, cell_index=None, fit_index=None, jack_index=None, mod_idx=None):
        """
        :param fit_index: which model fit to use (default use self.fit_index
        :return: list of phi dictionaries, or None for modules with no phi
        """
        if fit_index is None:
            fit_index = self.fit_index
        if mod_idx is None:
            return [m.get('phi') for m in self.raw[self.cell_index, fit_index, self.jack_index]]
        else:
            return self.raw[self.cell_index, fit_index, self.jack_index][mod_idx].get('phi')

    @property
    def phi_mean(self, mod_idx=None):
        """
        Returns mean of phi across fit_indexes
        :param mod_idx: which module to use (default all modules)
        :return: list of phi dictionaries, mean of each value
        """
        if len(self.raw) == 1:
            return self.phi(mod_idx=mod_idx)

        phi = []
        if type(mod_idx) is list:
            mod_range = mod_idx
        elif mod_idx is not None:
            mod_range = [mod_idx]
        else:
            mod_range = range(len(self.raw[0]))

        for mod_idx in mod_range:
            p = {}
            for k in self.raw[0][mod_idx]['phi'].keys():
                maxdim = len(self.raw[0][mod_idx]['phi'][k].shape)
                p[k] = np.mean(np.concatenate([np.expand_dims(f[mod_idx]['phi'][k], axis=maxdim)
                                               for f in self.raw], axis=maxdim), axis=maxdim, keepdims=False)
            phi.append(p)

        return phi

    @property
    def phi_sem(self, mod_idx=None):
        """
        Returns SEM of phi across fit_indexes
        :param mod_idx: which module to use (default all modules)
        :return: list of phi dictionaries, jackknife sem of each value
        """
        fit_count = len(self.raw)
        if fit_count == 1:
            return self.phi(mod_idx=mod_idx)

        phi = []
        if type(mod_idx) is list:
            mod_range = mod_idx
        elif mod_idx is not None:
            mod_range = [mod_idx]
        else:
            mod_range = range(len(self.raw[0]))

        for mod_idx in mod_range:
            p = {}
            for k in self.raw[0][mod_idx]['phi'].keys():
                maxdim = len(self.raw[0][mod_idx]['phi'][k].shape)
                p[k] = np.std(np.concatenate([np.expand_dims(f[mod_idx]['phi'][k], axis=maxdim)
                                              for f in self.raw], axis=maxdim), axis=maxdim, keepdims=False) * \
                    np.sqrt(fit_count-1)
            phi.append(p)

        return phi

    @property
    def phi_vector(self, fit_index=None):
        """
        :param fit_index: which model fit to use (default use self.fit_index
        :return: vector of phi values from all modules
        """
        if fit_index is None:
            fit_index = self.fit_index
        m = self.copy(fit_index)
        packer, unpacker, bounds = simple_vector(m)
        return packer(self)

    #
    # plotting support
    #
    def plot_fn(self, mod_index=None, plot_fn_idx=None, fit_index=None):
        """get function for plotting something about a module"""
        if mod_index is None:
            mod_index = self.mod_index
        if fit_index is not None:
            self.fit_index = fit_index

        module = self.get_module(mod_index)
        if plot_fn_idx is None:
            plot_fn_idx = module.get('plot_fn_idx', 0)
        try:
            fn_path = module.get('plot_fns')[plot_fn_idx]
        except:
            fn_path = 'nems.plots.timeseries.mod_output'

        return _lookup_fn_at(fn_path)

    def plot(self, mod_index=None, rec=None, ax=None, plot_fn_idx=None,
             fit_index=None, sig_name='pred', channels=None, **options):
        """generate plot for a single module"""

        if rec is None:
            rec = self.recording
        if channels is None:
            channels = self.plot_channel
        plot_fn = self.plot_fn(mod_index=mod_index, plot_fn_idx=plot_fn_idx,
                               fit_index=fit_index)
        plot_fn(rec=rec, modelspec=self, sig_name=sig_name, idx=mod_index,
                channels=channels, ax=ax, **options)

    def quickplot(self, rec=None):

        if rec is None:
            rec = self.recording
        fig = plt.figure()
        plot_count = len(self)
        for i in range(plot_count):
            ax = fig.add_subplot(plot_count, 1, i+1)
            self.plot(mod_index=i, rec=rec, ax=ax)
        return fig

    def append(self, module):
        self.raw[self.cell_index, self.fit_index, self.jack_index].append(module)

    def pop_module(self):
        del self.raw[self.cell_index, self.fit_index, self.jack_index][-1]

    def get_priors(self, data):
        # Here, we query each module for it's priors. A prior will be a
        # distribution, so we set phi to the mean (i.e., expected value) for
        # each parameter, then ask the module to evaluate the input data. By
        # doing so, we give each module an opportunity to perform sensible data
        # transforms that allow the following module to initialize its priors as
        # it sees fit.
        result = data.copy()
        priors = []
        for module in self.modules:
            module_priors = module.get_priors(result)
            priors.append(module_priors)

            phi = {k: p.mean() for k, p in module_priors.items()}
            module_output = module.evaluate(result, phi)
            result.update(module_output)

        return priors

    def evaluate(self, rec=None, start=None, stop=None):
        """
        Evaluate the Model on a recording.
        """
        if rec is None:
            rec = self.recording

        rec = evaluate(rec, self.raw[self.cell_index, self.fit_index, self.jack_index], start=start, stop=stop)

        return rec

    def fast_eval_on(self, rec=None, subset=None):
        """
        enter fast eval mode, where model is evaluated up through the
        first module that has a fittable phi. evaluate model on rec up through
        the preceeding module and save in self.freeze_rec
        """
        if rec is None:
            raise ValueError("Must provide valid rec=<recording> object")
        if subset is not None:
            start_mod = subset[0]
        else:
            start_mod = len(self)-1
            for i in range(len(self)-1,0,-1):
                if ('phi' in self[i]) and self[i]['phi']:
                    start_mod = i

        # eval from 0 to start position and save the result in freeze_rec
        self.fast_eval_start = 0
        self.freeze_rec = evaluate(rec, self, start=0, stop=start_mod)

        # then switch to fast_eval mode
        self.fast_eval = True
        self.fast_eval_start = start_mod
        log.info('Freezing fast rec at start=%d', self.fast_eval_start)

    def fast_eval_off(self):
        """ turn off fast_eval and purge freeze_rec to free up memory """
        self.fast_eval = False
        self.freeze_rec = None
        self.fast_eval_start = 0

    def generate_tensor(self, data, phi):
        '''
        Evaluate the module given the input data and phi

        Parameters
        ----------
        data : dictionary of arrays and/or tensors
        phi : list of dictionaries
            Each entry in the list maps to the corresponding module in the
            model. If a module does not require any input parameters, use a
            blank dictionary. All elements in phi must be scalars, arrays or
            tensors.

        Returns
        -------
        data : dictionary of Signals
            dictionary of arrays and/or tensors
        '''
        # Loop through each module in the stack and transform the data.
        result = data.copy()
        for module, module_phi in zip(self.modules, phi):
            module_output = module.generate_tensor(result, module_phi)
            result.update(module_output)
        return result

    def get_shortname(self):
        '''
        Returns a string that is just the module ids in this modelspec.
        '''
        keyword_string = '_'.join([m['id'] for m in self])
        return keyword_string

    def get_longname(self):
        '''
        Returns a LONG name for this modelspec suitable for use in saving to disk
        without a path.
        '''
        meta = self.meta

        recording_name = meta.get('exptid')
        if recording_name is None:
            recording_name = meta.get('recording', 'unknown_recording')
        if 'modelspecname' in self.meta:
            keyword_string = self.meta['modelspecname']
        else:
            keyword_string = get_modelspec_shortname(self)
        fitter_name = meta.get('fitkey', meta.get('fitter', 'unknown_fitter'))
        date = nems.utils.iso8601_datestring()
        guess = '.'.join([recording_name, keyword_string, fitter_name, date])

        # remove problematic characters
        guess = re.sub('[:]', '', guess)
        guess = re.sub('[,]', '', guess)

        return guess


def get_modelspec_metadata(modelspec):
    '''
    Returns a dict of the metadata for this modelspec. Purely by convention,
    metadata info for the entire modelspec is stored in the first module.
    '''
    return modelspec.meta


def set_modelspec_metadata(modelspec, key, value):
    '''
    Sets a key/value pair in the modelspec's metadata. Purely by convention,
    metadata info for the entire modelspec is stored in the first module.
    '''
    if not modelspec.meta:
        modelspec[0]['meta'] = {}
    modelspec[0]['meta'][key] = value
    return modelspec


def get_modelspec_shortname(modelspec):
    '''
    Returns a string that is just the module ids in this modelspec.
    '''
    return modelspec.get_shortname()


def get_modelspec_longname(modelspec):
    '''
    Returns a LONG name for this modelspec suitable for use in saving to disk
    without a path.
    '''
    return modelspec.get_longname()


def _modelspec_filename(basepath, number):
    suffix = '.{:04d}.json'.format(number)
    return (basepath + suffix)


def save_modelspec(modelspec, filepath):
    '''
    Saves a modelspec to filepath. Overwrites any existing file.
    '''
    if type(modelspec) is list:
        nems.uri.save_resource(filepath, json=modelspec)
    else:
        nems.uri.save_resource(filepath, json=modelspec.raw)

def save_modelspecs(directory, modelspecs, basename=None):
    '''
    Saves one or more modelspecs to disk with stereotyped filenames:
        directory/basename.0000.json
        directory/basename.0001.json
        directory/basename.0002.json
        ...etc...
    Basename will be automatically generated if not provided.
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o777)

    for idx, modelspec in enumerate(modelspecs):
        if not basename:
            bname = get_modelspec_longname(modelspec)
        else:
            bname = basename
        basepath = os.path.join(directory, bname)
        filepath = _modelspec_filename(basepath, idx)
        if type(modelspec) is list:
            # TODO fix save with array
            save_modelspec(modelspec, filepath)
        else:
            # HACK for backwards compatibility. if saving a modelspecs list
            # then only need a single fit_index from the ModelSpec class
            save_modelspec(modelspec.raw[0,0,0], filepath)
    return filepath


def load_modelspec(uri):
    '''
    Returns a single modelspecs loaded from uri
    '''
    ms = nems.uri.load_resource(uri)
    return ModelSpec(ms)


def load_modelspecs(directory, basename, regex=None):
    '''
    Returns a list of modelspecs loaded from directory/basename.*.json
    '''
    #regex = '^' + basename + '\.{\d+}\.json'
    # TODO: fnmatch is not matching pattern correctly, replacing
    #       with basic string matching for now.  -jacob 2/17/2018
    #files = fnmatch.filter(os.listdir(directory), regex)
    #       Also fnmatch was returning list of strings? But
    #       json.load expecting file object
    #modelspecs = [json.load(f) for f in files]
    dir_list = os.listdir(directory)
    if regex:
        # TODO: Not sure why this isn't working? No errors but
        #       still isn't matching the things it should be matching.
        #       ( tested w/ regex='^TAR010c-18-1\.{\d+}\.json')
        #       -jacob 2/25/18
        if isinstance(regex, str):
            regex = re.compile(regex)
        files = [os.path.join(directory, s) for s in dir_list
                 if re.search(regex, s)]
    else:
        files = [os.path.join(directory, s) for s in dir_list
                 if (basename in s and '.json' in s)]
    modelspecs = []
    for file in files:
        with open(file, 'r') as f:
            try:
                m = json.load(f)
                m[0]['meta']['filename']=file
            except json.JSONDecodeError as e:
                print("Couldn't load modelspec: {0}"
                      "Error: {1}".format(file, e))
            modelspecs.append(m)
    return ModelSpec(modelspecs)
    #return modelspecs


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

    # default is nems.xforms.<fn_path>
    if not '.' in fn_path:
        fn_path = 'nems.xforms.' + fn_path

    if fn_path in lookup_table:
        fn = lookup_table[fn_path]
    else:
        api, fn_name = nems.utils.split_to_api_and_fn(fn_path)
        api_obj = importlib.import_module(api)
        fn = getattr(api_obj, fn_name)
        lookup_table[fn_path] = fn
    return fn


def fit_mode_on(modelspec, rec=None, subset=None):
    '''
    turn no norm.recalc for each module when present
    '''
    """
    # norm functions deprecated. too messy
    for m in modelspec:
        if 'norm' in m.keys():
            m['norm']['recalc'] = 1
    """
    if rec is None:
        raise ValueError('rec must be specified')
    modelspec.fast_eval_on(rec, subset)


def fit_mode_off(modelspec):
    '''
    turn off norm.recalc for each module when present
    '''
    """
    # norm functions deprecated. too messy
    for m in modelspec:
        if 'norm' in m.keys():
            m['norm']['recalc'] = 0
    """
    modelspec.fast_eval_off()


def evaluate(rec, modelspec, start=None, stop=None):
    '''
    Given a recording object and a modelspec, return a prediction.
    Does not alter its arguments in any way.
    Only evaluates modules at indices start through stop-1.
    Note that a value of None for start will include the beginning
    of the list, and a value of None for stop will include the end
    of the list (whereas a value of -1 for stop will not).
    '''
    if modelspec.fast_eval:
        start = modelspec.fast_eval_start
        d = modelspec.freeze_rec.copy()
        #import pdb
        #pdb.set_trace()
    else:
        # d = copy.deepcopy(rec)  # Paranoid, but 100% safe
        d = rec.copy()  # About 10x faster & fine if Signals are immutable

    for m in modelspec[start:stop]:
        fn = _lookup_fn_at(m['fn'])
        fn_kwargs = m.get('fn_kwargs', {})
        phi = m.get('phi', {})
        kwargs = {**fn_kwargs, **phi}  # Merges both dicts
        new_signals = fn(rec=d, **kwargs)

        #if type(new_signals) is not list:
        #    raise ValueError('Fn did not return list of signals: {}'.format(m))

        # testing normalization
        """
        if 'norm' in m.keys():
            s = new_signals[0]
            k = s.name
            if m['norm']['recalc']:
                if m['norm']['type'] == 'minmax':
                    m['norm']['d'] = np.nanmin(s.as_continuous(), axis=1,
                                               keepdims=True)
                    m['norm']['g'] = np.nanmax(s.as_continuous(), axis=1,
                                               keepdims=True) - \
                        m['norm']['d']
                    m['norm']['g'][m['norm']['g'] <= 0] = 1
                elif m['norm']['type'] == 'none':
                    m['norm']['d'] = np.array([0])
                    m['norm']['g'] = np.array([1])
                else:
                    raise ValueError('norm format not supported')

            fn = lambda x: (x - m['norm']['d']) / m['norm']['g']
            new_signals = [s.transform(fn, k)]
        """

        for s in new_signals:
            d.add_signal(s)
        d.signal_views[d.view_idx] = d.signals

    return d


def summary_stats(modelspecs, mod_key='fn', meta_include=[], stats_keys=[]):
    '''
    Generates summary statistics for a list of modelspecs.
    Each modelspec must be of the same length and contain the same
    modules (though they need not be in the same order).

    For example, ten modelspecs composed of the same modules that
    were fit to ten different datasets can be compared. However, ten
    modelspecs all with different modules fit to the same data cannot
    be compared because there is no guarantee that they contain
    comparable parameter values.

    Arguments:
    ----------
    modelspecs : list of modelspecs
        See docs/modelspecs.md

    Returns:
    --------
    stats : nested dictionary
        {'module.function---parameter':
            {'mean':M, 'std':S, 'values':[v1,v2 ...]}}
        Where M, S and v might be scalars or arrays depending on the
        typical type for the parameter.
    '''
    # Make sure the modelspecs themselves aren't modified
    # deepcopy for nested container structure
    modelspecs = [copy.deepcopy(m) for m in modelspecs]

    # Modelspecs must have the same length to compare
    length = None
    for m in modelspecs:
        if length:
            if len(m) != length:
                raise ValueError("All modelspecs must have the same length")
        length = len(m)

    # Modelspecs must have the same modules to compare
    fns = [m['fn'] for m in modelspecs[0]]
    for mspec in modelspecs[1:]:
        m_fns = [m['fn'] for m in mspec]
        if not sorted(fns) == sorted(m_fns):
            raise ValueError("All modelspecs must have the same modules")

    # Create a dictionary with a key for each parameter associated with
    # to a list of one value per modelspec
    columns = {}
    for mspec in modelspecs:
        for i, m in enumerate(mspec):
            name = '%d--%s' % (i, m[mod_key]) if mod_key else str(i)
            # Abbreviate by default if using 'fn'
            if name.startswith('nems.modules.'):
                name = name[13:]

            # Add information from first-module meta
            if i == 0:
                meta = m['meta']
                meta_keys = [k for k in meta.keys() if k in meta_include]
                for k in meta_keys:
                    column_entry = 'meta--%s' % (k)
                    if column_entry in columns.keys():
                        columns[column_entry].append(meta[k])
                    else:
                        columns.update({column_entry: [meta[k]]})

            # Add in phi values
            phi = m['phi']
            params = phi.keys()
            for p in params:
                column_entry = '%s--%s' % (name, p)
                if column_entry in columns.keys():
                    columns[column_entry].append(phi[p])
                else:
                    columns.update({column_entry: [phi[p]]})


    # Convert entries from lists of values to dictionaries
    # containing keys for mean, std and the raw values.
    with_stats = {}

    for col, values in columns.items():
        try:
            mean = try_scalar((np.mean(values, axis=0)))
            std = try_scalar((np.std(values, axis=0)))
            sem = try_scalar((st.sem(values, axis=0)))
            max = try_scalar((np.max(values, axis=0)))
            min = try_scalar((np.min(values, axis=0)))
        except:
            mean = np.nan
            std = np.nan
            sem = np.nan
            max = np.nan
            min = np.nan
        values = try_scalar((np.array(values)))

        with_stats[col] = {
                'mean': mean, 'std': std, 'sem': sem, 'max': max, 'min': min,
                'values': values
                }

    return with_stats


def get_best_modelspec(modelspecs, metakey='r_test', comparison='greatest'):
    '''
    Examines the first-module meta information within each modelspec in a list,
    and returns a singleton list containing the modelspec with the greatest
    value for the specified metakey by default (or the least value optionally).
    '''
    idx = None
    best = None
    for i, m in enumerate(modelspecs):
        if metakey in m[0]['meta']:
            metaval = m[0]['meta'][metakey]
            if comparison == 'greatest':
                if best is None:
                    best = metaval
                    idx = i
                else:
                    if metaval > best:
                        best = metaval
                        idx = i

            elif comparison == 'least':
                if best is None:
                    best = metaval
                    idx = i
                else:
                    if metaval < best:
                        best = metaval
                        idx = i

            else:
                raise NotImplementedError("Only supports 'greatest' or 'least'"
                                          "as arguments for comparison")

    return [modelspecs[idx]]


def sort_modelspecs(modelspecs, metakey='r_test', order='descending'):
    '''
    Sorts modelspecs in order of the given metakey, which should be in
    the first-module meta entry of each modelspec.
    '''
    find_meta = lambda m: m[0]['meta'][metakey]
    sort = sorted(modelspecs, key=find_meta)
    if order.lower() in ['ascending', 'asc', 'a']:
        return sort
    elif order.lower() in ['descending', 'desc', 'd']:
        return list(reversed(sort))
    else:
        raise ValueError("Not a recognized sorting order: %s" % order)


def try_scalar(x):
    """Try to convert x to scalar, in case of ValueError just return x."""
    # TODO: Maybe move this to an appropriate utilities module?
    try:
        x = np.asscalar(x)
    except ValueError:
        pass
    return x


# TODO: Check that the word 'phi' is not used in fn_kwargs
# TODO: Error checking the modelspec before execution;
# TODO: Validation of modules json schema; all require args should be present
