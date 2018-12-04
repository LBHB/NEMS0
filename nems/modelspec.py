import re
import os
import copy
import json
import importlib
import numpy as np
import scipy.stats as st
import nems.utils
import nems.uri

# Functions for saving, loading, and evaluating modelspecs

# TODO: In retrospect, this should have been a class, just like Recording.
#       Refactoring would not be too hard and would shorten many of these
#       function names. If you do so, see /docs/planning/models.py and
#       bring the ideas into this file, then delete it from docs/planning.

class ModelSpec:
    """
    key attribute: raw --- equivalent of old model spec
    is raw everything you need to specify the model? Or should we un-embed meta?
    """
    def __init__(self, raw=None, fit_index=None):
        if raw is None:
            self.raw = [[]]
        else:
            self.raw = raw

        # a Model can have multiple fits, each of which contains a different
        # set of phi values. fit_index references the default phi
        if fit_index is None:
            self.fit_index = 0
        else:
            self.fit_index = fit_index
        self.mod_index = 0

    def __getitem__(self, key):
        try:
            return self.get_module(key)
        except:
            if type(key) is slice:
                return [self.raw[self.fit_index][ii] for ii in range(*key.indices(len(self)))]
            elif key=='meta':
                return self.meta()
            elif key=='phi':
                return self.phi()
            else:
                raise ValueError('key {} not supported'.format(key))

    def __setitem__(self, key, val):
        if type(key) is int:
            self.raw[self.fit_index][key] = val
        else:
            raise ValueError('key {} not supported'.format(key))
        return self

    def __iter__(self):
        self.mod_index = -1
        return self

    def __next__(self):
        if self.mod_index < len(self.raw[self.fit_index])-1:
            self.mod_index += 1
            return self.get_module(self.mod_index)
        else:
            raise StopIteration

    def __repr__(self):
        return repr(self.raw)

    def __str__(self):
        x = [m['fn'] for m in self.raw[0]]
        return "\n".join(x)

    def __len__(self):
        return len(self.raw[0])

    def get_module(self, index):
        """
        :param index:
        :return: single module from current fit_index (doesn't create a copy!)
        """
        return self.raw[self.fit_index][index]

    def copy(self, lb=None, ub=None):
        """
        :param lb: start module (default 0)
        :param ub: stop module (default -1)
        :return: A deep copy of the modelspec (subset of modules if specified)
        """
        raw = [copy.deepcopy(m[lb:ub]) for m in self.raw]
        return ModelSpec(raw)

    def fit_count(self):
        """Number of fits in this modelspec"""
        return len(self.raw)

    def get_fit(self, fit_index=None):
        """return copy, fit_index set to specified value"""
        m = self.copy()

        if fit_index is not None:
            m.fit_index = fit_index

        return m

    def fits(self):
        """List of modelspecs, one for each fit, for compatibility with some
           old functions"""
        m_list = []
        for f in range(len(self.raw)):
            m_list += [ModelSpec(self.raw, f)]

        return m_list

    def meta(self):
        return self.raw[0][0]['meta']

    def fn(self):
        return [m['fn'] for m in self.raw[0]]

    def phi(self, fit_index=None):
        """
        :param fit_index: which model fit to use (default use self.fit_index
        :return: list of phi dictionaries, or None for modules with no phi
        """
        if fit_index is None:
            fit_index = self.fit_index
        return [m.get('phi') for m in self.raw[fit_index]]

    def append(self, module):
        self.raw[0].append(module)

    def tile_fits(self, fit_count):
        self.raw = [self.raw[0].copy() for i in range(fit_count)]
        return self

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

    def evaluate(self, rec, start=None, stop=None):
        """
        Evaluate the Model on a recording.
        """
        rec = evaluate(rec, self.raw[self.fit_index], start=start, stop=stop)

        return rec

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


def get_modelspec_metadata(modelspec):
    '''
    Returns a dict of the metadata for this modelspec. Purely by convention,
    metadata info for the entire modelspec is stored in the first module.
    '''
    return modelspec[0].get('meta', {})


def set_modelspec_metadata(modelspec, key, value):
    '''
    Sets a key/value pair in the modelspec's metadata. Purely by convention,
    metadata info for the entire modelspec is stored in the first module.
    '''
    if not modelspec[0].get('meta'):
        modelspec[0]['meta'] = {}
    modelspec[0]['meta'][key] = value
    return modelspec


def get_modelspec_shortname(modelspec):
    '''
    Returns a string that is just the module ids in this modelspec.
    '''
    keyword_string = '_'.join([m['id'] for m in modelspec])
    return keyword_string


def get_modelspec_longname(modelspec):
    '''
    Returns a LONG name for this modelspec suitable for use in saving to disk
    without a path.
    '''
    meta = get_modelspec_metadata(modelspec)

    recording_name = meta.get('exptid')
    if recording_name is None:
        recording_name = meta.get('recording', 'unknown_recording')

    keyword_string = get_modelspec_shortname(modelspec)
    fitter_name = meta.get('fitkey', meta.get('fitter', 'unknown_fitter'))
    date = nems.utils.iso8601_datestring()
    guess = '.'.join([recording_name, keyword_string, fitter_name, date])

    guess = re.sub('[:]', '', guess)
    guess = re.sub('[,]', '', guess)

    return guess


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
            save_modelspec(modelspec, filepath)
        else:
            # HACK for backwards compatibility. if saving a modelspecs list
            # then only need a single fit_index from the ModelSpec class
            save_modelspec(modelspec.raw[0], filepath)
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


def fit_mode_on(modelspec):
    '''
    turn no norm.recalc for each module when present
    '''
    for m in modelspec:
        if 'norm' in m.keys():
            m['norm']['recalc'] = 1


def fit_mode_off(modelspec):
    '''
    turn off norm.recalc for each module when present
    '''
    for m in modelspec:
        if 'norm' in m.keys():
            m['norm']['recalc'] = 0


def evaluate(rec, modelspec, start=None, stop=None):
    '''
    Given a recording object and a modelspec, return a prediction.
    Does not alter its arguments in any way.
    Only evaluates modules at indices start through stop-1.
    Note that a value of None for start will include the beginning
    of the list, and a value of None for stop will include the end
    of the list (whereas a value of -1 for stop will not).
    '''
    # d = copy.deepcopy(rec)  # Paranoid, but 100% safe
    d = rec.copy()  # About 10x faster & fine if Signals are immutable
    for m in modelspec[start:stop]:
        fn = _lookup_fn_at(m['fn'])
        fn_kwargs = m.get('fn_kwargs', {})
        kwargs = {**fn_kwargs, **m['phi']}  # Merges both dicts
        new_signals = fn(rec=d, **kwargs)
        if type(new_signals) is not list:
            raise ValueError('Fn did not return list of signals: {}'.format(m))

        # testing normalization
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
