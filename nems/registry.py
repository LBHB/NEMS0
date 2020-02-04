import logging
import importlib.util
import inspect
import re
import os
from pathlib import Path
import sys

log = logging.getLogger(__name__)


def is_keyword(name, obj):
    '''
    Determine whether named object is a keyword

    Parameters
    ----------
    name : string
        Name of object
    obj : object
        Object

    False if private function, all-caps global variable or not callable. True
    otherwise.
    '''
    private = name.startswith('_')
    caps = name.upper() == name
    is_callable = callable(obj)
    return is_callable and (not private) and not(caps)


class KeywordMissingError(Exception):
    """Raised when a keyword lookup fails"""
    def __init__(self, value):
        default_message = f'"{value}" could not be found in the keyword registry.'
        super().__init__(default_message)


class KeywordRegistry():
    '''
    Behaves similar to a dictionary, except that
    registry[key_string] will trigger a function call based on
    the leading portion of the key (either before the first period or
    until the first non-alpha character).

    The function call is determined by the Keyword instance referenced by
    the leading portion of the key, and it will receive the keyword string
    itself as the first argument followed by **kwargs.

    For example:
        def add_kwargs(kw, myarg1='', myarg2=''):
            return kw + myarg1 + myarg2

        kws = KeywordRegistry(myarg1=' first', myarg2=' second')
        kws['mykey'] = add_kwargs

        In[0]: kws['mykey.test']
        Out[0]: 'mykey.test first second'

    See register_plugin and register_modules for how to add entire
    directories or modules of keyword definitions.
    '''

    def __init__(self, **kwargs):
        self.keywords = {}
        self.kwargs = kwargs

    def __getitem__(self, kw_string):
        kw = self.lookup(kw_string)
        accepted_args = inspect.getfullargspec(kw.parse)[0]
        kwargs = {k: v for k, v in self.kwargs.items() if k in accepted_args}
        return kw.parse(kw_string, **kwargs)

    def __setitem__(self, kw_head, parse):
        # TODO: Warning either here or in register_module / register_plugins
        #       to notify if keyword being overwritten?
        self.keywords[kw_head] = Keyword(kw_head, parse)

    def kw_head(self, kw_string):
        # if the full kw_string is in the registry as-is, then it's a
        # backwards-compatibility alias and overrides the normal kw head rule.
        if kw_string in self.keywords:
            return kw_string
        # Look for '.' first. If not present, use first alpha-only string
        # as head instead.
        h = kw_string.split('.')
        if len(h) == 1:
            # no period, do regex for first all-alpha string
            alpha = re.compile('^[a-zA-Z]*')
            kw_head = re.match(alpha, kw_string).group(0)
        else:
            kw_head = h[0]
        return kw_head

    def lookup(self, kw_string):
        kw_head = self.kw_head(kw_string)
        if kw_head not in self.keywords:
            raise KeywordMissingError(kw_head)
        
        return self.keywords[kw_head]

    def source(self, kw_string):
        kw_head = self.kw_head(kw_string)
        return self.keywords[kw_head].source_string

    def register_plugin(self, location):
        '''
        Registers a plugin

        Parameters
        ----------
        location: string
            Can be one of:
            * module name (e.g., `my_code.plugins.keywords`)
            * file name (e.g., `/path/my_code/plugins/keywords.py')
            * path name (e.g., '/path/my_code/plugins')
        '''
        pathname = Path(location)
        if pathname.exists():
            if pathname.is_dir():
                self._register_plugin_by_path(pathname)
            else:
                self._register_plugin_by_file(pathname)
        else:
            self._register_plugin_by_module_name(location)

    def _register_plugin_by_module_name(self, module_name):
        module = importlib.import_module(module_name)
        module.__location__ = module_name
        self.register_module(module)

    def _register_plugin_by_path(self, pathname):
        for filename in pathname.glob('*.py'):
            self._register_plugin_by_file(filename)

    def _register_plugin_by_file(self, filename):
        '''
        Adds all callable() names defined inside the Python file as keywords

        Parameters
        ----------
        filename : str
            Path to a file containing one or more modules that define keyword
            functions.
        '''
        spec = importlib.util.spec_from_file_location('', filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.__location__ = filename
        self.register_module(module)

    def register_plugins(self, locations):
        '''Invokes self.register_plugin for each package listed in pkgs.'''
        for loc in locations:
            self.register_plugin(loc)

    def register_module(self, module):
        '''
        As register_plugin, but module should be a single module object
        instead of a path for a directory.
        Default nems keywords are added via this method.
        '''
        try:
            location = str(module.__location__)
        except AttributeError:
            # Always default to the name of the module rather than the file
            # because this makes code more portable across platforms.
            location = str(module.__name__)

        for name, obj in vars(module).items():
            if is_keyword(name, obj):
                self.keywords[name] = Keyword(name, obj, location)

    def register_modules(self, modules):
        '''Invokes self.register_module for each module listed in modules.'''
        for module in modules:
            self.register_module(module)

    def __iter__(self):
        # Pass through keywords dictionary's iteration to allow
        # `for k in registry: do x`
        return self.keywords.__iter__()

    def __next__(self):
        return self.keywords.__next__()

    def to_json(self):
        d = {k: v.source_string for k, v in self.keywords.items()}
        d['_KWR_ARGS'] = self.kwargs
        return d

    @classmethod
    def from_json(self, d):
        registry = KeywordRegistry(*d['_KWR_ARGS'])
        d.pop('_KWR_ARGS')
        unique_sources = set(d.values())
        for source in unique_sources:
            try:
                registry.register_plugin(source)
            except:
                log.warn('Unable to register plugin %s', source)
                # Assume that equivalent plugins are specified in configs.
                # Will happen when loading a model on a PC that it was not fit
                # on, for example.
        return registry

    def info(self, kw=None):
        """
        :param kw:
        :return: dictionary with source info for keywords containing string kw
                 or (if kw is None) all keywords
        """
        kw_list = list(self.keywords.keys())
        s = {}
        for k in kw_list:
            if (kw is None) or kw in k:
                s[k] = self.source(k)
        return s

class Keyword():
    '''
    Parameters
    ----------
    key : string
        Keyword (just the base, excluding parameters). e.g., 'ozgf'
    parse : callable
        Function that parses full keyword string
    source_string : string
        Location (i.e., module or file) that keyword was defined in
    '''

    def __init__(self, key, parse, source_string):
        self.key = key
        self.parse = parse
        self.source_string = source_string

def test_xforms_kw(kw_string, **xforms_kwargs):
    from nems.plugins import (default_loaders,
                              default_initializers, default_fitters)
    from nems import get_setting

    xforms_lib = KeywordRegistry(**xforms_kwargs)
    xforms_lib.register_modules([default_loaders, default_fitters,
                                default_initializers])
    xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

    return xforms_lib[kw_string]

def test_modelspec_kw(kw_string):
    from nems.plugins import (default_keywords)
    from nems import get_setting

    kw_lib = KeywordRegistry()
    kw_lib.register_modules([default_keywords])
    kw_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

    return kw_lib[kw_string]

