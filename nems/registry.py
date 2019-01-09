import logging
import importlib.util
import inspect
import re
import os
from pathlib import Path
import sys

log = logging.getLogger(__name__)


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
        return self.keywords[kw_head]

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
        self.register_module(module)

    def register_plugins(self, locations):
        '''Invokes self.register_plugin for each package listed in pkgs.'''
        print(locations)
        print(type(locations))
        for loc in locations:
            self.register_plugin(loc)

    def register_module(self, module):
        '''
        As register_plugin, but module should be a single module object
        instead of a path for a directory.
        Default nems keywords are added via this method.
        '''
        if not module:
            return
        for a in dir(module):
            if self._validate(module, a):
                self.keywords[a] = Keyword(a, getattr(module, a))

    def register_modules(self, modules):
        '''Invokes self.register_module for each module listed in modules.'''
        for module in modules:
            self.register_module(module)

    def _validate(self, m, a):
        '''
        Ignore private functions, all-caps global variables,
        and anything that isn't callable.
        '''
        private = (a.startswith('_'))
        caps = (a.upper() == a)
        is_callable = (callable(getattr(m, a)))
        if (is_callable) and (not private) and (not caps):
            return True
        else:
            return False

    def __iter__(self):
        # Pass through keywords dictionary's iteration to allow
        # `for k in registry: do x`
        return self.keywords.__iter__()

    def __next__(self):
        return self.keywords.__next__()

    def to_json(self):
        d = {k: v.file_string() for k, v in self.keywords.items()}
        d['_KWR_ARGS'] = self.kwargs
        return d

    @classmethod
    def from_json(self, d):
        r = KeywordRegistry(*d['_KWR_ARGS'])
        d.pop('_KWR_ARGS')
        try:
            r.keywords = {k: getattr(importlib.import_module(v), k)
                      for k, v in d.items()}
            plugins = set([v for k, v in d.items()])
            r.register_plugins(plugins)
        except:
            # Assume that equivalent plugins are specified in configs.
            # Will happen when loading a model on a PC that it
            # was not fit on, for example.
            pass

        return r


class Keyword():
    '''
    Ex: `kw_head = 'ozgf'`
        `parse = mymodule.keywords.ozgf`
        `parse('ozgf.123') = [{'fn': module1}, {'fn': module2}]`
    '''

    def __init__(self, kw_head, parse):
        self.key = kw_head
        self.parse = parse

    def file_string(self):
        return inspect.getmodule(self.parse).__file__
