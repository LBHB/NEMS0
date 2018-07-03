import re
import os
import sys
import importlib as imp
import logging
log = logging.getLogger(__name__)


class KeywordRegistry():
    '''
    Behaves similar to a dictionary, except that
    registry[key_string] will trigger a function call based on
    the leading portion of the key (either before the first hyphen or
    until the first non-alpha character).

    The function call is determined by the Keyword instance referenced by
    the leading portion of the key, and it will receive the keyword string
    itself as the first argument followed by *args.

    For example:
        kws = KeywordRegistry(' arg1', ' arg2')
        kws['mykey'] = lambda x, y, z: x + y + z

        In[0]: kws['mykey-testing']
        Out[0]: 'mykey-testing arg1 arg2'

        kws['myfir'] = {'fn': nems.modules.fir.basic, 'phi': ...}
        modelspec.append(kws['myfir15x2'])

    See register_plugin and register_modules for how to easily add entire
    directories or modules of keyword definitions.
    '''

    def __init__(self, *args):
        self.keywords = {}
        self.args = args

    def __getitem__(self, kw_string):
        kw = self.lookup(kw_string)
        return kw.parse(kw_string, *self.args)

    def __setitem__(self, kw_head, parse):
        # TODO: Warning either here or in register_module / register_plugins
        #       to notify if keyword being overwritten?
        self.keywords[kw_head] = Keyword(kw_head, parse)

    def kw_head(self, kw_string):
        # if the full kw_string is in the registry as-is, then it's a
        # backwards-compatibility alias and overrides the normal kw head rule.
        if kw_string in self.keywords:
            return kw_string
        # look for '-' first. if not present, use first alpha-only string
        # as head instead.
        h = kw_string.split('.')
        if len(h) == 1:
            # no hypen, do regex for first all-alpha string
            alpha = re.compile('^[a-zA-Z]*')
            kw_head = re.match(alpha, kw_string)[0]
        else:
            kw_head = h[0]
        return kw_head

    def lookup(self, kw_string):
        kw_head = self.kw_head(kw_string)
        return self.keywords[kw_head]

    def register_plugin(self, d):
        '''
        Adds all callable() variables from all modules contained within
        the specified package directory as eponymous keywords.
        Additional, user-defined keywords should be registered via this method.

        Parameters
        ----------
        d : str
            A path to a directory containing one or more modules that
            define keyword functions.
        '''
        if d.endswith('.py'):
            package, mod = os.path.split(d)
            sys.path.append(package)
            package_name = os.path.split(package)[-1]
            module_name = mod[:-3]
            modules = [imp.import_module(module_name, package=package_name)]
        else:
            sys.path.append(d)
            if d.endswith('/'):
                d = d[:-1]
            package_name = os.path.split(d)[-1]
            modules = [
                    imp.import_module(f[:-3], package=package_name)
                    for f in os.listdir(d) if f.endswith('.py')
                    ]
        self.register_modules(modules)

    def register_plugins(self, pkgs):
        '''Invokes self.register_plugin for each package listed in pkgs.'''
        [self.register_plugin(p) for p in pkgs]

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
        [self.register_module(m) for m in modules]

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
        d = {k: v.to_string() for k, v in self.keywords.items()}
        d['_KWR_ARGS'] = self.args
        return d

    @classmethod
    def from_json(self, d):
        r = KeywordRegistry(*d['_KWR_ARGS'])
        d.pop('_KWR_ARGS')
        r.keywords = {k: getattr(imp.import_module(v), k)
                      for k, v in d.items()}
        return r


class Keyword():
    '''
    Ex: `kw_head = 'ozgf'`
        `parse = mymodule.keywords.ozgf`
        `parse('ozgf-123') = [{'fn': module1}, {'fn': module2}]`
    '''

    def __init__(self, kw_head, parse):
        self.key = kw_head
        self.parse = parse

    def to_string(self):
        return self.parse.__module__
