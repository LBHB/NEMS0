import pytest

from nems.registry import KeywordRegistry
from nems.plugins import default_loaders
from nems.plugins import default_keywords
from nems.plugins import default_fitters


@pytest.fixture
def loader_registry():
    loaders = KeywordRegistry(recording_uri='dummy_recording_uri')
    loaders.register_module(default_loaders)
    return loaders


@pytest.fixture
def model_registry():
    models = KeywordRegistry()
    models.register_module(default_keywords)
    models.register_plugins(['tests/resources/plugin1.py'])
    return models


@pytest.fixture
def fitter_registry():
    fitters = KeywordRegistry()
    fitters.register_module(default_fitters)
    return fitters


def test_loader_registry(loader_registry):
    # All of these should be fine and should return equivalent xfspecs.
    one = loader_registry['ld']
    two = loader_registry['ld.']
    three = loader_registry['ld.whatever']
    assert one == two == three

    # But .n should add normalization
    four = loader_registry['ld.n']
    assert one != four


def test_model_registry(model_registry):
    # Each of these keywords should return a value error since they are
    # missing required options (like inputs x outputs for wc)
    errors = ['wc', 'stp', 'fir', 'lvl', 'dexp']
    for e in errors:
        with pytest.raises((AttributeError, ValueError)):
            model_registry[e]

    # These ones should all work. Not an exhaustive list, but should be
    # a representative sample.
    fine = ['wc.2x15', 'wc.2x15.g.n', 'dlog', 'dlog.n18', 'fir.15x2',
            'fir.15x2x4', 'stp.2', 'stp.2.z.n.b', 'dexp.2', 'lvl.1',
            'qsig.5', 'logsig', 'tanh.4', 'stategain.2x4', 'rep.4', 'mrg']
    for f in fine:
        x = model_registry[f]

    # TODO: Specific test cases?


def test_fitter_registry(fitter_registry):
    tests = ['basic', 'iter', 'basic.cd', 'iter.cd', 'basic.shr',
             'iter.shr', 'basic.nf5', 'iter.nf10.shr', 'basic.st',
             'iter.st', 'iter.cd.nf10.shr.st.T3,5,7.S0,1.S1,2.ti50.fi20']
    for t in tests:
        x = fitter_registry[t]


def test_register_plugins():
    registry = KeywordRegistry()
    registry.register_plugins(['tests/resources/plugin1.py'])
    registry['firstkw']
    with pytest.raises(KeyError):
        registry['secondkw']
    registry.register_plugins(['tests/resources/plugin2.py'])
    registry['secondkw']

    # Hack to test importable module names
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    registry = KeywordRegistry()
    registry.register_plugins(['tests.resources.plugin1'])
    registry['firstkw']
    with pytest.raises(KeyError):
        registry['secondkw']

    registry.register_plugins(['tests/resources'])
    registry['firstkw']
    registry['secondkw']


def test_jsonify(model_registry):
    json = model_registry.to_json()
    unjson = KeywordRegistry.from_json(json)
    unjson.keywords == model_registry.keywords
