import pytest

from nems.registry import KeywordRegistry
from nems.plugins.loaders import default_loaders
from nems.plugins.keywords import default_keywords
from nems.plugins.fitters import default_fitters


@pytest.fixture
def loader_registry():
    loaders = KeywordRegistry('dummy_recording_uri')
    loaders.register_module(default_loaders)
    return loaders


@pytest.fixture
def model_registry():
    models = KeywordRegistry()
    models.register_module(default_keywords)
    return models


@pytest.fixture
def fitter_registry():
    fitters = KeywordRegistry()
    fitters.register_module(default_fitters)
    return fitters


def test_loader_registry(loader_registry):
    # Default loader has no options since it just loads the recording.
    # So all of these should be fine and should return equivalent xfspecs.
    one = loader_registry['load']
    two = loader_registry['load.']
    three = loader_registry['load.whatever']

    assert one == two == three


def test_model_registry(model_registry):
    # Each of these keywords should return a value error since they are
    # missing required options (like inputs x outputs for wc)
    errors = ['wc', 'stp', 'fir', 'lvl', 'dexp']
    for e in errors:
        with pytest.raises(ValueError):
            model_registry[e]

    # These ones should all work. Not an exhaustive list, but should be
    # a representative sample.
    fine = ['wc.2x15', 'wc.2x15.g.n', 'dlog', 'dlog.n18', 'fir.15x2',
            'fir.15x2x4', 'stp.2', 'stp.2.z.n.b', 'dexp.2', 'lvl.1',
            'qsig.5', 'logsig', 'tanh.4', 'stategain.2', 'rep.4', 'mrg']
    for f in fine:
        x = model_registry[f]

    # TODO: Specific test cases?


def test_fitter_registry(fitter_registry):
    tests = ['basic', 'iter', 'basic.cd', 'iter.cd', 'basic.shr',
             'iter.shr', 'basic.nf5', 'iter.nf10.shr', 'basic.st',
             'iter.st', 'iter.cd.nf10.shr.st.T3,5,7.S0,1.S1,2.ti50.fi20']
    for t in tests:
        x = fitter_registry[t]
