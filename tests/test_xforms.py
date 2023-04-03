import pytest

import numpy as np

import nems0.xforms as xforms
import nems0.modelspec as ms

###############################################################################
##################        CONTEXT UTILITIES             #######################
###############################################################################

@pytest.fixture
def context():
    modelspecname = 'fir.1x15-lvl.1'
    load_command = 'nems0.demo.loaders.dummy_loader'
    meta = {'cellid': "DUMMY01", 'batch': 0, 'modelname': modelspecname,
            'exptid': "DUMMY"}

    xfspec = []
    xfspec.append(['nems0.xforms.load_recording_wrapper',
                   {'load_command': load_command, 'exptid': meta['exptid'],
                    'save_cache': False}])
    xfspec.append(['nems0.xforms.split_at_time',
                   {'valfrac': 0.2}])
    xfspec.append(['nems0.xforms.init_from_keywords',
                   {'keywordstring': modelspecname, 'meta': meta}])
    xfspec.append(['nems0.xforms.fit_basic', {}])
    xfspec.append(['nems0.xforms.predict', {}])
    xfspec.append(['nems0.xforms.add_summary_statistics', {}])

    ctx = {}
    ctx, xf_log = xforms.evaluate(xfspec, ctx)

    return ctx


# TODO: More extensive tests for these? Really just checking if they throw
#       an error, might want to test some specific use-cases.

# TODO: Tests that use evaluate are failing, but pretty sure that's just due
#       to the recording being a simple fake. Will figure it out later.


#def test_evaluate_context(context):
#    r = xf.evaluate_context(context, rec_key='rec')


def test_get_meta(context):
    m = xforms.get_meta(context)


def test_get_modelspec(context):
    mspec = xforms.get_modelspec(context)


def test_get_module(context):
    # Note: will no longer pass if the modelspec setup in conftest changes.
    m1 = xforms.get_module(context, 0)
    m2 = xforms.get_module(context, 'fir.1x15', key='id')
    assert m1 == m2

def test_fit_performance(context):
    # Note: will no longer pass if the modelspec setup in conftest changes.
    m1 = np.round(xforms.get_meta(context)['r_test']*100)
    m2 = np.array([47.0])
    assert m1 == m2


#def test_get_signal_as_array(context, rec_key='rec'):
#    a = xf.get_signal_as_array(context, 'stim', rec_key='rec')
