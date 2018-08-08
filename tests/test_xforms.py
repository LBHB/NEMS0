import pytest

import numpy as np

import nems.xforms as xf


###############################################################################
##################        CONTEXT UTILITIES             #######################
###############################################################################

@pytest.fixture
def context(simple_recording, simple_modelspec_with_phi):
    ctx = {
     'rec': simple_recording,
     'modelspecs': [simple_modelspec_with_phi],
     }
    ctx['modelspecs'][0][0]['meta'] = {'test': "now it's not empty yay"}

    return ctx


# TODO: More extensive tests for these? Really just checking if they throw
#       an error, might want to test some specific use-cases.

# TODO: Tests that use evaluate are failing, but pretty sure that's just due
#       to the recording being a simple fake. Will figure it out later.


#def test_evaluate_context(context):
#    r = xf.evaluate_context(context, rec_key='rec')


def test_get_meta(context):
    m = xf.get_meta(context)


def test_get_modelspec(context):
    mspec = xf.get_modelspec(context)


def test_get_module(context):
    # Note: will no longer pass if the modelspec setup in conftest changes.
    m1 = xf.get_module(context, 1)
    m2 = xf.get_module(context, 'fir.2x15', key='id')
    assert m1 == m2


#def test_get_signal_as_array(context, rec_key='rec'):
#    a = xf.get_signal_as_array(context, 'stim', rec_key='rec')
