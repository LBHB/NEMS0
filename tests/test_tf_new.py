from functools import wraps

import numpy as np
import pytest

from tensorflow import config

from nems.modelspec import eval_ms_layer
from nems.tf.cnnlink_new import eval_tf_layer


@pytest.fixture()
def data(shape=(20, 100, 18)):
    data = np.random.random(shape).astype('float32')
    data[:, :10, :] = 0
    data[:, -10:, :] = 0

    return data


@pytest.fixture()
def state_data(shape=(20, 100, 3)):
    state_data = np.random.random(shape).astype('float32')
    state_data[:, :10, :] = 0
    state_data[:, -10:, :] = 0

    return state_data


@pytest.fixture()
def kern_size():
    return 4


def requires_gpu(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        if config.list_physical_devices('GPU'):
            return f(*args, **kwds)
        else:
            print(f'No GPUs detected, skipping "{f.__name__}".')
            return True
    return wrapper


def compare_ms_tf(layer_spec, test_data, test_state_data=None):
    """Evaluate and compare layers."""
    ms_resp = eval_ms_layer(test_data, layer_spec, state_data=test_state_data)
    tf_resp = eval_tf_layer(test_data, layer_spec, state_data=test_state_data)

    return np.allclose(ms_resp, tf_resp, rtol=1e-5, atol=1e-5)


def test_wc_b(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'wc.{in_size}x{kern_size}.b'
    assert compare_ms_tf(layer_spec, data)


def test_wc_g(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'wc.{in_size}x{kern_size}.g'
    assert compare_ms_tf(layer_spec, data)


def test_fir(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'fir.{in_size}x{kern_size}'
    assert compare_ms_tf(layer_spec, data)


def test_fir_bank(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'fir.{in_size}x{kern_size}x3'
    assert compare_ms_tf(layer_spec, data)


def test_fir_bank2(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'fir.{in_size}x3x{kern_size}'
    assert compare_ms_tf(layer_spec, data)


#@requires_gpu
def test_fir_bank3(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'fir.2x3x{in_size // 2}'
    assert compare_ms_tf(layer_spec, data)


@requires_gpu
def test_fir_bank4(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'fir.2x4x{in_size // 2}'
    assert compare_ms_tf(layer_spec, data)


@requires_gpu
def test_fir_bank_single(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'fir.1x{kern_size}x{in_size}'
    assert compare_ms_tf(layer_spec, data)


@pytest.mark.xfail
def test_do(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'do.{in_size}x{kern_size}'
    assert compare_ms_tf(layer_spec, data)


def test_stategain(data, state_data, kern_size):
    in_size = state_data.shape[-1]
    layer_spec = f'stategain.{in_size}'
    assert compare_ms_tf(layer_spec, data, state_data)


def test_relu(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'relu.{in_size}'
    assert compare_ms_tf(layer_spec, data)


def test_dlog(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'dlog.c{in_size}'
    assert compare_ms_tf(layer_spec, data)


def test_dexp(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'dexp.{in_size}'
    assert compare_ms_tf(layer_spec, data)


@pytest.mark.xfail
# failing because of some new changes made to stpq in tf
# but not also made in nems
def test_stp_q(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'stp.{in_size}.q'
    assert compare_ms_tf(layer_spec, data)


@pytest.mark.xfail
def test_stp(data, kern_size):
    in_size = data.shape[-1]
    layer_spec = f'stp.{in_size}'
    assert compare_ms_tf(layer_spec, data)
