"""Various tensorflow initializer functions."""

import tensorflow as tf


def weights_tnorm(shape, sig=0.1, seed=0):
    """Truncated normal distribution."""
    weights = tf.Variable(tf.random.truncated_normal(shape, stddev=sig, mean=sig, seed=seed))
    return weights


def weights_norm(shape, sig=0.1, seed=0):
    """Normal distribution."""
    weights = tf.Variable(tf.random.normal(shape, stddev=sig, mean=0, seed=seed))
    return weights


def weights_zeros(shape, sig=0.1, seed=0):
    """All zeros."""
    weights = tf.Variable(tf.zeros(shape))
    return weights


def weights_uniform(shape, sig=0.1, seed=0, minval=0, maxval=1):
    """Random uniform distribution."""
    weights = tf.Variable(tf.random.uniform(shape, minval=minval, maxval=maxval, seed=seed))
    return weights


def weights_glorot_uniform(shape, sig=None, seed=0):
    """Uniform Glorot distribution."""
    weights = tf.Variable(tf.compat.v1.keras.initializers.glorot_uniform(seed)(shape))
    return weights


def weights_he_uniform(shape, sig=None, seed=0):
    """Uniform He distribution."""
    weights = tf.Variable(tf.compat.v1.keras.initializers.he_uniform(seed)(shape))
    return weights


def weights_matrix(d):
    """Variable with specified initial values."""
    weights = tf.Variable(d)
    return weights
