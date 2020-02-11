"""Various loss functions for tensorflow."""

import numpy as np
import tensorflow as tf


def poisson(response, prediction):
    """Poisson loss."""
    return tf.reduce_mean(prediction - response * tf.log(prediction + 1e-5), name='poisson')


def loss_se(response, prediction):
    """Squared error loss."""
    return tf.reduce_mean(tf.square(response - prediction)) / tf.reduce_mean(tf.square(response))


def loss_tf_nmse_shrinkage(response, prediction):
    """Normalized means squared error with shrinkage loss."""
    return tf_nmse_shrinkage(response, prediction)


def loss_tf_nmse(response, prediction):
    """Normalized means squared error loss."""
    mE, sE = tf_nmse(response, prediction)
    return mE


def tf_nmse_shrinkage(response, prediction, shrink_factor=0.5, per_cell=False, thresh=False):
    """Calculates the normalized mean squared error, with an adjustment for error.

    Averages across the batches, but optionally can return a per cell error.

    :param response:
    :param prediction:
    :param float shrink_factor:
    :param bool per_cell: Whether to also average over cells or not
    :param bool thresh:
    :return: a "shrunk" normalized mean squared error
    """
    mE, sE = tf_nmse(response, prediction, per_cell)

    def shrink(mE, sE, shrink_factor, thresh):
        def shrink_all(mE, sE, shrink_factor, thresh):
            return tf_shrinkage(mE, sE, shrink_factor, thresh)

        def shrink_some(mE, sE, shrink_factor, thresh):
            mask_gt, mask_lt = mE >= 1, mE < 1
            # make zero where mE was > 1
            shrunk = tf_shrinkage(mE, sE, shrink_factor, thresh) * tf.dtypes.cast(mask_lt, mE.dtype)
            # add back in
            mE = shrunk + mE * tf.dtypes.cast(mask_gt, mE.dtype)
            return mE

        mE = tf.cond(tf.math.reduce_all(mE < 1), lambda: shrink_all(mE, sE, shrink_factor, thresh),
                     lambda: shrink_some(mE, sE, shrink_factor, thresh))
        return mE

    mE = tf.cond(tf.math.reduce_any(mE < 1), lambda: shrink(mE, sE, shrink_factor, thresh), lambda: mE)

    if per_cell:
        mE = tf.math.reduce_mean(mE)

    return mE


def tf_nmse(response, prediction, per_cell=False):
    """Calculates the normalized mean squared error across batches.

    Optionally can return an average per cell.

    :param response:
    :param prediction:
    :param per_cell: Whether to average across all cells or not
    :return: 2 tensors, one of the mean error, the other of the std of the error. If not per cell, then
     tensor is of shape (), else tensor if of shape (n_cells,) (i.e. last dimension of the resp/pred tensor)
    """
    n_drop = list(response.shape)[-2] % 10
    if n_drop:
        # use slices to handle varying tensor shapes
        drop_slice = [slice(None) for i in range(len(response.shape))]

        # second last dim is time
        drop_slice[-2] = slice(None, -n_drop)
        drop_slice = tuple(drop_slice)

        _response = response[drop_slice]
        _prediction = prediction[drop_slice]
    else:
        _response = response
        _prediction = prediction

    if per_cell:
        _response = tf.transpose(_response, np.roll(np.arange(len(response.shape)), 1))
        _prediction = tf.transpose(_prediction, np.roll(np.arange(len(response.shape)), 1))

        _response = tf.reshape(_response, shape=(_response.shape[0], 10, -1))
        _prediction = tf.reshape(_prediction, shape=(_prediction.shape[0], 10, -1))
    else:
        _response = tf.reshape(_response, shape=(10, -1))
        _prediction = tf.reshape(_prediction, shape=(10, -1))

    squared_error = ((_response - _prediction) ** 2)
    nmses = tf.math.reduce_mean(squared_error, axis=-1) ** 0.5 / tf.math.reduce_std(_response, axis=-1)

    mE = tf.math.reduce_mean(nmses, axis=-1)
    sE = tf.math.reduce_std(nmses, axis=-1) / 10 ** 0.5

    return mE, sE


def tf_shrinkage(mE, sE, shrink_factor=0.5, thresh=False):
    """Adjusts the mean error based on the standard error"""
    mE = 1 - mE
    smd = tf.math.divide_no_nan(abs(mE), sE) / shrink_factor
    smd = 1 - smd ** -2

    if thresh:
        return 1 - mE * tf.dtypes.cast(smd > 1, mE.dtype)

    smd = smd * tf.dtypes.cast(smd > 0, smd.dtype)

    return 1 - mE * smd
