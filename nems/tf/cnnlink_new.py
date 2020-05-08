"""Helpers to build a Tensorflow Keras model from a modelspec."""

import logging
import os
import typing
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import pyplot as plt

from nems import recording, get_setting, initializers, modelspec
from nems.tf import callbacks, loss_functions, modelbuilder
import nems.utils

log = logging.getLogger(__name__)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def tf2modelspec(model, modelspec):
    """Populates the modelspec phis with the model layers.

    Does some checking on dims and names as well.
    """
    log.info('Populating modelspec with model weights.')

    # first layer is the input shahpe, so skip it
    for idx, layer in enumerate(model.layers[1:]):
        ms = modelspec[idx]

        if layer.ms_name != ms['fn']:
            raise AssertionError('Model layers and modelspec layers do not match up!')

        phis = layer.weights_to_phi()
        # check that phis/weights match up in both names and shapes
        if phis.keys() != ms['phi'].keys():
            raise AssertionError(f'Model layer "{layer.ms_name}" weights and modelspec phis do not have matching names!')
        for model_weights, ms_phis in zip(phis.values(), ms['phi'].values()):
            if model_weights.shape != ms_phis.shape:
                if layer.ms_name == 'nems.modules.nonlinearity.double_exponential':
                    continue  # dexp has weird weight shapes due to basic init
                raise AssertionError(f'Model layer "{layer.ms_name}" weights and modelspec phis do not have matching '
                                     f'shapes!')

        ms['phi'] = phis

    return modelspec


def eval_tf(model: tf.keras.Model, stim: np.ndarray) -> np.ndarray:
    """Evaluate the mode on the stim."""
    return model.predict(stim)


def compare_ms_tf(ms, model, rec_ms, stim_tf):
    """Compares the evaluation between a modelspec and a keras model.

    For the modelspec, uses a recording object. For the tf model, uses the formatted data from fit_tf."""
    pred_tf = model.predict(stim_tf)
    pred_ms = np.swapaxes(ms.evaluate(rec_ms)['pred']._data, 0, 1).reshape(pred_tf.shape)

    return np.nanstd(pred_ms.flatten() - pred_tf.flatten())


def fit_tf(
        modelspec,
        est: recording.Recording,
        use_modelspec_init: bool = True,
        optimizer: str = 'adam',
        max_iter: int = 10000,
        cost_function: str = 'squared_error',
        early_stopping_steps: int = 5,
        early_stopping_tolerance: float = 5e-4,
        learning_rate: float = 1e-4,
        batch_size: typing.Union[None, int] = None,
        seed: int = 0,
        initializer: str = 'random_normal',
        filepath: typing.Union[str, Path] = None,
        freeze_layers: typing.Union[None, list] = None,
        **context
        ) -> dict:
    """TODO

    :param est:
    :param modelspec:
    :param use_modelspec_init:
    :param optimizer:
    :param max_iter:
    :param cost_function:
    :param early_stopping_steps:
    :param early_stopping_tolerance:
    :param learning_rate:
    :param batch_size:
    :param seed:
    :param filepath:
    :param freeze_layers: Indexes of layers to freeze prior to training. Indexes are modelspec indexes, so are offset
      from model layer indexes.
    :param context:

    :return:
    """
    log.info('Building tensorflow keras model from modelspec.')
    nems.utils.progress_fun()

    # figure out where to save model checkpoints
    if filepath is None:
        filepath = modelspec.meta['modelpath']

    # if job is running on slurm, need to change model checkpoint dir
    job_id = os.environ.get('SLURM_JOBID', None)
    if job_id is not None:
       # keep a record of the job id
       modelspec.meta['slurm_jobid'] = job_id

       log_dir_root = Path('/mnt/scratch')
       assert log_dir_root.exists()
       log_dir_sub = Path('SLURM_JOBID' + job_id) / str(modelspec.meta['batch'])\
                     / modelspec.meta.get('cellid', "NOCELL")\
                     / modelspec.meta['modelname']
       filepath = log_dir_root / log_dir_sub

    filepath = Path(filepath)
    if not filepath.exists():
        filepath.mkdir(exist_ok=True, parents=True)

    filepath = filepath / 'checkpoints'

    # update seed based on fit index
    seed += modelspec.fit_index

    # need to get duration of stims in order to reshape data
    epoch_name = 'REFERENCE'  # TODO: this should not be hardcoded
    # also grab the fs
    fs = est['stim'].fs

    # extract out the raw data, and reshape to (batch, time, channel)
    stim_train = np.transpose(est['stim'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    resp_train = np.transpose(est['resp'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    log.info(f'Feature dimensions: {stim_train.shape}; Data dimensions: {resp_train.shape}.')

    # correlation for monitoring
    # TODO: tf.utils?
    def pearson(y_true, y_pred):
        return tfp.stats.correlation(y_true, y_pred, event_axis=None, sample_axis=None)

    # get the layers and build the model
    cost_fn = loss_functions.get_loss_fn(cost_function)
    model_layers = modelspec.modelspec2tf2(use_modelspec_init=use_modelspec_init, seed=seed, fs=fs, initializer=initializer)
    model = modelbuilder.ModelBuilder(
        name='Test-model',
        layers=model_layers,
        learning_rate=learning_rate,
        loss_fn=cost_fn,
        optimizer=optimizer,
        metrics=[pearson],
    ).build_model(input_shape=stim_train.shape)

    # create the callbacks
    early_stopping = callbacks.DelayedStopper(monitor='loss',
                                              patience=30 * early_stopping_steps,
                                              min_delta=early_stopping_tolerance,
                                              verbose=1,
                                              restore_best_weights=False)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(filepath),
                                                    save_best_only=False,
                                                    save_weights_only=True,
                                                    save_freq=100 * stim_train.shape[0],
                                                    monitor='loss',
                                                    verbose=0)
    sparse_logger = callbacks.SparseProgbarLogger(n_iters=10)
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()

    # freeze layers
    if freeze_layers is not None:
        for freeze_index in freeze_layers:
            model.layers[freeze_index + 1].trainable = False
            log.info(f'Freezing layer "{model.layers[freeze_index + 1].name}".')

    log.info('Fitting model...')
    history = model.fit(
        stim_train,
        resp_train,
        # validation_split=0.2,
        verbose=0,
        epochs=max_iter,
        batch_size=stim_train.shape[0] if batch_size == 0 else batch_size,
        callbacks=[
            sparse_logger,
            nan_terminate,
            early_stopping,
            checkpoint,
            # TODO: tensorboard
        ]
    )

    # did we terminate on a nan? Load checkpoint if so
    if np.all(np.isnan(model.predict(stim_train))):
        log.warning('Model terminated on nan, restoring saved weights.')
        try:
            # this can fail if it nans out before a single checkpoint gets saved
            model.load_weights(str(filepath))
        except tf.errors.NotFoundError:
            pass

    modelspec = tf2modelspec(model, modelspec)
    # compare the predictions from the model and modelspec
    error = compare_ms_tf(modelspec, model, est, stim_train)
    if error > 1e-5:
        log.warning(f'Mean difference between NEMS and TF model prediction: {error}')
    else:
        log.info(f'Mean difference between NEMS and TF model prediction: {error}')

    # add in some relevant meta information
    modelspec.meta['n_parms'] = len(modelspec.phi_vector)
    try:
        n_epochs = len(history.history['loss'])
    except KeyError:
        n_epochs = 0
    try:
        max_iter = modelspec.meta['extra_results']
        modelspec.meta['extra_results'] = max(max_iter, n_epochs)
    except KeyError:
        modelspec.meta['extra_results'] = n_epochs

    nems.utils.progress_fun()

    return {'modelspec': modelspec}


def fit_tf_init(
        modelspec,
        est: recording.Recording,
        **kwargs
        ) -> dict:
    """TODO"""
    layers_to_freeze = [
        'levelshift',
        'relu',
        'stp',
        'state_dc_gain',
        'state_gain',
        'rdt_gain',
    ]

    frozen_idxes = []

    for idx, ms in enumerate(modelspec):
        for layer_to_freeze in layers_to_freeze:
            if layer_to_freeze in ms['fn']:
                frozen_idxes.append(idx)
                break  # break out of inner loop

    modelspec = fit_tf(modelspec, est, freeze_layers=frozen_idxes, **kwargs)['modelspec']

    init_static_nl_layers = [
        'double_exponential',
        'relu',
        'logistic_sigmoid',
        'saturated_rectifier',
    ]

    init_static_idxes = []

    mlen = len(modelspec)
    for idx, ms in enumerate(modelspec[-2:], mlen - 2):
        found = False
        for init_static_nl_layer in init_static_nl_layers:
            if init_static_nl_layer in ms:
                found = True
                break

        if not found:
            init_static_idxes.append(idx)

    init_static_idxes = list(range(mlen-2)) + init_static_idxes
    kwargs['use_modelspec_init'] = True  # don't overwrite the new  phis
    return fit_tf(modelspec, est, freeze_layers=init_static_idxes, **kwargs)


def eval_tf_layer(data: np.ndarray,
                  layer_spec: typing.Union[None, str] = None,
                  stop: typing.Union[None, int] = None,
                  modelspec: modelspec.ModelSpec = None,
                  **kwargs,  # temporary until state data is implemented
                  ) -> np.ndarray:
    """Takes in a numpy array and applies a single tf layer to it.

    :param data: The input data. Shape of (reps, time, channels).
    :param layer_spec: A layer spec for layers of a modelspec.
    :param stop: What layer to eval to. Non inclusive. If not passed, will evaluate the whole layer spec.
    :param modelspec: Optionally use an existing modelspec. Takes precedence over layer_spec.

    :return: The processed data.
    """
    if layer_spec is None and modelspec is None:
        raise ValueError('Either of "layer_spec" or "modelspec" must be specified.')

    if modelspec is not None:
        ms = modelspec
    else:
        ms = initializers.from_keywords(layer_spec)

    layers = ms.modelspec2tf2(use_modelspec_init=True)[:stop]
    model = modelbuilder.ModelBuilder(layers=layers).build_model(input_shape=data.shape)

    pred = model.predict(data)
    return pred
