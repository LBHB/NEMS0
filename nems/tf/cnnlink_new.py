"""Helpers to build a Tensorflow Keras model from a modelspec."""

import copy
import logging
import os
import typing
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import nems.utils
from nems import initializers, recording, get_setting
from nems import modelspec as mslib
from nems.tf import callbacks, loss_functions, modelbuilder
from nems.tf.layers import Conv2D_NEMS
from nems.initializers import init_static_nl

log = logging.getLogger(__name__)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def tf2modelspec(model, modelspec):
    """Populates the modelspec phis with the model layers.

    Does some checking on dims and names as well.
    """
    log.info('Populating modelspec with model weights.')

    # need to keep track of difference between tf layer indices and ms indices because of additional input layers
    idx_offset = 0

    # first layer is the input shape, so skip it
    for idx, layer in enumerate(model.layers):
        # skip input layers (input layers can be midway if using state inputs
        if isinstance(layer, tf.keras.layers.InputLayer):
            idx_offset += 1
            continue

        ms = modelspec[idx - idx_offset]

        if layer.ms_name != ms['fn']:
            raise AssertionError('Model layers and modelspec layers do not match up!')

        phis = layer.weights_to_phi()
        if np.any(['tf_only' in fn for fn in modelspec.fn()]):
            # These TF layers don't have corresponding phi pre-set, so there's nothing to check against.
            # TODO: format phi properly for this model version
            pass
        else:
            # check that phis/weights match up in name
            if phis.keys() != ms['phi'].keys():
                raise AssertionError(f'Model layer "{layer.ms_name}" weights and modelspec phis do not have matching names!')

            # check that phis/weights match up in shape
            for model_weights, ms_phis in zip(phis.values(), ms['phi'].values()):
                if model_weights.shape != ms_phis.shape:
                    if layer.ms_name == 'nems.modules.nonlinearity.double_exponential':
                        continue  # dexp has weird weight shapes due to basic init
                    raise AssertionError(f'Model layer "{layer.ms_name}" weights and modelspec phis do not have matching '
                                         f'shapes!')

            # for non trainable layers, check that values didn't change (within tolerance of gpu computation)
            # TODO: for non trainable layers, should we just not update the modelspec weights?
            if not layer.trainable:
                for model_weights, ms_phis in zip(phis.values(), ms['phi'].values()):
                    if not np.allclose(model_weights, ms_phis, rtol=5e-2, atol=5e-2):
                        log.warning(f'Frozen layer weights changed:\n{ms_phis}\n{model_weights}')
                        log.warning(f'Model layer "{layer.ms_name}" weights changed significantly despite being frozen!')

        ms['phi'] = phis

    return modelspec


def eval_tf(model: tf.keras.Model, stim: np.ndarray) -> np.ndarray:
    """Evaluate the mode on the stim."""
    return model.predict(stim)


def compare_ms_tf(ms, model, rec_ms, train_data):
    """Compares the evaluation between a modelspec and a keras model.

    For the modelspec, uses a recording object. For the tf model, uses the formatted data from fit_tf,
    which may include state data."""
    pred_tf = model.predict(train_data)
    pred_ms = np.swapaxes(ms.evaluate(rec_ms.apply_mask())['pred']._data, 0, 1).reshape(pred_tf.shape)

    # for idx, layer in enumerate(model.layers[1:]):
    #     for weight in layer.weights:
    #         if np.any(np.isnan(weight.numpy())):
    #             print(idx, layer.name, weight.name)

    allclose = np.allclose(pred_ms, pred_tf, rtol=1e-5, atol=1e-5)
    # if not allclose and not np.isnan(allclose):
    #     import matplotlib.pyplot as plt
    #     d = np.abs(pred_ms - pred_tf)
    #     ind = np.unravel_index(np.argmax(d, axis=None), d.shape)
    #     ind = np.s_[ind[0], :, ind[2]]
    #
    #     fig, axes = plt.subplots(figsize=(20, 8), nrows=2)
    #
    #     upto = 200
    #
    #     disp_ms = pred_ms[ind]
    #     disp_tf = pred_tf[ind]
    #
    #     axes[0].plot(disp_ms[:upto], alpha=0.5, label='ms')
    #     axes[0].plot(disp_tf[:upto], alpha=0.5, label='tf')
    #     _ = axes[0].legend()
    #
    #     diff = (disp_ms - disp_tf)[:upto]
    #     axes[1].plot(np.abs(diff), label='diff')
    #     _ = axes[1].legend()
    #     plt.show()

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
        IsReload: bool = False,
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

    if IsReload:
        return {}

    tf.random.set_seed(seed)
    np.random.seed(seed)
    #os.environ['TF_DETERMINISTIC_OPS'] = '1'   # makes output deterministic, but reduces prediction accuracy

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

    checkpoint_filepath = filepath / 'weights.hdf5'
    tensorboard_filepath = filepath / 'logs'
    gradient_filepath = filepath / 'gradients'

    # update seed based on fit index
    seed += modelspec.fit_index

    # need to get duration of stims in order to reshape data
    epoch_name = 'REFERENCE'  # TODO: this should not be hardcoded
    input_name = modelspec.meta.get('input_name', 'stim')
    output_name = modelspec.meta.get('output_name', 'resp')

    # also grab the fs
    fs = est[input_name].fs

    # extract out the raw data, and reshape to (batch, time, channel)
    stim_train = np.transpose(est[input_name].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    resp_train = np.transpose(est[output_name].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
    log.info(f'Feature dimensions: {stim_train.shape}; Data dimensions: {resp_train.shape}.')

    # get state if present, and setup training data
    if 'state' in est.signals:
        state_train = np.transpose(est['state'].extract_epoch(epoch=epoch_name, mask=est['mask']), [0, 2, 1])
        state_shape = state_train.shape
        log.info(f'State dimensions: {state_shape}')
        train_data = [stim_train, state_train]
    else:
        state_train, state_shape = None, None
        train_data = stim_train

    # correlation for monitoring
    # TODO: tf.utils?
    def pearson(y_true, y_pred):
        return tfp.stats.correlation(y_true, y_pred, event_axis=None, sample_axis=None)

    # get the layers and build the model
    cost_fn = loss_functions.get_loss_fn(cost_function)
    model_layers = modelspec.modelspec2tf2(use_modelspec_init=use_modelspec_init, seed=seed, fs=fs, initializer=initializer)
    if np.any([isinstance(layer, Conv2D_NEMS) for layer in model_layers]):
        # need a "channel" dimension for Conv2D (like rgb channels, not frequency). Only 1 channel for our data.
        stim_train = stim_train[..., np.newaxis]
        train_data = train_data[..., np.newaxis]

    # do some batch sizing logic
    batch_size = stim_train.shape[0] if batch_size == 0 else batch_size

    model = modelbuilder.ModelBuilder(
        name='Test-model',
        layers=model_layers,
        learning_rate=learning_rate,
        loss_fn=cost_fn,
        optimizer=optimizer,
        metrics=[pearson],
    ).build_model(input_shape=stim_train.shape, state_shape=state_shape, batch_size=batch_size)

    # tracking early termination
    model.early_terminated = False

    # create the callbacks
    early_stopping = callbacks.DelayedStopper(monitor='loss',
                                              patience=30 * early_stopping_steps,
                                              min_delta=early_stopping_tolerance,
                                              verbose=1,
                                              restore_best_weights=False)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_filepath),
                                                    save_best_only=False,
                                                    save_weights_only=True,
                                                    save_freq=100 * stim_train.shape[0],
                                                    monitor='loss',
                                                    verbose=0)
    sparse_logger = callbacks.SparseProgbarLogger(n_iters=10)
    nan_terminate = tf.keras.callbacks.TerminateOnNaN()
    nan_weight_terminate = callbacks.TerminateOnNaNWeights()
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=str(tensorboard_filepath),  # TODO: generic tensorboard dir?
                                                histogram_freq=0,  # record the distribution of the weights
                                                write_graph=False,
                                                update_freq='epoch',
                                                profile_batch=0)
    # gradient_logger = callbacks.GradientLogger(filepath=str(gradient_filepath),
    #                                            train_input=stim_train,
    #                                            model=model)

    # freeze layers
    if freeze_layers is not None:
        for freeze_index in freeze_layers:
            model.layers[freeze_index + 1].trainable = False
            log.info(f'Freezing layer #{freeze_index}: "{model.layers[freeze_index + 1].name}".')

    # save an initial set of weights before freezing, in case of termination before any checkpoints
    model.save_weights(str(checkpoint_filepath), overwrite=True)

    log.info(f'Fitting model (batch_size={batch_size}...')
    history = model.fit(
        train_data,
        resp_train,
        # validation_split=0.2,
        verbose=2,
        epochs=max_iter,
        batch_size=batch_size,
        callbacks=[
            #sparse_logger,
            nan_terminate,
            nan_weight_terminate,
            early_stopping,
            checkpoint,
            # enable the below to log tracked parameters to tensorboard
            # tensorboard,
            # enable the below to record gradients to visualize in tensorboard; this is very slow,
            # and loading all this into tensorboard can use A LOT of memory
            # gradient_logger,
        ]
    )

    # did we terminate on a nan loss or weights? Load checkpoint if so
    if np.all(np.isnan(model.predict(train_data))) or model.early_terminated:  # TODO: should this be np.any()?
        log.warning('Model terminated on nan loss or weights, restoring saved weights.')
        try:
            # this can fail if it nans out before a single checkpoint gets saved, either because no saved weights
            # exist, or it tries to load a in different model from the init
            model.load_weights(str(checkpoint_filepath))
            log.warning('Reloaded previous saved weights after nan loss.')
        except (tf.errors.NotFoundError, ValueError):
            pass

    modelspec = tf2modelspec(model, modelspec)

    contains_tf_only_layers = np.any(['tf_only' in m['fn'] for m in modelspec.modules])
    if not contains_tf_only_layers:
        # compare the predictions from the model and modelspec
        error = compare_ms_tf(modelspec, model, est, train_data)
        if error > 1e-5:
            log.warning(f'Mean difference between NEMS and TF model prediction: {error}')
        else:
            log.info(f'Mean difference between NEMS and TF model prediction: {error}')
    else:
        # nothing to compare, ms evaluation is not implemented for this type of model
        pass

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
        nl_init: str = 'tf',
        IsReload: bool = False,
        **kwargs
        ) -> dict:
    """Inits a model using tf.

    Makes a new model up to the last relu or first levelshift, in the process setting levelshift to the mean of the
    resp. Excludes in the new model stp, rdt_gain, state_dc_gain, state_gain. Fits this. Then runs init_static_nl ,
    which looks at the last 2 layers of the original model, and if any of dexp, relu, log_sig, sat_rect are in those
    last two, only fits the first it encounters (freezes all other layers).
    """
    if IsReload:
        return {}

    def first_substring_index(strings, substring):
        try:
            return next(i for i, string in enumerate(strings) if substring in string)
        except StopIteration:
            return None

    # find the first 'lvl' or last 'relu'
    ms_modules = [ms['fn'] for ms in modelspec]
    #up_to_idx = first_substring_index(ms_modules, 'levelshift')
    relu_idx = first_substring_index(reversed(ms_modules), 'relu')
    lvl_idx = first_substring_index(reversed(ms_modules), 'levelshift')
    _idxs = [i for i in [relu_idx, lvl_idx, len(modelspec)-1] if i is not None]
    up_to_idx = len(modelspec) - 1 - np.min(_idxs)
    #last_idx = np.min([relu_idx, lvl_idx])
    #up_to_idx = len(modelspec) - 1 - up_to_idx
    #if up_to_idx is None:
    #    up_to_idx = first_substring_index(reversed(ms_modules), 'levelshift')
    #    # because reversed, need to mirror the idx
    #    if up_to_idx is not None:
    #        up_to_idx = len(modelspec) - 1 - up_to_idx
    #    else:
    #        up_to_idx = len(modelspec) - 1
    log.info('up_to_idx=%d (%s)', up_to_idx, modelspec[up_to_idx]['fn'])
    
    # do the +1 here to avoid adding to None
    up_to_idx += 1
    # exclude the following from the init
    exclude = ['rdt_gain', 'state_dc_gain', 'state_gain']
    freeze = ['stp']
    # more complex version of first_substring_index: checks for not membership in init_static_nl_layers
    init_idxes = [idx for idx, ms in enumerate(ms_modules[:up_to_idx]) if not any(sub in ms for sub in exclude)]

    freeze_idxes = []
    # make a temp modelspec
    temp_ms = mslib.ModelSpec()
    log.info('Creating temporary model for init with:')
    for idx in init_idxes:
        # TODO: handle 'merge_channels'
        ms = copy.deepcopy(modelspec[idx])
        log.info(f'{ms["fn"]}')

        # fix levelshift if present (will always be the last module)
        if idx == init_idxes[-1] and 'levelshift' in ms['fn']:
            output_name = modelspec.meta.get('output_name', 'resp')
            try:
                mean_resp = np.nanmean(est[output_name].as_continuous(), axis=1, keepdims=True)
            except NotImplementedError:
                # as_continuous only available for RasterizedSignal
                mean_resp = np.nanmean(est[output_name].rasterize().as_continuous(), axis=1, keepdims=True)
            if len(ms['phi']['level'][:]) == len(mean_resp):
               log.info(f'Fixing "{ms["fn"]}" to: {mean_resp.flatten()[0]:.3f}')
               ms['phi']['level'][:] = mean_resp

        temp_ms.append(ms)
        if any(fr in ms['fn'] for fr in freeze):
            freeze_idxes.append(len(temp_ms)-1)

    log.info('Running first init fit: model up to first lvl/relu without stp/gain.')
    log.debug('freeze_idxes: %s', freeze_idxes)
    filepath = Path(modelspec.meta['modelpath']) / 'init_part1'
    temp_ms = fit_tf(temp_ms, est, freeze_layers=freeze_idxes, filepath=filepath, **kwargs)['modelspec']

    # put back into original modelspec
    for ms_idx, temp_ms_module in zip(init_idxes, temp_ms):
        modelspec[ms_idx] = temp_ms_module

    if nl_init == 'scipy':
        # pre-fit static NL if it exists
        _d = init_static_nl(est=est, modelspec=modelspec)
        modelspec = _d['modelspec']
        # TODO : Initialize relu in some intelligent way?

        log.info('finished fit_tf_init, fit_idx=%d/%d', modelspec.fit_index + 1, modelspec.fit_count)
        return {'modelspec': modelspec}
    else:
        # init the static nl
        init_static_nl_mapping = {
            'double_exponential': initializers.init_dexp,
            'relu': None,
            'logistic_sigmoid': initializers.init_logsig,
            'saturated_rectifier': initializers.init_relsat,
        }
        # first find the first occurrence of a static nl in last two layers
        # if present, remove it from the idxes of the modules to freeze, init the nl and fit, and return the modelspec
        for idx, ms in enumerate(modelspec[-2:], len(modelspec)-2):
            for init_static_layer, init_fn in init_static_nl_mapping.items():
                if init_static_layer in ms['fn']:
                    log.info(f'Initializing static nl "{ms["fn"]}" at layer #{idx}')
                    # relu has a custom init
                    if init_static_layer == 'relu':
                        ms['phi']['offset'][:] = -0.1
                    else:
                        modelspec = init_fn(est, modelspec)

                    static_nl_idx_not = list(set(range(len(modelspec))) - set([idx]))
                    log.info('Running second init fit: all frozen but static nl.')

                    # don't overwrite the phis in the modelspec
                    kwargs['use_modelspec_init'] = True
                    filepath = Path(modelspec.meta['modelpath']) / 'init_part2'
                    return fit_tf(modelspec, est, freeze_layers=static_nl_idx_not,
                                  filepath=filepath, **kwargs)

    # no static nl to init
    return {'modelspec': modelspec}


def eval_tf_layer(data: np.ndarray,
                  layer_spec: typing.Union[None, str] = None,
                  state_data: np.ndarray = None,
                  stop: typing.Union[None, int] = None,
                  modelspec: mslib.ModelSpec = None,
                  **kwargs,  # temporary until state data is implemented
                  ) -> np.ndarray:
    """Takes in a numpy array and applies a single tf layer to it.

    :param data: The input data. Shape of (reps, time, channels).
    :param layer_spec: A layer spec for layers of a modelspec.
    :param state_data: The input state data. Shape of (reps, time, channels).
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

    data_shape = data.shape
    state_shape = None
    if state_data is not None:
        state_shape = state_data.shape
        data = [data, state_data]

    model = modelbuilder.ModelBuilder(layers=layers).build_model(input_shape=data_shape, state_shape=state_shape)

    pred = model.predict(data)
    return pred


# TODO: move this into tf.utils and whatever else needs to go there too
@tf.function
def get_jacobian(model: tf.keras.Model,
                 tensor: tf.Tensor,
                 index: int,
                 out_channel: int = 0,
                 ) -> np.array:
    """Gets the jacobian at the given index.

    This needs to be a tf.function for a huge speed increase."""
    with tf.GradientTape(persistent=True) as g:
        g.watch(tensor)
        z = model(tensor)[0, index, out_channel]

    w = g.jacobian(z, tensor)
    return w
