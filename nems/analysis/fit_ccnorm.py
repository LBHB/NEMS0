import copy
import logging
import time
import typing
from functools import partial
import re

import numpy as np

from nems.analysis.cost_functions import basic_cost
from nems.fitters.api import scipy_minimize
import nems.priors
import nems.fitters.mappers
import nems.modelspec as ms
import nems.metrics.api as metrics
import nems.segmentors
import nems.utils
import nems.recording as recording


log = logging.getLogger(__name__)


def fit_ccnorm(modelspec,
        est: recording.Recording,
        metric=None,
        use_modelspec_init: bool = True,
        optimizer: str = 'adam',
        max_iter: int = 10000,
        early_stopping_steps: int = 5,
        tolerance: float = 5e-4,
        learning_rate: float = 1e-4,
        batch_size: typing.Union[None, int] = None,
        seed: int = 0,
        initializer: str = 'random_normal',
        freeze_layers: typing.Union[None, list] = None,
        epoch_name: str = "REFERENCE",
        **context):
    '''
    Required Arguments:
     est          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     <copied from fit_tf for now

    Returns
     dictionary: {'modelspec': updated_modelspec}
    '''

    # Hard-coded
    cost_function = basic_cost
    fitter=scipy_minimize
    segmentor=nems.segmentors.use_all_data
    mapper=nems.fitters.mappers.simple_vector
    fit_kwargs = {'tolerance': tolerance, 'max_iter': max_iter}

    start_time = time.time()

    modelspec = copy.deepcopy(modelspec)

    # apply mask to remove invalid portions of signals and allow fit to
    # only evaluate the model on the valid portion of the signals
    if 'mask' in est.signals.keys():
        log.info("Data len pre-mask: %d", est['mask'].shape[1])
        est = est.apply_mask()
        log.info("Data len post-mask: %d", est['mask'].shape[1])

    conditions = ["_".join(k.split("_")[1:]) for k in est.signals.keys() if k.startswith("mask_")]
    if (len(conditions)>2) and any([c.split("_")[-1]=='lg' for c in conditions]):
        conditions.remove("small")
        conditions.remove("large")
        
    #conditions = ['large','small']
    group_idx = [est['mask_'+c].as_continuous()[0,:] for c in conditions]
    cg_filtered = [(c, g) for c,g in zip(conditions,group_idx) if g.sum()>0]
    conditions, group_idx = zip(*cg_filtered)
    
    for c,g in zip(conditions, group_idx):
        log.info(f"cc data for {c} len {g.sum()}")
        
    resp = est['resp'].as_continuous()
    pred0 = est['pred0'].as_continuous()

    group_cc = [np.cov(resp[:,idx]-pred0[:,idx]) for idx in group_idx]


    # Computing PCs
    from sklearn.decomposition import PCA
    epoch_regex = "^STIM_"
    stims = (est.epochs['name'].value_counts() >= 8)
    stims = [stims.index[i] for i, s in enumerate(stims) 
             if bool(re.search(epoch_regex, stims.index[i])) and s == True]

    # can't simply extract evoked for refs because can be longer/shorted if it came after target 
    # and / or if it was the last stim. So, masking prestim / postim doesn't work. Do it manually
    d = est['resp'].extract_epochs(stims, mask=est['mask'])

    R = [v.mean(axis=0) for (k, v) in d.items()]
    #R = [np.reshape(np.transpose(v,[1,0,2]),[v.shape[1],-1]) for (k, v) in d.items()]
    Rall_u = np.hstack(R).T

    pca = PCA(n_components=2)
    pca.fit(Rall_u)
    pc_axes = pca.components_

    # variance of projection onto PCs
    pcproj0 = (resp-pred0).T.dot(pc_axes.T).T
    pcproj_std = pcproj0.std(axis=1)
    
    if metric is None:
        #def cc_err(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred',
        #   group_idx=None, group_cc=None, pcproj_std=None, pc_axes=None):

        #metric = lambda d: metrics.cc_err(d, pred_name='pred', pred0_name='pred0',
        #                                  group_idx=group_idx, group_cc=group_cc, 
        #                                  pcproj_std=pcproj_std, pc_axes=pc_axes)
        metric = lambda d: metrics.cc_err(d, pred_name='pred', pred0_name='pred0',
                                          group_idx=group_idx, group_cc=group_cc, 
                                          pcproj_std=None, pc_axes=None)

    # turn on "fit mode". currently this serves one purpose, for normalization
    # parameters to be re-fit for the output of each module that uses
    # normalization. does nothing if normalization is not being used.
    ms.fit_mode_on(modelspec, est)

    # Create the mapper functions that translates to and from modelspecs.
    # It has three functions that, when defined as mathematical functions, are:
    #    .pack(modelspec) -> fitspace_point
    #    .unpack(fitspace_point) -> modelspec
    #    .bounds(modelspec) -> fitspace_bounds
    packer, unpacker, pack_bounds = mapper(modelspec)

    # A function to evaluate the modelspec on the data
    evaluator = nems.modelspec.evaluate

    my_cost_function = cost_function
    my_cost_function.counter = 0

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(my_cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data=est, segmentor=segmentor, evaluator=evaluator,
                      metric=metric, display_N=1000)

    # get initial sigma value representing some point in the fit space,
    # and corresponding bounds for each value
    sigma = packer(modelspec)
    bounds = pack_bounds(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn, bounds=bounds, **fit_kwargs)
    improved_modelspec = unpacker(improved_sigma)
    elapsed_time = (time.time() - start_time)

    start_err = cost_fn(sigma)
    final_err = cost_fn(improved_sigma)
    log.info("Delta error: %.06f - %.06f = %e", start_err, final_err, final_err-start_err)

    # TODO: Should this maybe be moved to a higher level
    # so it applies to ALL the fittters?
    ms.fit_mode_off(improved_modelspec)
    ms.set_modelspec_metadata(improved_modelspec, 'fitter', 'ccnorm')
    ms.set_modelspec_metadata(improved_modelspec, 'fit_time', elapsed_time)
    ms.set_modelspec_metadata(improved_modelspec, 'n_parms', len(improved_sigma))

    return {'modelspec': improved_modelspec.copy(),
            'save_context': True}


