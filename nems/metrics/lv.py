import numpy as np
import logging

log = logging.getLogger(__name__)


def cc_err(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred', 
           group_idx=None, group_cc=None, pcproj_std=None, pc_axes=None):
    '''
    Given the evaluated data, return the mean squared error for correlation coefficient computed
    separately for each group of the data (eg, passive vs. active or big vs. small pupil)

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    (these other parameters can be hard-coded into a partial function that is then passed onto the fitter:)
    pred_name : string
        Name of prediction in the result recording
    pred0_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording
    group_idx: list of array indexes defining which samples belong in which group
    group_cc: list of CC matrices, one for each group
    pcproj_std: std of projection onto first N pcs
    pc_axes: projection vectors for first N pcs--to project prediction and compute difference from
             actual pcproj_std

    Returns
    -------
    E : float
        Mean-squared difference between the CC matrix for each group

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = cc_err(result, 'pred', 'resp', <other parameters>)

    '''
    if type(result) is dict:
        pred=result[pred_name]
        pred0=result[pred0_name]
    else:
        pred = result[pred_name].as_continuous()
        pred0 = result[pred0_name].as_continuous()
    E = 0
    for idx,cc_act in zip(group_idx, group_cc):
       E += np.sum((np.cov(pred[:,idx] - pred0[:,idx])-cc_act)**2) / np.sum(cc_act**2)
    if pc_axes is not None:
        pcproj = (pred-pred0).T.dot(pc_axes.T).T
        pp_std = pcproj.std(axis=1)
        E += np.sum((pcproj_std-pp_std)**2)*10
    return E


def cc_err_md(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred', 
           group_idx=None, group_cc=None, pcproj_std=None, pc_axes=None):
    '''
    Given the evaluated data, return the mean squared error for correlation coefficient computed for special groupings of 
    the data. Mean of two adjacent groups and difference between two adjacent groups. Assumes that xx_sm and xx_lg are in 
    sequence, which is what ccnorm does by default

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    (these other parameters can be hard-coded into a partial function that is then passed onto the fitter:)
    pred_name : string
        Name of prediction in the result recording
    pred0_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording
    group_idx: list of array indexes defining which samples belong in which group
    group_cc: list of CC matrices, one for each group
    pcproj_std: std of projection onto first N pcs
    pc_axes: projection vectors for first N pcs--to project prediction and compute difference from
             actual pcproj_std

    Returns
    -------
    E : float
        Mean-squared difference between the CC matrix for each group

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = cc_err(result, 'pred', 'resp', <other parameters>)

    '''
    if type(result) is dict:
        pred=result[pred_name]
        pred0=result[pred0_name]
    else:
        pred = result[pred_name].as_continuous()
        pred0 = result[pred0_name].as_continuous()
    E = 0
    for idx in range(0,len(group_idx),2):
        idx2=idx+1
        ccact_m = group_cc[idx]+group_cc[idx2]
        ccact_d = group_cc[idx]-group_cc[idx2]
        c1 = np.cov(pred[:,group_idx[idx]] - pred0[:,group_idx[idx]])
        c2 = np.cov(pred[:,group_idx[idx2]] - pred0[:,group_idx[idx2]])
        cm = c1+c2
        cd = c1-c2
        E += (np.sum((cm-ccact_m)**2) + np.sum((cd-ccact_d)**2)) / np.sum(ccact_m**2)
        
    #for idx,cc_act in zip(group_idx, group_cc):
    #   E += np.sum((np.cov(pred[:,idx] - pred0[:,idx])-cc_act)**2) / np.sum(cc_act**2)
    if pc_axes is not None:
        pcproj = (pred-pred0).T.dot(pc_axes.T).T
        pp_std = pcproj.std(axis=1)
        E += np.sum((pcproj_std-pp_std)**2)*10
    return E



def cc_err_w(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred', 
             alpha=1, group_idx=None, group_cc=None, pcproj_std=None, pc_axes=None, verbose=False):
    '''
    Given the evaluated data, return the mean squared error for correlation coefficient computed
    separately for each group of the data (eg, passive vs. active or big vs. small pupil)

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    (these other parameters can be hard-coded into a partial function that is then passed onto the fitter:)
    pred_name : string
        Name of prediction in the result recording
    pred0_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording
    group_idx: list of array indexes defining which samples belong in which group
    group_cc: list of CC matrices, one for each group
    pcproj_std: std of projection onto first N pcs
    pc_axes: projection vectors for first N pcs--to project prediction and compute difference from
             actual pcproj_std
    alpha: how much to weigh diagonal vs. off-diagonal terms of cc matrix error. 
           alpha>1 weigh diag more, alpha<1 weigh diag less.
    
    Returns
    -------
    E : float
        Mean-squared difference between the CC matrix for each group

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = cc_err(result, 'pred', 'resp', <other parameters>)

    '''
    if type(result) is dict:
        pred=result[pred_name]
        pred0=result[pred0_name]
    else:
        pred = result[pred_name].as_continuous()
        pred0 = result[pred0_name].as_continuous()
    E = 0
    for i,idx,cc_act in zip(range(len(group_idx)),group_idx, group_cc):
        c1 = (np.cov(pred[:,idx] - pred0[:,idx])-cc_act)
        if alpha != 1:
            a = np.diagonal(c1)*alpha
            np.fill_diagonal(c1, a)
            if verbose:
                derr=np.sum(a**2) / np.sum(cc_act**2)
                oderr = np.sum(c1**2) / np.sum(cc_act**2) - derr
                log.info(f"   E {i}: diag: {derr:.5f} off-diag: {oderr:.5f}")
        E += np.sum(c1**2) / np.sum(cc_act**2)  # / (alpha**2)
    
    if pc_axes is not None:
        pcproj = (pred-pred0).T.dot(pc_axes.T).T
        pp_std = pcproj.std(axis=1)
        E += np.sum((pcproj_std-pp_std)**2)*10
    return E


def resp_cc_err(result, pred_name='pred_lv', resp_name='resp', pred0_name='pred',
           group_idx=None, group_cc=None, pcproj_std=None, pc_axes=None, beta=1):
    '''
    Given the evaluated data, return the mean squared error for correlation coefficient computed
    separately for each group of the data (eg, passive vs. active or big vs. small pupil)

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    (these other parameters can be hard-coded into a partial function that is then passed onto the fitter:)
    pred_name : string
        Name of prediction in the result recording
    pred0_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording
    group_idx: list of array indexes defining which samples belong in which group
    group_cc: list of CC matrices, one for each group
    pcproj_std: std of projection onto first N pcs
    pc_axes: projection vectors for first N pcs--to project prediction and compute difference from
             actual pcproj_std

    Returns
    -------
    E : float
        Mean-squared difference between the CC matrix for each group

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = cc_err(result, 'pred', 'resp', <other parameters>)

    '''
    if type(result) is dict:
        pred=result[pred_name]
        pred0=result[pred0_name]
    else:
        pred = result[pred_name].as_continuous()
        pred0 = result[pred0_name].as_continuous()
    E = np.mean((pred-result[resp_name].as_continuous())**2) / \
        np.mean(result[resp_name].as_continuous()**2)
    cc_count = len(group_cc) / beta
    for idx,cc_act in zip(group_idx, group_cc):
       E += np.sum((np.cov(pred[:,idx] - pred0[:,idx])-cc_act)**2) / np.sum(cc_act**2) / cc_count
    if pc_axes is not None:
        pcproj = (pred-pred0).T.dot(pc_axes.T).T
        pp_std = pcproj.std(axis=1)
        E += np.sum((pcproj_std-pp_std)**2)*10
    return E

def pup_dep_LV(result, pred_name='pred', resp_name='resp', **context):
    '''
    For purely LV model. Constrain first LV (lv_slow) to correlate with pupil,
    second LV (lv_fast) to have variance that correlates with pupil.
    Weigh these constraints vs. minimizing nsme.

    Will also work if only have one or the other of the two LVs
    '''
    result = result.apply_mask()
    lv_chans = result['lv'].chans
    X1 = result[pred_name].as_continuous()
    X2 = result[resp_name].as_continuous()
    respstd = np.nanstd(X2)
    squared_errors = (X1-X2)**2
    mse = np.sqrt(np.nanmean(squared_errors))
    nmse = mse / respstd

    alpha = context['alpha']
    signed_correlation = context.get('signed_correlation', False)

    if len(lv_chans) > 3:
        pass
        #log.info("WARNING: Not set up to handle greater than 2 LVs right now due to \
        #                complications with hyperparameter specification. Be aware that if you're not careful \
        #                with parameter spec here, sum of alpha could be > 1")

    if type(alpha) is dict:
        # passed different hyperparameters for each of the LVs
        fast_alpha = alpha['fast_alpha']
        slow_alpha = alpha['slow_alpha']

        if (fast_alpha + slow_alpha) > 1:
                raise ValueError("Hyperparameter values must sum to < 1")

    else:
        fast_alpha = slow_alpha = alpha

    if ('lv_fast' not in lv_chans) & ('lv_slow' not in lv_chans):
        # don't know how to constrain LV(s), just minimizing nmse
        return nmse
    
    elif ('lv_fast' in lv_chans) & ('lv_slow' in lv_chans):
        ref_len = result.meta['ref_len']
        p = result['pupil']._data.reshape(-1, ref_len)
        
        fast_lv_chans = [c for c in lv_chans if 'lv_fast' in c]
        fast_cc = []
        p = np.mean(p, axis=-1)
        for c in fast_lv_chans:
            lv_fast = result['lv'].extract_channels([c])._data.reshape(-1, ref_len)
            lv_fast = np.std(lv_fast, axis=-1)
            if signed_correlation:
                cc = lv_corr_pupil(p, lv_fast)
            else:
                cc = -abs(lv_corr_pupil(p, lv_fast))
            fast_cc.append(cc)


        p = result['pupil']._data
        lv_slow = result['lv'].extract_channels(['lv_slow'])._data
        slow_cc = -abs(lv_corr_pupil(p, lv_slow))

        cost = (slow_alpha * slow_cc) + ((1 - (slow_alpha + fast_alpha)) * nmse)

        for i, c in enumerate(fast_lv_chans):
            cost += (fast_alpha * fast_cc[i])

        return cost

    elif ('lv_fast' in lv_chans):
        ref_len = result.meta['ref_len']
        p = result['pupil']._data.reshape(-1, ref_len)
        lv_fast = result['lv'].extract_channels(['lv_fast'])._data.reshape(-1, ref_len)
        
        p = np.mean(p, axis=-1)
        lv_fast = np.std(lv_fast, axis=-1)
        if signed_correlation:
            fast_cc = lv_corr_pupil(p, lv_fast)
        else:
            fast_cc = -abs(lv_corr_pupil(p, lv_fast))

        if np.sum(lv_fast)==0:
            fast_cc = 0

        cost = (fast_alpha * fast_cc) + ((1 - fast_alpha) * nmse)
        return cost

    elif ('lv_slow' in lv_chans):
        p = result['pupil']._data
        lv_slow = result['lv'].extract_channels(['lv_slow'])._data
        slow_cc = -abs(lv_corr_pupil(p, lv_slow))

        cost = (slow_alpha * slow_cc) + ((1 - slow_alpha) * nmse)
        return cost

def lv_corr_pupil(p, lv):
    """
    return correlation of pupil and lv
    """
    return np.corrcoef(p.squeeze(), lv.squeeze())[0, 1]
