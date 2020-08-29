import logging

import numpy as np
import scipy.stats as st
from scipy import linalg
import pandas as pd

import nems.db as nd
import nems.xform_helper as xhelp

log = logging.getLogger(__name__)


# Basically just a standardized wrapper for np.corrcoef
def equivalence_correlation(psth1, psth2):
    '''Computes correlation-based equivalence of two model predictions.'''
    psth1 = psth1.flatten()
    psth2 = psth2.flatten()
    return np.corrcoef(psth1, psth2)[0,1]


def equivalence_partial(psth1, psth2, baseline):
    '''Computes partial correlation-based equivalence of two model predictions.

    Parameters:
    ----------
    psth1: np.ndarray
        First model prediction, will be converted to (1, time) array.
    psth2: np.ndarray
        Second model prediction.
    baseline: np.ndarray
        Model prediction to use as baseline - partial correlation will be
        computed between psth1 and psth2 relative to this variable.

    Returns:
    -------
    p: float
        Partial correlation between psth1 and psth2 relative to baseline.

    '''

    # psths may or may not have extra dim already, so flatten and then
    # add dim again to make sure they all match.
    psth1 = psth1.flatten()
    psth2 = psth2.flatten()
    baseline = baseline.flatten()
    C = np.hstack((np.expand_dims(psth1, 0).transpose(),
                   np.expand_dims(psth2, 0).transpose(),
                   np.expand_dims(baseline, 0).transpose()))
    p = partial_corr(C)[0,1]

    return p


# https://gist.github.com/fabianp/9396204419c7b638d38f
def partial_corr(C):
    """
    Returns the sample linear partial correlation coefficients between pairs of
    variables in C, controlling for the remaining variables in C.

    Parameters
    ----------
    C : array-like, shape (n, p)
        Each column of C is taken as a variable

    Returns
    -------
    P : array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j]
        controlling for the remaining variables in C.

    """

    C = np.asarray(C)
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)
    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i+1, p):
            idx = np.ones(p, dtype=np.bool)
            idx[i] = False
            idx[j] = False
            beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            res_j = C[:, j] - C[:, idx].dot( beta_i)
            res_i = C[:, i] - C[:, idx].dot(beta_j)

            corr = st.pearsonr(res_i, res_j)[0]
            P_corr[i, j] = corr
            P_corr[j, i] = corr

    return P_corr


def equivalence_partial_batch(batch, model1, model2, baseline,
                              manual_cellids=None,
                              performance_stat='r_ceiling', load_path=None,
                              save_path=None):
    '''
    Computes equivalence and effect size for all cells in a batch.

    Equivalence is computed as the partial correlation between model1 and
    model2 relative to baseline. Effect size is computed as the average
    improvement in prediction correlation relative to baseline.

    Note: this process can take several hours. It is strongly recommended
    that you specify a `save_path` on the first run so that the results can be
    quickly loaded in the future.

    Parameters:
    -----------
    batch: int
        CellDB batch number.
    model1, model2, baseline: str
        Model names indicating which models should be compared.
    manual_cellids: list
        List of cellids to use. By default, all cellids for the given
        batch will be used.
    performance_stat: str
        One of 'r_test', 'r_fit', or 'r_ceiling' to indicate which statistic
        should be used for computing effect size.
    load_path: str
        Filepath where pandas dataframe of the results should be stored,
        will be saved as a Python pickle file (.pkl).
    save_path: str
        Filepath to load a pickled dataframe from.

    Returns:
    --------
    df: Pandas dataframe
        Indexed by cellid with columns: 'partial_corr' (equivalence) and
        'performance_effect' (effect size).

    '''

    if manual_cellids is not None:
        cellids = nd.get_batch_cells(batch=batch, as_list=True)

    if load_path is None:
        partials = []
        models = [model1, model2, baseline]
        for cell in cellids:
            try:
                pred1, pred2, baseline_pred = load_models(cell, batch, models)
            except ValueError as e:
                # Missing result
                log.warning('Missing result: \n%s', e)

            partial = equivalence_partial(pred1, pred2, baseline_pred)
            partials.append(partial)

        df = nd.batch_comp(batch=batch, modelnames=models, stat=performance_stat)
        stats = [df[model].values for model in models]
        rel_1 = stats[0] - stats[2]
        rel_2 = stats[1] - stats[2]
        effect_sizes = 0.5*(rel_1 + rel_2)

        results = {'cellid': cellids, 'partial_corr': partials,
                   'performance_effect': effect_sizes}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    return df


def equivalence_corr_batch(batch, model1, model2, manual_cellids=None,
                           load_path=None, save_path=None):
    '''Computes correlation-based equivalence for all cells in a batch.

    Parameters:
    ----------
    batch: int
        CellDB batch number.
    model1, model2: str
        Model names indicating which models should be compared.
    manual_cellids: list
        List of cellids to use. By default, all cellids for the given
        batch will be used.
    load_path: str
        Filepath where pandas dataframe of the results should be stored,
        will be saved as a Python pickle file (.pkl).
    save_path: str
        Filepath to load a pickled dataframe from.

    Returns:
    --------
    df: Pandas dataframe
        Indexed by cellid with column: 'correlation' (equivalence).

    '''

    if manual_cellids is not None:
        cellids = nd.get_batch_cells(batch=batch, as_list=True)

    if load_path is None:
        corrs = []
        models = [model1, model2]
        for cell in cellids:
            try:
                pred1, pred2 = load_models(cell, batch, models)
            except ValueError as e:
                # Missing result
                log.warning('Missing result: \n%s', e)

            corr = equivalence_correlation(pred1, pred2)
            corrs.append(corr)

        results = {'cellid': cellids, 'correlation': corrs}
        df = pd.DataFrame.from_dict(results)
        df.set_index('cellid', inplace=True)
        if save_path is not None:
            df.to_pickle(save_path)
    else:
        df = pd.read_pickle(load_path)

    return df


def load_models(cell, batch, models):
    '''Load standardized psth from each model, error if not all fits exist.'''

    ctxs = []
    for model in models:
        xf, ctx = xhelp.load_model_xform(cell, batch, model)
        ctxs.append(ctx)

    preds = [ctx['val'].apply_mask()['pred'].as_continuous() for ctx in ctxs]
    ff = np.isfinite(preds[0])
    for pred in preds[1:]:
        ff &= np.isfinite(pred)
    no_nans = [pred[ff] for pred in preds]

    return no_nans


def within_model_equivalence(batch, model1, model2, model1_h1, model1_h2,
                             model2_h1, model2_h2, baseline_h1, baseline_h2,
                             manual_cellids=None,
                             save_path=None, load_path=None,
                             within_save_path1=None, within_save_path2=None,
                             cross_save_path1=None, cross_save_path2=None,
                             within_load_path1=None, within_load_path2=None,
                             cross_load_path1=None, cross_load_path2=None):

    between_df = equivalence_corr_batch(batch, model1, model2,
                                        manual_cellids=manual_cellids,
                                        save_path=save_path, load_path=load_path)

    within_df1 = equivalence_corr_batch(batch, model1_h1, model1_h2,
                                        manual_cellids=manual_cellids,
                                        save_path=within_save_path1,
                                        load_path=within_load_path1)
    within_df2 = equivalence_corr_batch(batch, model2_h1, model2_h2,
                                        manual_cellids=manual_cellids,
                                        save_path=within_save_path2,
                                        load_path=within_load_path2)

    cross_df1 = equivalence_corr_batch(batch, model1_h1, model2_h2,
                                         manual_cellids=manual_cellids,
                                         save_path=cross_save_path1,
                                         load_path=cross_load_path1)
    cross_df2 = equivalence_corr_batch(batch, model1_h2, model2_h1,
                                         manual_cellids=manual_cellids,
                                         save_path=cross_save_path2,
                                         load_path=cross_load_path2)

    return between_df, within_df1, within_df2, cross_df1, cross_df2
