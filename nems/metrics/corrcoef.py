# -*- coding: utf-8 -*-
import numpy as np
import scipy.special
import scipy.stats as stats
import nems.epoch as ep

import logging
log = logging.getLogger(__name__)


def corrcoef(result, pred_name='pred', resp_name='resp'):
    '''
    Given the evaluated data, return the mean squared error

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    pred_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording

    Returns
    -------
    cc : float
        Correlation coefficient between the prediction and response.

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> cc = corrcoef(result, 'pred', 'resp')

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing. (@bburan TODO: Is this still true?)
    '''
    pred = result[pred_name].as_continuous()
    resp = result[resp_name].as_continuous()
    if pred.shape[0] > 1:
        raise ValueError("multi-channel signals not supported yet.")

    ff = np.isfinite(pred) & np.isfinite(resp)
    if (np.sum(ff) == 0) or (np.sum(pred[ff]) == 0) or (np.sum(resp[ff]) == 0):
        return 0
    else:
        cc = np.corrcoef(pred[ff], resp[ff])
        return cc[0, 1]


def j_corrcoef(result, pred_name='pred', resp_name='resp', njacks=20):
    '''
    Jackknifed estimate of mean and SE on correlation coefficient

    Parameters
    ----------
    result : A Recording object
        Generally the output of `model.evaluate(phi, data)`
    pred_name : string
        Name of prediction in the result recording
    resp_name : string
        Name of response in the result recording

    Returns
    -------
    cc : float
        Correlation coefficient between the prediction and response.
    ee : float
        Standard error on cc, calculated by jackknife (Efron & Tibshirani 1986)

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> cc,ee = j_corrcoef(result, 'pred', 'resp', njacks=10)

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing. (@bburan TODO: Is this still true?)
    '''

    predmat = result[pred_name].as_continuous()
    respmat = result[resp_name].as_continuous()

    channel_count = predmat.shape[0]
    cc = np.zeros(channel_count)
    ee = np.zeros(channel_count)

    for i in range(channel_count):
        pred = predmat[i, :]
        resp = respmat[i, :]
        ff = np.isfinite(pred) & np.isfinite(resp)

        if (np.sum(ff) == 0) or (np.sum(pred[ff]) == 0) or (np.sum(resp[ff]) == 0):
            cc[i] = 0
            ee[i] = 0
        else:
            pred = pred[ff]
            resp = resp[ff]
            chunksize = int(np.ceil(len(pred) / njacks / 10))
            chunkcount = int(np.ceil(len(pred) / chunksize / njacks))
            idx = np.zeros((chunkcount, njacks, chunksize))
            for jj in range(njacks):
                idx[:, jj, :] = jj
            idx = np.reshape(idx, [-1])[:len(pred)]
            jc = np.zeros(njacks)
            for jj in range(njacks):
                ff = (idx != jj)
                jc[jj] = np.corrcoef(pred[ff], resp[ff])[0, 1]

            cc[i] = np.nanmean(jc)
            ee[i] = np.nanstd(jc) * np.sqrt(njacks-1)

    return cc, ee


def r_floor(result, pred_name='pred', resp_name='resp'):
    '''
    corr coef floor based on shuffled responses
    '''
    # if running validation test, also measure r_floor
    X1mat = result[pred_name].as_continuous()
    X2mat = result[resp_name].as_continuous()
    channel_count = X2mat.shape[0]
    r_floor = np.zeros(channel_count)

    for i in range(channel_count):
        X1 = X1mat[i, :]
        X2 = X2mat[i, :]

        # remove all nans from pred and resp
        ff = np.isfinite(X1) & np.isfinite(X2)
        X1=X1[ff]
        X2=X2[ff]

        # figure out how many samples to use in each shuffle
        if len(X1)>500:
            n=500
        else:
            n=len(X1)

        # compute cc for 1000 shuffles
        rf = np.zeros([1000, 1])
        for rr in range(0, len(rf)):
            n1 = (np.random.rand(n) * len(X1)).astype(int)
            n2 = (np.random.rand(n) * len(X2)).astype(int)
            rf[rr] = np.corrcoef(X1[n1], X2[n2])[0, 1]

        rf = np.sort(rf[np.isfinite(rf)], 0)
        if len(rf):
            r_floor[i] = rf[np.int(len(rf) * 0.95)]
        else:
            r_floor[i] = 0

    return r_floor


def _r_single(X, N=100):
    """
    Assume X is trials X time raster

    test data from SPN recording
    X=rec['resp'].extract_epoch('STIM_BNB+si464+si1889')
    """

    if X.shape[1] > 1:
        raise ValueError("multi-channel signals not supported yet.")

    repcount = X.shape[0]
    if repcount <= 1:
        log.info('repcount<=1, rnorm=0')
        return 0

    paircount = np.int(scipy.special.comb(repcount, 2))
    pairs = []
    for p1 in range(repcount):
        for p2 in range(p1+1, repcount):
            pairs.append([p1, p2])

    if paircount < N:
        N = paircount

    if N == 1:
        # only two repeats, break up data in time to get a better
        # estimate of single-trial correlations
        # raise ValueError("2 repeats condition not supported yet.")
        # N=10;
        # bstep=size(pred,1)./N;
        # rac=zeros(N,1);
        # for nn=1:N,
        #     tt=round((nn-1).*bstep+1):round(nn*bstep);
        #     if ~isempty(tt) && std(resp(tt,1))>0 && std(resp(tt,2))>0,
        #         rac(nn)=xcov(resp(tt,1),resp(tt,2),0,'coeff');
        #     end
        # end
        print('r_ceiling invalid')
        return 0.05
    else:

        rac = np.zeros(N)
        sidx = np.argsort(np.random.rand(paircount))
        for nn in range(N):
            X1 = X[pairs[sidx[nn]][0], 0, :]
            X2 = X[pairs[sidx[nn]][1], 0, :]

            # remove all nans from pred and resp
            ff = np.isfinite(X1) & np.isfinite(X2)
            X1 = X1[ff]
            X2 = X2[ff]
            if (np.sum(X1) > 0) and (np.sum(X2) > 0):
                rac[nn] = np.corrcoef(X1, X2)[0, 1]
            else:
                rac[nn] = 0

    # hard limit on single-trial correlation to prevent explosion
    # TODO: better logic for this
    rac = np.mean(rac)
    if rac < 0.05:
        rac = 0.05

    return rac


def r_ceiling(result, fullrec, pred_name='pred', resp_name='resp', N=100):
    """
    Compute noise-corrected correlation coefficient based on single-trial
    correlations in the actual response.
    """
    if fullrec[resp_name].shape[0] > 1:
        log.info('multi-channel data not supported in r_ceiling. returning 0')
        return 0

    epoch_regex = '^STIM_'
    epochs_to_extract = ep.epoch_names_matching(result[resp_name].epochs,
                                                epoch_regex)
    folded_resp = result[resp_name].extract_epochs(epochs_to_extract)

    epochs_to_extract = ep.epoch_names_matching(result[pred_name].epochs,
                                                epoch_regex)
    folded_pred = result[pred_name].extract_epochs(epochs_to_extract)

    resp = fullrec[resp_name].rasterize()
    rnorm_c = 0
    n = 0

    for k, d in folded_resp.items():
        if np.sum(np.isfinite(d)) > 0:

            X = resp.extract_epoch(k)
            rac = _r_single(X, N)

            # print("{0} shape: {1},{2}".format(k,X.shape[0],X.shape[2]))
            # print(rac)

            if rac > 0:
                p = folded_pred[k]

                repcount = X.shape[0]
                rs = np.zeros(repcount)
                for nn in range(repcount):
                    X1 = X[nn, 0, :]
                    X2 = p[0, 0, :]

                    # remove all nans from pred and resp
                    ff = np.isfinite(X1) & np.isfinite(X2)
                    X1 = X1[ff]
                    X2 = X2[ff]

                    if (np.sum(X1) > 0) and (np.sum(X2) > 0):
                        rs[nn] = np.corrcoef(X1, X2)[0, 1]
                    else:
                        rs[nn] = 0

                rs = np.mean(rs)

                rnorm_c += (rs / np.sqrt(rac)) * X1.shape[-1]
                n += X1.shape[-1]

                # print(rnorm_c)
                # print(n)

    # weighted average based on number of samples in each epoch
    if n > 0:
        rnorm = rnorm_c / n
    else:
        rnorm = rnorm_c

    return rnorm


def r_ceiling_test(result, fullrec, pred_name='pred',
              resp_name='resp', N=100):

    """
    Compute noise-corrected correlation coefficient based on single-trial
    correlations in the actual response.
    """
    epoch_regex = '^STIM_'
    epochs_to_extract = ep.epoch_names_matching(result[resp_name].epochs,
                                                epoch_regex)
    folded_resp = result[resp_name].extract_epochs(epochs_to_extract)

    epochs_to_extract = ep.epoch_names_matching(result[pred_name].epochs,
                                                epoch_regex)
    folded_pred = result[pred_name].extract_epochs(epochs_to_extract)

    resp = fullrec[resp_name].rasterize()

    X = np.array([])
    Y = np.array([])
    for k, d in folded_resp.items():
        if np.sum(np.isfinite(d)) > 0:

            x = resp.extract_epoch(k)
            X = np.concatenate((X, x.flatten()))

            p = folded_pred[k]
            if p.shape[0] < x.shape[0]:
                p = np.tile(p, (x.shape[0], 1, 1))
            Y = np.concatenate((Y, p.flatten()))

    # exclude nan values of X or Y
    gidx = (np.isfinite(X) & np.isfinite(Y))
    X = X[gidx]
    Y = Y[gidx]

    sx = X
    mx = X

    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(X)

    mu = [np.reshape(stats.gamma.rvs(
            fit_alpha + sx, loc=fit_loc,
            scale=fit_beta / (1 + fit_beta)), (1, -1))
          for a in range(10)]
    mu = np.concatenate(mu)
    mu[mu > np.max(X)] = np.max(X)
    xc_set = [np.corrcoef(mu[i, :], X)[0, 1] for i in range(10)]
    log.info("Simulated r_single: %.3f +/- %.3f",
             np.mean(xc_set), np.std(xc_set)/np.sqrt(10))

    xc_act = np.corrcoef(Y, X)[0, 1]
    log.info("actual r_single: %.03f", xc_act)

    # weighted average based on number of samples in each epoch
    rnorm = xc_act / np.mean(xc_set)

    return rnorm
