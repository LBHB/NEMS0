import numpy as np
import logging
log = logging.getLogger(__name__)

def reverse_correlation(rec, modelspec, input_name, tol=1e-5):
    """
    return a new modelspec with updated phi for fir coefficients and level shift
    """

    # apply mask, if it exists
    if 'mask' in rec.signals.keys():
        log.info("Data len pre-mask: %d", rec['mask'].shape[1])
        rec = rec.apply_mask()
        log.info("Data len post-mask: %d", rec['mask'].shape[1])

    # determine if doing forward or reverse model
    if input_name == 'stim':
        sig1 = 'stim'
        sig2 = 'resp'
    elif input_name == 'resp':
        sig1 = 'resp'
        sig2 = 'stim'

    # define "R" and "S" based on if doing forward or reverse model
    # mean center S.
    R = rec[sig1].as_continuous()
    S = rec[sig2].as_continuous()
    S_mean = S.mean(axis=-1)[:, np.newaxis]
    S = S - S_mean

    # create delay lines
    ndelays = modelspec[0]['phi']['coefficients'].shape[1]
    for i in range(0, ndelays):
        if input_name == 'stim':
            roll = i
        else:
            roll = -i
        if i == 0:
            r_roll = np.roll(R, roll, axis=-1)
            R_delay = r_roll
        else:
            r_roll = np.roll(R, roll, axis=-1)
            R_delay = np.concatenate((R_delay, r_roll), axis=0)

    # perform normalized reverse correlation
    Rs = np.matmul(S, R_delay.T)
    CC = np.matmul(R_delay, R_delay.T) 
    h = np.matmul(np.linalg.pinv(CC, tol), Rs.T)

    # predict the new stimulus
    S_hat = np.matmul(h.T, R_delay) + S_mean

    # reshape "phis" into the form that modelspecs expects and save
    d1 = R.shape[0]
    d2 = S.shape[0]
    h = h.reshape(ndelays, d1, d2).transpose(0, 2, 1).reshape(ndelays, d1*d2).T

    modelspec[0]['phi']['coefficients'] = h
    modelspec[1]['phi']['level'] = S_mean

    # add a field holding the prediction, for testing purposes
    modelspec[0]['pred'] = S_hat

    return modelspec
