import numpy as np
from numpy import exp

def short_term_plasticity(rec, i, o, u, tau, crosstalk=0):
    '''
    STP applied to each input channel.
    parameterized by Markram-Todyks model:
        u (release probability)
        tau (recovery time constant)
    '''
    fn = lambda x : _stp(x, u, tau, crosstalk, rec[i].fs)

    return [rec[i].transform(fn, o)]


def _stp(X, u, tau, crosstalk=0, fs=1):
    """
    STP core function
    """
    s = X.shape
    tstim = X.copy()
    tstim[np.isnan(tstim)] = 0
    tstim[tstim < 0] = 0

    # TODO: deal with upper and lower bounds on dep and facilitation parms
    #       need to know something about magnitude of inputs???

    # TODO: move bounds to fitter? slow

    # limits, assumes input (X) range is approximately -1 to +1
    ui = u
    ui[ui > 2] = 2
    ui[ui < -0.5] = -0.5

    # convert tau units from sec to bins
    taui = np.absolute(tau) * fs
    taui[taui < 3] = 3

    # TODO : enable crosstalk
    if crosstalk:
        raise ValueError('crosstalk not yet supported')

    # TODO : allow >1 STP channel per input?

    # go through each stimulus channel
    stim_out = tstim  # allocate scaling term
    for i in range(0, s[0]):
        td = 1  # initialize, dep state of previous time bin
        a = 1/taui[i]
        ustim = 1.0/taui[i] + ui[i] * tstim[i, :]
        # ustim = ui[i] * tstim[i, :]
        if ui[i] == 0:
            # passthru, no STP, preserve stim_out = tstim
            pass
        elif ui[i] > 0:
            # depression
            for tt in range(1, s[1]):
                # td = di[i, tt - 1]  # previous time bin depression
                # delta = (1 - td) / taui[i] - ui[i] * td * tstim[i, tt - 1]
                # delta = 1/taui[i] - td * (1/taui[i] - ui[i] * tstim[i, tt - 1])
                # then a=1/taui[i] and ustim=1/taui[i] - ui[i] * tstim[i,:]
                delta = a - td * ustim[tt - 1]
                if td + delta > 0:
                    td = td + delta
                else:
                    td = 0
                # td = np.max([td, 0])
                stim_out[i, tt] *= td
        else:
            # facilitation
            for tt in range(1, s[1]):
                delta = a - td * ustim[tt - 1]
                td = td + delta
                # td = np.min([td, 1])
                stim_out[i, tt] *= td
    # print("(u,tau)=({0},{1})".format(ui,taui))

    stim_out[np.isnan(X)] = np.nan
    return stim_out
