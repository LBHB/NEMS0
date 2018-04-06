import numpy as np
from numpy import exp

def short_term_plasticity(rec, i, o, u, tau, crosstalk=0):
    '''
    STP applied to each input channel.
    parameterized by Markram-Todyks model:
        u (release probability)
        tau (recovery time constant)
    '''

    fn = lambda x : _stp(x, u, tau, crosstalk)

    return [rec[i].transform(fn, o)]


def _stp(X, u, tau, crosstalk=0):
    """
    STP core function
    """
    s = X.shape
    tstim=X.copy()
    tstim[np.isnan(tstim)]=0
    tstim[tstim<0]=0

    # ui=self.u
    # force only depression, no facilitation
    # TODO: move bounds to fitter?
    # limits, assumes input (X) range is approximately -1 to +1
    ui = np.absolute(u) / 100
    ui[ui>0.5] = 0.5

    # convert tau units from sec to bins
    #taui = np.absolute(self.tau[:, j]) * self.d_in[0]['fs']
    taui = np.absolute(tau)
    taui[taui<1] = 1

    # TODO : enable crosstalk
    # TODO : allow >1 STP channel per input?

    # go through each stimulus channel
    stim_out = tstim  # allocate scaling term
    for i in range(0, s[0]):
        td = 1  # initialize, dep state of previous time bin
        a = 1/taui[i]
        ustim = 1.0/taui[i] + ui[i] * tstim[i, :]
        # ustim = ui[i] * tstim[i, :]
        if ui[i] > 0:
            # depression
            for tt in range(1, s[1]):
                #td = di[i, tt - 1]  # previous time bin depression
                #delta = (1 - td) / taui[i] - ui[i] * td * tstim[i, tt - 1]
                #delta = (1 - td) / taui[i] - td * ustim[tt - 1]
                delta = a - td * ustim[tt - 1]
                td = td + delta
                td = np.max([td,0])
                stim_out[i, tt] *= td
        else:
            # facilitation
            for tt in range(1, s[1]):
                delta = a - td * ustim[tt - 1]
                td = td + delta
                td = np.min([td,1])
                stim_out[i, tt] *= td
    #print("(u,tau)=({0},{1})".format(ui,taui))

    return stim_out





