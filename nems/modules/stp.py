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
    
    s = X.shape
    
    tstim=X.copy()
    tstim[np.isnan(tstim)]=0
    tstim[tstim<0]=0
    # ui=self.u
    # force only depression, no facilitation
    ui = np.absolute(u)

    # convert tau units from sec to bins
    #taui = np.absolute(self.tau[:, j]) * self.d_in[0]['fs']
    taui = np.absolute(tau)

    # TODO : enable crosstalk

    # TODO : allow >1 STP channel per input?
    
    di = np.ones(s)
    
    # go through each stimulus channel
    for i in range(0, s[0]):

        # limits, assumes input (X) range is approximately -1 to +1
        # TODO: move bounds to fitter?
        if ui[i] > 0.5:
            ui[i] = 0.5
        elif ui[i] < -0.5:
            ui[i] = -0.5
        if taui[i] < 0.5:
            taui[i] = 0.5

        for tt in range(1, s[1]):
            td = di[i, tt - 1]  # previous time bin depression
            if ui[i] > 0:
                # facilitation
                delta = (1 - td) / taui[i] - \
                    ui[i] * td * tstim[i, tt - 1]
                td = td + delta
                td = np.max([td,0])
            else:
                #depression
                delta = (1 - td) / taui[i] - \
                    ui[i] * td * tstim[i, tt - 1]
                td = td + delta
                td = np.min([td,1])
            di[i, tt] = td
    #print("(u,tau)=({0},{1})".format(u,tau))
    return di * X





