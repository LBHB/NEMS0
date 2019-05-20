import numpy as np
from numpy import exp
from scipy.integrate import cumtrapz
from scipy.signal import boxcar

def short_term_plasticity(rec, i, o, u, tau, x0=None, crosstalk=0,
                          reset_signal=None, quick_eval=False):
    '''
    STP applied to each input channel.
    parameterized by Markram-Todyks model:
        u (release probability)
        tau (recovery time constant)
    '''

    r = None
    if reset_signal is not None:
        if reset_signal in rec.signals.keys():
            r = rec[reset_signal].as_continuous()

    fn = lambda x : _stp(x, u, tau, x0, crosstalk, rec[i].fs, r, quick_eval)

    return [rec[i].transform(fn, o)]


def _stp(X, u, tau, x0=None, crosstalk=0, fs=1, reset_signal=None, quick_eval=False):
    """
    STP core function
    """
    s = X.shape
    tstim = X.copy()
    tstim[np.isnan(tstim)] = 0
    if x0 is not None:
        tstim -= np.expand_dims(x0, axis=1)

    tstim[tstim < 0] = 0

    # TODO: deal with upper and lower bounds on dep and facilitation parms
    #       need to know something about magnitude of inputs???
    #       move bounds to fitter? slow

    # limits, assumes input (X) range is approximately -1 to +1
    ui = np.abs(u.copy())

    #ui[ui > 1] = 1
    #ui[ui < -0.4] = -0.4

    # convert tau units from sec to bins
    taui = np.absolute(tau.copy()) * fs
    taui[taui < 2] = 2

    # avoid ringing if combination of strong depression and
    # rapid recovery is too large
    rat = ui**2 / taui
    ui[rat>0.1] = np.sqrt(0.1 * taui[rat>0.1])
    #taui[rat>0.08] = (ui[rat>0.08]**2) / 0.08

    #print("rat: %s" % (ui**2 / taui))

    # TODO : allow >1 STP channel per input?

    # go through each stimulus channel
    stim_out = tstim  # allocate scaling term

    if crosstalk:
        # assumes dim of u is 1 !
        tstim = np.mean(tstim, axis=0, keepdims=True)

    for i in range(0, len(u)):
        if quick_eval and (reset_signal is not None):

            a = 1 / taui[i]
            x = ui[i] * tstim[i, :] / fs

            if reset_signal is None:
                reset_times = np.array([0, len(x)])
            else:
                reset_times = np.argwhere(reset_signal[0, :])[:, 0]
                reset_times = np.append(reset_times, len(x))

            mu = np.zeros_like(x)
            imu = np.zeros_like(x)
            for j in range(len(reset_times)-1):
                si = slice(reset_times[j], reset_times[j+1])

                ix = cumtrapz(a + x[si], dx=1, initial=0)

                mu[si] = np.exp(ix)
                imu[si] = cumtrapz(mu[si]*x[si], dx=1, initial=0)

            td = np.ones_like(x)
            ff = np.bitwise_and(mu>0, imu>0)
            td[ff] = 1 - np.exp(np.log(imu[ff]) - np.log(mu[ff]))
            #td[mu>0] = 1 - imu[mu>0]/mu[mu>0]

            if crosstalk:
                stimout *= np.expand_dims(td, 0)
            else:
                stim_out[i, :] *= td
        else:

            a = 1/taui[i]
            ustim = 1.0/taui[i] + ui[i] * tstim[i, :]
            # ustim = ui[i] * tstim[i, :]
            td = np.ones_like(ustim)  # initialize dep state vector

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
                    delta = a - td[tt - 1] * ustim[tt - 1]
                    td[tt] = td[tt-1] + delta
                    if td[tt] < 0:
                        td[tt] = 0
            else:
                # facilitation
                for tt in range(1, s[1]):
                    delta = a - td[tt-1] * ustim[tt - 1]
                    td[tt] = td[tt-1] + delta
                    if td[tt] > 5:
                        td[tt] = 5

            if crosstalk:
                stim_out *= np.expand_dims(td, 0)
            else:
                stim_out[i, :] *= td

    if np.sum(np.isnan(stim_out)):
    #    import pdb
    #    pdb.set_trace()
        log.info('nan value in stp stim_out')

    # print("(u,tau)=({0},{1})".format(ui,taui))

    stim_out[np.isnan(X)] = np.nan
    return stim_out
