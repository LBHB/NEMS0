import numpy as np
from numpy import exp
from scipy.integrate import cumtrapz
from scipy.signal import boxcar

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
    ui = u.copy()

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

    # TODO : enable crosstalk
    if crosstalk:
        raise ValueError('crosstalk not yet supported')

    # TODO : allow >1 STP channel per input?

    # go through each stimulus channel
    stim_out = tstim  # allocate scaling term
    alt=False
    for i in range(0, s[0]):
        if alt:

            #cumstim = np.cumsum(tstim[i,:])/fs
            #t = np.arange(tstim.shape[1])/fs
            #u = exp(t/tau[i] + u[i]*cumstim)
            #td = 1/mu * np.cumsum(mu/tau[i])/fs
            #td = 1 - exp(-t/tau[i] + ui[i]*cumstim)

            a = 1 / tau[i]
            x = 100 * u[i] * tstim[i, :]

            box = np.ones(int(tau[i]*fs*2))
            cx=np.where(np.convolve(x, box, mode='same')==0)


            ix = cumtrapz(a + x, dx=1/fs, initial=0)

            mu = np.exp(ix)
            imu = cumtrapz(mu*x, dx=1/fs, initial=0)

            #mu = np.roll(np.exp(np.cumsum(a + x)), 1)
            #mu[0] = 1
            #mu = np.exp(np.cumsum(a+x) - a)

            #imu = np.roll(np.cumsum(a*mu), 1)
            #imu[0]=0
            #imu += 1

            #imu = np.cumsum(a*mu) + (1-a)
            #imu += (1-imu[0])
            td = 1 - (1/mu * imu)

            #import pdb
            #pdb.set_trace()
            #import matplotlib.pyplot as plt
            #print('mu[0]: ', mu[0])
            #print('imu[0]: ', imu[0])
            #print('mu[end]: ', mu[-1])
            #print('imu[end]: ', imu[-1])
            #plt.figure()
            #plt.subplot(2, 1, 1)
            ##plt.plot(x)
            #plt.plot(np.log(1/mu) + np.log(imu))
            #plt.plot(np.log(mu))
            ##plt.plot((-t/tau[i] + ui[i]*cumstim))
            #plt.plot(np.log(imu))

            ##plt.plot(td)
            #plt.legend(('log(d)','log mu','log imu'))
            stim_out[i, :] *= td
        else:

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
                    td += delta
                    if td < 0:
                        td = 0
                    # td = np.max([td, 0])
                    stim_out[i, tt] *= td
            else:
                # facilitation
                for tt in range(1, s[1]):
                    delta = a - td * ustim[tt - 1]
                    td += delta
                    if td > 5:
                        td = 5
                    # td = np.min([td, 1])
                    stim_out[i, tt] *= td
    if np.sum(np.isnan(stim_out)):
        import pdb
        pdb.set_trace()

    # print("(u,tau)=({0},{1})".format(ui,taui))

    stim_out[np.isnan(X)] = np.nan
    return stim_out
