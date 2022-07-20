import numpy as np
from numpy import exp
from scipy.integrate import cumtrapz
from scipy.signal import boxcar
import logging

log = logging.getLogger(__name__)


def short_term_plasticity(rec, i, o, u, tau, x0=None, crosstalk=0,
                          reset_signal=None, quick_eval=False, **kwargs):
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

    fn = lambda x: _stp(x, u, tau, x0, crosstalk, rec[i].fs, r, quick_eval)

    return [rec[i].transform(fn, o)]


def short_term_plasticity2(rec, i, o, u, u2, tau, tau2, urat=0.5, x0=None, crosstalk=0,
                           reset_signal=None, quick_eval=False):
    '''
    STP applied to each input channel.
    parameterized by Neher model
        u (release probability)
        tau (recovery time constant)
        tau2 second time constant
    '''

    r = None
    if reset_signal is not None:
        if reset_signal in rec.signals.keys():
            r = rec[reset_signal].as_continuous()

    fn = lambda x: _stp2(x, u, u2, tau, tau2, urat, x0, crosstalk, rec[i].fs, r, quick_eval)

    return [rec[i].transform(fn, o)]


def _cumtrapz(x, dx=1., initial=0., axis=1):
    x = (x[:, :-1] + x[:, 1:]) / 2.0
    x = np.pad(x, ((0, 0), (1, 0)), 'constant', constant_values=(initial,initial))
    #x = tf.pad(x, ((0, 0), (1, 0), (0, 0)), constant_values=initial)
    return np.cumsum(x, axis=axis) * dx


def _stp(X, u, tau, x0=None, crosstalk=0, fs=1, reset_signal=None, quick_eval=False, dep_only=False,
         chunksize=5):
    """
    STP core function
    """
    s = X.shape
    #X = X.astype('float64')
    u = u.astype('float64')
    tau = tau.astype('float64')

    tstim = X.astype('float64')
    tstim[np.isnan(tstim)] = 0
    if x0 is not None:
        tstim -= np.expand_dims(x0, axis=1)

    tstim[tstim < 0] = 0

    # TODO: deal with upper and lower bounds on dep and facilitation parms
    #       need to know something about magnitude of inputs???
    #       move bounds to fitter? slow

    # limits, assumes input (X) range is approximately -1 to +1
    if dep_only or quick_eval:
        ui = np.abs(u.copy())
    else:
        ui = u.copy()

    # force tau to have positive sign (probably better done with bounds on fitter)
    taui = np.absolute(tau.copy())
    taui[taui < 2/fs] = 2/fs

    #ui[ui > 1] = 1
    #ui[ui < -0.4] = -0.4

    # avoid ringing if combination of strong depression and
    # rapid recovery is too large
    #rat = ui**2 / taui

    # MK comment out
    #ui[rat>0.1] = np.sqrt(0.1 * taui[rat>0.1])

    #taui[rat>0.08] = (ui[rat>0.08]**2) / 0.08
    #print("rat: %s" % (ui**2 / taui))

    # convert u & tau units from sec to bins
    taui = taui * fs
    ui = ui / fs * 100

    # convert chunksize from sec to bins
    chunksize = int(chunksize * fs)

    # TODO : allow >1 STP channel per input?

    # go through each stimulus channel
    stim_out = tstim  # allocate scaling term

    if crosstalk:
        # assumes dim of u is 1 !
        tstim = np.mean(tstim, axis=0, keepdims=True)
    if len(ui.shape)==1:
        ui = np.expand_dims(ui, axis=1)
        taui = np.expand_dims(taui, axis=1)
    for i in range(0, len(u)):
        if quick_eval:

            a = 1 / taui
            x = ui * tstim

            if reset_signal is None:
                reset_times = np.arange(0, s[1] + chunksize - 1, chunksize)
            else:
                reset_times = np.argwhere(reset_signal[0, :])[:, 0]
                reset_times = np.append(reset_times, s[1])

            td = np.ones_like(x)
            x0, imu0 = 0., 0.
            for j in range(len(reset_times) - 1):
                si = slice(reset_times[j], reset_times[j + 1])
                xi = x[:, si]

                ix = _cumtrapz(a + xi, dx=1, initial=0, axis=1) + a + (x0 + xi[:, :1]) / 2

                mu = np.exp(ix)
                imu = _cumtrapz(mu * xi, dx=1, initial=0, axis=1) + (x0 + mu[:, :1] * xi[:, :1]) / 2 + imu0

                ff = np.bitwise_and(mu > 0, imu > 0)
                _td = np.ones_like(mu)
                _td[ff] = 1 - np.exp(np.log(imu[ff]) - np.log(mu[ff]))
                td[:, si] = _td

                x0 = xi[:, -1:]
                imu0 = imu[:, -1:] / mu[:, -1:]

            # shift td forward in time by one to allow STP to kick in after the stimulus changes (??)
            #stim_out = tstim * td

            # offset depression by one to allow transients
            stim_out = tstim * np.pad(td[:, :-1], ((0,0), (1,0)), 'constant', constant_values=(1, 1))

            """ 
            a = 1 / taui[i]
            x = ui[i] * tstim[i, :]  # / fs

            #if reset_signal is None:
            reset_times = np.arange(0, len(x) + chunksize - 1, chunksize)
            #else:
            #    reset_times = np.argwhere(reset_signal[0, :])[:, 0]
            #    reset_times = np.append(reset_times, len(x))
            td = np.ones_like(x)
            td0, mu0, imu0, x0 = 1., 1., 0., 0.
            for j in range(len(reset_times) - 1):
                si = slice(reset_times[j], reset_times[j + 1])
                #mu = np.zeros_like(x[si])
                #imu = np.zeros_like(x[si])

                # ix = _cumtrapz(a + x[si])
                ix = cumtrapz(a + x[si], dx=1, initial=0)
                ix += np.log(mu0) + a + (x0 + x[si][0]) / 2

                mu = np.exp(ix)
                # imu = _cumtrapz(mu*x[si]) + td0
                imu = cumtrapz(mu * x[si], dx=1, initial=0)  # /mu0 + imu0/mu0
                imu += imu0 + (mu0 * x0 + mu[0] * x[si][0]) / 2

                ff = np.bitwise_and(mu > 0, imu > 0)
                _td = np.ones_like(mu)
                # _td[mu>0] = 1 - imu[mu>0]/mu[mu>0]
                _td[ff] = 1 - np.exp(np.log(imu[ff]) - np.log(mu[ff]))
                td[si] = _td

                x0 = x[si][-1]
                mu0 = mu[-1]
                imu0 = imu[-1]
                td0 = _td[-1]
                mu0, imu0 = 1, imu0 / mu0

                # if i==0:
                #    plt.figure(figsize=(16,3))
                #    plt.plot(x[si])
                #    plt.plot(_td)
            if crosstalk:
                stim_out *= np.expand_dims(td, 0)
            else:
                stim_out[i, :] *= td
            """
            """
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
                stim_out *= np.expand_dims(td, 0)
            else:
                stim_out[i, :] *= td           
            """
        else:

            a = 1 / taui[i]
            ustim = 1.0 / taui[i] + ui[i] * tstim[i, :]
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
                    td[tt] = td[tt - 1] + delta
                    if td[tt] < 0:
                        td[tt] = 0
            else:
                # facilitation
                for tt in range(1, s[1]):
                    delta = a - td[tt - 1] * ustim[tt - 1]
                    td[tt] = td[tt - 1] + delta
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
    try:
        stim_out[np.isnan(X)] = np.nan
    except:
        import pdb
        pdb.set_trace()

    return stim_out


def _stp2(X, u, u2, tau, tau2, urat=0.5, x0=None, crosstalk=0, fs=1, reset_signal=None, quick_eval=False,
          dep_only=False):
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
    if dep_only or quick_eval:
        ui = np.abs(u.copy())
        ui2 = np.abs(ui2)
    else:
        ui = u.copy()
        ui2 = u2.copy()

    # convert tau units from sec to bins
    taui = np.absolute(tau) * fs
    taui[taui < 2] = 2
    taui2 = np.absolute(tau2) * fs
    # taui2[taui2 < 2] = 2

    # avoid ringing if combination of strong depression and
    # rapid recovery is too large
    rat = ui ** 2 / taui
    ui[rat > 0.1] = np.sqrt(0.1 * taui[rat > 0.1])
    rat = ui2 ** 2 / taui2
    ui2[rat > 0.1] = np.sqrt(0.1 * taui2[rat > 0.1])

    # print("rat: %s" % (ui**2 / taui))

    # TODO : allow >1 STP channel per input?

    # go through each stimulus channel
    stim_out = tstim  # allocate scaling term

    if crosstalk:
        # assumes dim of u is 1 !
        tstim = np.mean(tstim, axis=0, keepdims=True)

    for i in range(0, len(u)):

        a = 1 / taui[i]
        ustim = 1.0 / taui[i] + ui[i] * tstim[i, :]
        # ustim = ui[i] * tstim[i, :]
        td = np.ones_like(ustim)  # initialize dep state vector
        td2 = np.ones_like(ustim)  # initialize dep state vector

        if ui[i] == 0:
            # passthru, no STP, preserve stim_out = tstim
            pass
        elif ui[i] > 0:
            # depression
            for tt in range(1, s[1]):
                # td = di[i, tt - 1]  # previous time bin depression
                delta = (1 - td[tt - 1]) / taui[i] - ui[i] * td[tt - 1] * tstim[i, tt - 1]
                delta2 = (1 - td2[tt - 1]) / taui2[i] - ui2[i] * td2[tt - 1] * tstim[i, tt - 1]

                # delta = 1/taui[i] - td * (1/taui[i] - ui[i] * tstim[i, tt - 1])
                # then a=1/taui[i] and ustim=1/taui[i] - ui[i] * tstim[i,:]
                # delta = a - td[tt - 1] * ustim[tt - 1]
                td[tt] = td[tt - 1] + delta
                td2[tt] = td2[tt - 1] + delta2
                if td[tt] < 0:
                    td[tt] = 0
                if td2[tt] < 0:
                    td2[tt] = 0
        else:
            # facilitation
            for tt in range(1, s[1]):
                delta = a - td[tt - 1] * ustim[tt - 1]
                td[tt] = td[tt - 1] + delta
                if td[tt] > 5:
                    td[tt] = 5

        if crosstalk:
            stim_out *= np.expand_dims(td, 0)
        else:
            stim_out[i, :] *= td * urat + td2 * (1 - urat)

    if np.sum(np.isnan(stim_out)):
        import pdb
        pdb.set_trace()
        log.info('nan value in stp stim_out')

    # print("(u,tau)=({0},{1})".format(ui,taui))

    stim_out[np.isnan(X)] = np.nan
    return stim_out
