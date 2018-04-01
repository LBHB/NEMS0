import matplotlib.pyplot as plt
import numpy as np
import scipy

def state_vars_timeseries(rec, modelspec, ax=None):
    if ax is not None:
        plt.sca(ax)

    pred = rec['pred']
    resp = rec['resp']

    r1 = resp.as_continuous().T
    p1 = pred.as_continuous().T
    nnidx = np.isfinite(p1)

    r1 = scipy.signal.decimate(r1[nnidx], q=5, axis=0)
    p1 = scipy.signal.decimate(p1[nnidx], q=5, axis=0)

    plt.plot(r1)
    plt.plot(p1)
    mmax = np.nanmax(p1)

    if 'state' in rec.signals.keys():
        for m in modelspec:
            if 'state_dc_gain' in m['fn']:
                g = np.array(m['phi']['g'])
                d = np.array(m['phi']['d'])
            s = ",".join(rec["state"].chans)
            g_string = np.array2string(g, precision=3)
            d_string = np.array2string(d, precision=3)
            s += " g={} d={} ".format(g_string, d_string)

        num_vars = rec['state'].shape[0]
        for i in range(1, num_vars):
            d = rec['state'].as_continuous()[[i], :].T
            d = scipy.signal.decimate(d[nnidx], q=5, axis=0)
            d = d/np.nanmax(d)*mmax - mmax*1.1
            plt.plot(d)
        ax = plt.gca()
        plt.text(0.5, 0.9, s, transform=ax.transAxes,
                 horizontalalignment='center')
    plt.axis('tight')
