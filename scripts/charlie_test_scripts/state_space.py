#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:20:45 2018

@author: hellerc
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
import scipy.signal as ss


sys.path.append('/auto/users/hellerc/code/NEMS/')
from nems.utilities.baphy import baphy_load_dataset, spike_time_to_raster
from nems.signal import Signal
sys.path.append('/auto/users/hellerc/code/NEMS/scripts')
import pupil_processing as pp  
import dim_reduction_tools as drt

# load data in first (create a spike/response matrix)
rasterfs=100

# Tartufo NAT
#parmfilepath = '/auto/data/daq/Tartufo/TAR017/TAR017b10_p_NAT.m'
#options = {'rasterfs': rasterfs, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}

# Boleto VOC
parmfilepath = '/auto/data/daq/Boleto/BOL006/BOL006b09_p_PPS_VOC.m'
options = {'rasterfs': rasterfs, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True, 'runclass': 'VOC'}

# load baphy parmfile
out = baphy_load_dataset(parmfilepath,
                           options=options)
# unpack output
event_times, spike_dict, stim_dict, state_dict = out
event_times['start_index']=[int(x) for x in event_times['start']*rasterfs]
event_times['end_index']=[int(x) for x in event_times['end']*rasterfs]

# r is response matrix, created from dictionary of spike times
binfs = 10

out = spike_time_to_raster(spike_dict=spike_dict,
                         fs=binfs,
                         event_times=event_times)

r = out[0][:,:-int(binfs*0.75)]  # remove the time points where pupil is nan
cellids = out[1]

# =========================================================================
# TODO
# Fiter out response matrix based on isolation... need db connection for this
# =========================================================================

# Create state matrix
p = state_dict['pupiltrace'][0][:-int(rasterfs*0.75)]  # remove time shifted points from pupil
p_hp, p_lp = pp.filt(p, rasterfs=rasterfs)
dpos, dneg = pp.derivative(p, rasterfs=rasterfs)
der = dpos+dneg
state_matrix = np.vstack((p[:-1],p_hp[:-1],p_lp[:-1],der))

# downsample state_matrix
nbins = int(round(state_matrix.shape[-1]/(rasterfs/binfs)))
state_matrix=ss.resample(state_matrix.T, nbins)
p = state_matrix[:,0]
p_hp = state_matrix[:,1]
p_lp = state_matrix[:,2]
der = state_matrix[:,3]

# Attempt to find clusters in state space using PCA/Kmeans
U,S,V = np.linalg.svd(state_matrix.T,full_matrices=False)

kmeans = KMeans(n_clusters=2, random_state=0).fit(state_matrix.T)
labels = kmeans.labels_

plt.figure()
plt.title('Kmeans in PCA space')
plt.plot(S[0]*U[:,0][labels==1], S[1]*U[:,1][labels==1], 'r.')
plt.plot(S[0]*U[:,0][labels==0], S[1]*U[:,1][labels==0], 'b.')


plt.figure()
plt.title('Kmeans in pupil-time space')
plt.plot(np.argwhere(labels==1), p[:-1][labels==1], 'r.')
plt.plot(np.argwhere(labels==0), p[:-1][labels==0], 'b.')

# Visualize pupil diamter as sum of two gaussians

def gauss(x,mu,sigma,A):
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)

plt.figure()
n, bins, _ = plt.hist(p,bins=binfs, color='g')
bins=(bins[1:]+bins[:-1])/2
params,cov=curve_fit(bimodal,bins,n)
sigma=np.sqrt(np.diag(cov))
plt.title('Raw pupil diameter')
plt.plot(bins,bimodal(bins,*params),color='red',lw=3,label='model')

# ===== Look at dimesnionality of subspace under different conditions =======

# First, just look at PCs under different kmeans clusters
try:
    out_cluster1 = drt.PCA(r[:,:-1][:,labels==0].T)
    out_cluster2 = drt.PCA(r[:,:-1][:,labels==1].T)
except:
    out_cluster1 = drt.PCA(r[:,:-2][:,labels==0].T)
    out_cluster2 = drt.PCA(r[:,:-2][:,labels==1].T)

# Variance explained in each condition
plt.figure()
plt.plot(out_cluster1['variance'],'b')
plt.plot(out_cluster2['variance'],'r')


# ==== Determine if pupil is best fit with single gaussian or bimodal =====

def mse(r, p):
    mse = []
    for i in range(0,len(r)):
        mse.append((r[i]-p[i])**2)
    mse = sum(mse)/len(r)
    return mse

plt.figure(100)
n, bins, _ = plt.hist(p,bins=binfs, color='g')
plt.close(100)
bins=(bins[1:]+bins[:-1])/2
params_bi,cov_bi=curve_fit(bimodal,bins,n)
sigma_bi=np.sqrt(np.diag(cov_bi))
params_gauss,cov_gauss=curve_fit(gauss,bins,n)
sigma_gauss=np.sqrt(np.diag(cov_gauss))

gauss_fit = gauss(bins,*params_gauss)
bimodal_fit = bimodal(bins,*params_bi)

mse_gauss = mse(gauss_fit, n)
mse_bi = mse(bimodal_fit, n)\

if mse_gauss < mse_bi:
    print('WARNING: pupil dynamics come from single distribution')

out_small_p=drt.PCA(np.squeeze(r[:,np.argwhere(p<=int(round(params_bi[0])/binfs))]).T)
out_big_p=drt.PCA(np.squeeze(r[:,np.argwhere(p>=int(round(params_bi[-1])/binfs))]).T)

# look at variance in two extreme groups
plt.figure()
plt.plot(out_big_p['variance'],'b')
plt.plot(out_small_p['variance'],'r')
plt.legend(['big pupil', 'small pupil'])

    
    