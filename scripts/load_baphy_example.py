#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import os
import io
import re
import numpy as np
import scipy.io
#import nems.recording as Recording
import pandas as pd
import matplotlib.pyplot as plt

import nems.utilities as nu
import nems.db as nd
#import nems.utilities.baphy

# find all cells in A1 / natural sound dataset
batch=271

d=nd.get_batch_cells(batch=batch)
print(d['cellid'])

# figure out filepath for demo files
cellid='TAR010c-07-1'

# database can be used to locate files, but need to configure nems
d=nd.get_batch_cell_data(cellid=cellid,batch=batch,label='parm')
parmfilepath=d['parm'][0]

# less responsive site
#cellid='all'   # set cellid='all' to load all cells recorded at this site
#parmfilepath='/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'

options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': cellid, 'pupil': True}
event_times, spike_dict, stim_dict, state_dict = nu.baphy.baphy_load_dataset(parmfilepath,options)


# compute raster for specific unit and stimulus id with sampling rate rasterfs
unitidx=0 # which unit
eventidx=1

stimevents=list(stim_dict.keys())
cellids=list(spike_dict.keys())

event_name=stimevents[eventidx]
cellid=cellids[unitidx]

#event_name='TRIAL'

binlen=1.0/options['rasterfs']
h=np.array([])
ff = (event_times['name']==event_name)
## pull out each epoch from the spike times, generate a raster of spike rate
for i,d in event_times.loc[ff].iterrows():
    print("{0}-{1}".format(d['start'],d['end']))
    edges=np.arange(d['start'],d['end']+binlen,binlen)
    th,e=np.histogram(spike_dict[cellid],edges)
    
    print("{0}-{1}: {2}".format(edges[0],edges[1],sum((spike_dict[cellid]>edges[0]) & (spike_dict[cellid]<edges[1]))))
    th=np.reshape(th,[1,-1])
    if h.size==0:
        # lazy hack: intialize the raster matrix without knowing how many bins it will require
        h=th
    else:
        # concatenate this repetition, making sure binned length matches
        if th.shape[1]<h.shape[1]:
            h=np.concatenate((h,np.zeros([1,h.shape[1]])),axis=0)
            h[-1,:]=np.nan
            h[-1,:th.shape[1]]=th
        else:
            h=np.concatenate((h,th[:,:h.shape[1]]),axis=0)
    
m=np.nanmean(h,axis=0)

plt.figure()
plt.subplot(3,1,1)
plt.imshow(stim_dict[event_name],origin='lower',aspect='auto')
plt.title("stim {0}".format(event_name))
plt.subplot(3,1,2)
plt.imshow(h,origin='lower',aspect='auto')
plt.title("cell {0} raster".format(cellid))
plt.subplot(3,1,3)
plt.plot(np.arange(len(m))*binlen,m)
plt.title("cell {0} PSTH".format(cellid))
plt.tight_layout()

