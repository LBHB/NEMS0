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
import scipy.stats
#import nems.recording as Recording
import pandas as pd
import matplotlib.pyplot as plt

import nems.utilities.baphy
import nems.signal

# figure out filepath for demo files
USE_LOCAL_DATA=False
if USE_LOCAL_DATA:
    nems_path=os.path.dirname(nems.utilities.__file__)
    t=nems_path.split('/')
    nems_root='/'.join(t[:-2]) + '/'
    nems.utilities.baphy.stim_cache_dir=nems_root+'signals/baphy_example/'
    nems.utilities.baphy.spk_subdir=''

USE_DB=True
if USE_DB:
    import nems.db as nd

# Nat sound + pupil example
#cellid='TAR010c-CC-U'
#if USE_LOCAL_DATA:
#    parmfilepath=nems.utilities.baphy.stim_cache_dir+'TAR010c16_p_NAT.m'
#else:
#    parmfilepath='/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'
#    options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}
    
#parmfilepath='/auto/data/daq/Boleto/BOL005/BOL005c05_p_PPS_VOC.m'
#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf',
#         'chancount': 18, 'cellid': 'all', 'pupil': True,'runclass': 'VOC'}

#cellid='TAR017b-CC-U'
#parmfilepath='/auto/data/daq/Tartufo/TAR017/TAR017b10_p_NAT.m'
#cellid='eno024d-b1'
#parmfilepath='/auto/data/daq/Enoki/eno024/eno024d10_p_NAT.m'
#pupilfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.pup.mat'



# RDT example
#cellid="oys035b-a2"
#parmfilepath='/auto/data/daq/Oyster/oys035/oys035b04_p_RDT.m'
#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 
#         'cellid': cellid, 'pertrial': True}
#event_times, spike_dict, stim_dict, stim1_dict, stim2_dict, state_dict = nems.utilities.baphy.baphy_load_recording_RDT(parmfilepath,options)


# Behavior example
import nems.signal

cellid='BRT007c-a1'
#cellid='bbl071d-a2'
batch=304

cellid='TAR010c-06-1'
batch=301

if USE_DB:
    d=get_batch_cell_data(batch=batch, cellid=cellid, label='parm') 
    files=list(d['parm'])
    
else:
    files=['/auto/data/daq/Beartooth/BRT007/BRT007c04_p_PTD.m',
           '/auto/data/daq/Beartooth/BRT007/BRT007c05_a_PTD.m',
           '/auto/data/daq/Beartooth/BRT007/BRT007c06_p_PTD.m',
           '/auto/data/daq/Beartooth/BRT007/BRT007c07_a_PTD.m',
           '/auto/data/daq/Beartooth/BRT007/BRT007c09_p_PTD.m']

options={'rasterfs': 20, 'includeprestim': True, 'stimfmt': 'parm', 
         'chancount': 0, 'cellid': cellid, 'pupil': True, 'stim': False,
         'pupil_deblink': True, 'pupil_median': 1}

for i,parmfilepath in enumerate(files):
    
    event_times, spike_dict, stim_dict, state_dict = nems.utilities.baphy.baphy_load_recording(parmfilepath,options)
    
    # generate spike raster
    raster_all,cellids=nems.utilities.baphy.spike_time_to_raster(spike_dict,fs=options['rasterfs'],event_times=event_times)
    
    rlen=raster_all.shape[1]
    plen=state_dict['pupiltrace'].shape[1]
    if plen>rlen:
        state_dict['pupiltrace']=state_dict['pupiltrace'][:,0:-(plen-rlen)]
    elif rlen>plen:
        state_dict['pupiltrace']=state_dict['pupiltrace'][:,0:-(rlen-plen)]
        
    # generate response signal
    t_resp=nems.signal.Signal(fs=options['rasterfs'],matrix=raster_all,name='resp',recording=cellid,chans=cellids,epochs=event_times)
    
    # generate pupil signals
    t_pupil=nems.signal.Signal(fs=options['rasterfs'],matrix=state_dict['pupiltrace'],name='state',recording=cellid,chans=['pupil'],epochs=event_times)
    
    if i==0:
        resp=t_resp
        pupil=t_pupil
    else:
        resp=resp.concatenate_time([resp,t_resp])
        pupil=pupil.concatenate_time([pupil,t_pupil])
      
# generate state signals
hit_trials=pupil.epoch_to_signal('HIT_TRIAL')
miss_trials=pupil.epoch_to_signal('MISS_TRIAL')
fa_trials=pupil.epoch_to_signal('FA_TRIAL')
puretone_trials=pupil.epoch_to_signal('PURETONE_BEHAVIOR')
easy_trials=pupil.epoch_to_signal('EASY_BEHAVIOR')
hard_trials=pupil.epoch_to_signal('HARD_BEHAVIOR')
behavior_state=pupil.epoch_to_signal('ACTIVE_EXPERIMENT')
state=pupil.concatenate_channels([puretone_trials,easy_trials,hard_trials,pupil,hit_trials,fa_trials])

ff=event_times['name'].str.contains('TORC')
stim_names=list(event_times.loc[ff,'name'].unique())
stim_names.sort()

r_dict=resp.extract_epochs(stim_names)
for k,x in r_dict.items():
    r_dict[k]=np.nanmean(x,axis=0)
r2=resp.replace_epochs(r_dict)

r3=r2.select_epoch('REFERENCE')
s3=state.select_epoch('REFERENCE')

r=resp.as_continuous().T
p0=r3.as_continuous().T
s=state.as_continuous().T

ff=np.isfinite(p0)
r=r[ff,np.newaxis]
p0=p0[ff,np.newaxis]
s=s[ff[:,0],:]

# normalize s to have mean zero, variance 1
cols=state.chans
stds=np.std(s,axis=0)
s=s[:,stds>0]
sg,=np.where(stds>0)
cols=[cols[i] for i in sg]

s=s-np.mean(s,axis=0,keepdims=True)
s=s/np.std(s,axis=0,keepdims=True)
m0=np.mean(p0)
p0=p0-m0

X=np.concatenate([p0,m0*s,p0*s],axis=1)

import statsmodels.api as sm
import statsmodels.formula.api as smf


Xlabels=['p0'] + [x.replace(" ","_").replace(":","")+'_bs' for x in cols]+ \
    [x.replace(" ","_").replace(":","")+'_gn' for x in cols]
d=pd.DataFrame(data=X, columns=Xlabels)
d['r']=r

formula='r ~ ' + " + ".join(Xlabels)
results = smf.ols(formula, data=d).fit()
print(results.summary())

pred=np.matmul(X,results.params[1:])+results.params[0]

plt.figure()
plt.plot(r, linewidth=1)
plt.plot(p0+m0, linewidth=1)
plt.plot(pred, linewidth=1)
c=cols
for i in range(0,s.shape[1]):
    x=s[:,i]
    x=x-x.min()
    x=x/x.max()
    plt.plot(x-(i+1)*1.1)
    
    if results.pvalues[i+2]<0.01:
        sb='**'
    elif results.pvalues[i+2]<0.05:
        sb='*'
    else:
        sb=''
    if results.pvalues[i+len(cols)]<0.01:
        sg='**'
    elif results.pvalues[i+len(cols)]<0.05:
        sg='*'
    else:
        sg=''
       
    plt.text(0,-(i+1)*1.1,"{0} (b {1:.2f}{2} g {3:.2f}{4})".format(
            c[i],results.params[i+2],sb,results.params[i+len(cols)],sg))

plt.title("Cell {0} (batch {1})".format(cellid,batch))


if 0:
    # simpler model

    Y=r
    X=np.concatenate([s,p0,np.ones(p0.shape)],axis=1)
    res=np.linalg.lstsq(X,Y)
    beta=res[0]
    pred=np.matmul(X,beta)
    
    plt.figure()
    plt.plot(r, linewidth=1)
    plt.plot(p0, linewidth=1)
    plt.plot(pred, linewidth=1)
    c=state.chans
    for i in range(0,s.shape[1]):
        x=s[:,i]
        x=x-x.min()
        x=x/x.max()
        plt.plot(x-(i+1)*1.1)
        plt.text(0,-(i+1)*1.1,"{0} (b {1:.2f})".format(c[i],beta[i,0]))
    
    plt.title("Cell {0} (batch {1})".format(cellid,batch))





if 0:
    # compute raster for specific unit and stimulus id with sampling rate rasterfs
    stimevents=list(stim_dict.keys())
    unitidx=0 # which unit
    eventidx=1
    r=resp.extract_epoch(stimevents[0])

    stimevents=list(stim_dict.keys())
    cellids=list(spike_dict.keys())
    cellids.sort()
    
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

