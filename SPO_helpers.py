#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 14:00:51 2018

@author: luke
"""
import numpy as np
import matplotlib.pyplot as plt
import nems.db as nd  # NEMS database functions -- NOT celldb
import nems_lbhb.baphy as nb   # baphy-specific functions
import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
import nems.recording as recording
import numpy as np
import SPO_helpers as sp
import nems.preprocessing as preproc
import nems.metrics.api as nmet
import pickle as pl
import pandas as pd
import sys
import os
sys.path.insert(0,'/auto/users/luke/Code/Python/Utilities')
import fitEllipse as fE

def parse_stim_type(stim_name):
    stim_sep = stim_name.split('+')
    if len(stim_sep) == 1:
        stim_type = None
    elif stim_sep[1] == 'null':
        stim_type = 'B'
    elif stim_sep[2] == 'null':
        stim_type = 'A'
    elif stim_sep[1] == stim_sep[2]:
        stim_type = 'C'
    else:
        stim_type = 'I'
    return stim_type

def add_stimtype_epochs(sig):
        import pandas as pd
        df0=sig.epochs.copy()
        df0['name']=df0['name'].apply(parse_stim_type)
        df0=df0.loc[df0['name'].notnull()]
        sig.epochs=pd.concat([sig.epochs, df0])
        return sig
def scatterplot_print(x,y,names,ax=None,fn=None,fnargs={},dv=None,**kwargs):

    if ax is None:
        ax=plt.gca()
    if 'marker' not in kwargs:
        kwargs['marker']='.'
    if 'linestyle' not in kwargs:
        kwargs['linestyle']='none'
    good_inds = np.where(np.isfinite(x+y))[0]
    x=x[good_inds]
    y=y[good_inds]
    names=[names[g] for g in good_inds]
    if type(fn) is list:
        for i in range(len(fnargs)):
            if 'pth' in fnargs[i].keys():
                fnargs[i]['pth']=[fnargs[i]['pth'][gi] for gi in good_inds]
    art, = ax.plot(x, y, picker=5,**kwargs)
    #art=ax.scatter(x,y,picker=5,**kwargs)
    
    def onpick(event):
        if event.artist == art:
            #ind = good_inds[event.ind[0]]
            ind=event.ind[0]
            print('onpick scatter: {}: {} ({},{})'.format(ind, names[ind],np.take(x, ind), np.take(y, ind)))
            if dv is not None:
                dv[0]=names[ind]
            if fn is None:
                print('fn is none?')
            elif type(fn) is list:
               for fni,fna in zip(fn,fnargs):
                   fni(names[ind],**fna,ind=ind)
            else:
                fn(names[ind],**fnargs)
    def on_plot_hover(event):
        
        for curve in ax.get_lines():
            if curve.contains(event)[0]:
                print('over {0}'.format(curve.get_gid()))
    ax.figure.canvas.mpl_connect('pick_event', onpick) 
    return art          


def load_SPO(pcellid,fit_epochs,modelspec_name,loader='env100',
             modelspecs_dir='/auto/users/luke/Code/nems/modelspecs',fs=100,get_est=True,get_stim=True):
    import glob
    import nems.analysis.api
    import nems.modelspec as ms
    import warnings
    import nems.recording as recording
    import nems.preprocessing as preproc
    import pandas as pd
    import copy
    
    
    batch=306
    
    
    # load into a recording object
    recname = '/auto/data/nems_db/recordings/' + str(batch) + '/envelope0_fs100/' + pcellid +'.tgz'
    rec = recording.load_recording(recname)
    rec['resp'].fs=fs
    
        # ----------------------------------------------------------------------------
    # DATA PREPROCESSING
    #
    # GOAL: Split your data into estimation and validation sets so that you can
    #       know when your model exhibits overfitting.
    
    
    # Method #1: Find which stimuli have the most reps, use those for val
    if not get_stim:
        del rec.signals['stim']
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    
    # Optional: Take nanmean of ALL occurrences of all signals
    if get_est:
        est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    
    if get_est:
        df0=est['resp'].epochs.copy()
        df2=est['resp'].epochs.copy()
        df0['name']=df0['name'].apply(parse_stim_type)
        df0=df0.loc[df0['name'].notnull()]
        df3 = pd.concat([df0, df2])
    
        est['resp'].epochs=df3
        est_sub=copy.deepcopy(est)
        est_sub['resp']=est_sub['resp'].select_epochs(fit_epochs)
    else:
        est_sub=None
        
    df0=val['resp'].epochs.copy()
    df2=val['resp'].epochs.copy()
    df0['name']=df0['name'].apply(parse_stim_type)
    df0=df0.loc[df0['name'].notnull()]
    df3 = pd.concat([df0, df2])
    
    val['resp'].epochs=df3
    val_sub=copy.deepcopy(val)
    val_sub['resp']=val_sub['resp'].select_epochs(fit_epochs)
    
    # ----------------------------------------------------------------------------
    # GENERATE SUMMARY STATISTICS
    
    
    if modelspec_name is None: 
        return None, [est_sub] , [val_sub]
    else:
        fit_epochs_str="+".join([str(x) for x in fit_epochs])
        mn=loader +  '_subset_'+ fit_epochs_str + '.' + modelspec_name
        an_=modelspecs_dir + '/' + pcellid + '/' + mn
        an=glob.glob(an_+'*')
        if len(an) > 1:
            warnings.warn('{} models found, loading an[0]:{}'.format(len(an),an[0]))
            an=[an[0]]
        if len(an) == 1:
            filepath=an[0]
            modelspecs = [ms.load_modelspec(filepath)]
            modelspecs[0][0]['meta']['modelname']=mn
            modelspecs[0][0]['meta']['cellid']=pcellid
        else:
            raise RuntimeError('not fit')
        # generate predictions
        est_sub, val_sub = nems.analysis.api.generate_prediction(est_sub, val_sub, modelspecs)
        est_sub, val_sub = nems.analysis.api.generate_prediction(est_sub, val_sub, modelspecs)
        
        return modelspecs,est_sub,val_sub

def plot_all_vals(val,modelspec,signames=['resp','pred'],channels=[0,0,1],subset=None,plot_singles_on_dual=False):
    #NOTE TO SELF: Not sure why channels=[0,0,1]. Setting it as default, but when called by plot_linear_and_weighted_psths it should be [0,0,0]
    from nems.plots.timeseries import timeseries_from_epoch
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    if val[signames[0]].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    extracted = val[signames[0]].extract_epoch(epochname)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    
    epochs=val[signames[0]].epochs
    epochs=epochs[epochs['name'] ==  epochname].iloc[occurrences]
    st_mask=val[signames[0]].epochs['name'].str.contains('ST')
    inds=[]
    for index, row in epochs.iterrows():
        matchi = (val[signames[0]].epochs['start'] == row['start']) & (val[signames[0]].epochs['end'] == row['end'])
        matchi = matchi & st_mask
        inds.append(np.where(matchi)[0][0])
         
    names=val[signames[0]].epochs['name'].iloc[inds].tolist()
    
    A=[];B=[];
    for name in names:
        nm=name.split('+')
        A.append(nm[1])
        B.append(nm[2])

    if subset is None:
        plot_order=['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                'STIM_T+si464+si516', 'STIM_T+si516+si464']
    elif subset == 'C+I':
        plot_order=['STIM_T+si464+si464','STIM_T+si516+si516',
                'STIM_T+si464+si516', 'STIM_T+si516+si464']
    
    #OVERWRITE PLOT ORDER TO BE WHAT YOU WANT:
    #plot_order=['STIM_T+si464+null', 'STIM_T+null+si464','STIM_T+si464+si464']
    
    
    plot_order.reverse()
    order = np.array([names.index(nm) for nm in plot_order])
    names_short=[n.replace('STIM_T+','').replace('si464','1').replace('si516','2').replace('null','_') for n in names]
#    names2=sorted(names,key=lambda x: plot_order.index(x))
    
 #   idmap = dict((id,pos) for pos,id in enumerate(plot_order))
    
    sigs = [val[s] for s in signames]
    title = ''
    nplt=len(plot_order)
    gs_kw = dict(hspace=0,left=0.04,right=.99)
    fig, ax = plt.subplots(nrows=nplt, ncols=1, figsize=(10, 15),sharey=True,gridspec_kw=gs_kw)
    if signames==['resp', 'lin_model']:
        [axi.set_prop_cycle(cycler('color', ['k','g']) + cycler(linestyle=['-', 'dotted'])+ cycler(linewidth=[1,2])) for axi in ax]
    else:
        [axi.set_prop_cycle(cycler('color', ['k','#1f77b4', 'r']) + cycler('linestyle', ['-', '-', '--'])) for axi in ax]
    allsigs =np.hstack([s.as_continuous()[-1,:] for s in sigs])    
    yl=[np.nanmin(allsigs), np.nanmax(allsigs)]
    stimname='stim' #LAS was this
    stimname='resp'
    prestimtime=val[stimname].epochs.loc[0].end
    
    for i in range(nplt):
        timeseries_from_epoch(sigs, epochname, title=title,
                         occurrences=occurrences[order[i]],ax=ax[i],channels=channels,linestyle=None,linewidth=None)
        if names_short[order[i]] in ['1+_','2+_']:
            #timeseries_from_epoch([val['stim']], epochname, title=title,
            #             occurrences=occurrences[order[i]],ax=ax[i])
            ep=val[stimname].extract_epoch(names[order[i]]).squeeze()
            ep=80+20*np.log10(ep.T)
            ep=ep/ep.max()*yl[1]
            time_vector = np.arange(0, len(ep)) / val[stimname].fs
            ax[i].plot(time_vector-prestimtime,ep,'--',color='#ff7f0e')
        if plot_singles_on_dual:
            snA=names_short[order[i]][:2]+'_'
            snB='_'+names_short[order[i]][1:]
            snA_=names[names_short.index(snA)]
            snB_=names[names_short.index(snB)]
            epA=val['resp'].extract_epoch(snA_).squeeze()
            epB=val['resp'].extract_epoch(snB_).squeeze()
            time_vector = np.arange(0, len(epA)) / val['resp'].fs
            ax[i].plot(time_vector-prestimtime,epA,'--',color=(1,.5,0),linewidth=1.5)
            ax[i].plot(time_vector-prestimtime,epB,'--',color=(0,.5,1),linewidth=1.5)
        ax[i].set_ylabel(names_short[order[i]],rotation=0,horizontalalignment='right',verticalalignment='bottom')

    if modelspec is not None:
        ax[0].set_title('{}: {}'.format(modelspec[0]['meta']['cellid'],modelspec[0]['meta']['modelname']))
    [axi.get_xaxis().set_visible(False) for axi in ax[:-1]]
    [axi.get_yaxis().set_ticks([]) for axi in ax]  
    [axi.get_legend().set_visible(False) for axi in ax[:-1]]
    [axi.set_xlim([.8-1, 4.5-1]) for axi in ax]
    yl_margin = .01*(yl[1]-yl[0])
    [axi.set_ylim((yl[0]-yl_margin, yl[1]+yl_margin)) for axi in ax]
    if plot_singles_on_dual:
        ls=['resp A','resp B']
    else:
        ls=['Stim']
    ax[nplt-1].legend(signames+ls)
    return fig

def smooth(x,window_len=11,passes=2,window='flat'):
    import numpy as np
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    
    
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')
    y=s 
    for passnum in range(passes):
        y=np.convolve(w/w.sum(),y,mode='valid')
    return y

def export_all_vals(val,modelspec,signames=['resp','pred']):
    from nems.plots.timeseries import timeseries_from_epoch
    import matplotlib.pyplot as plt
    import numpy as np
    from cycler import cycler
    if val[signames[0]].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    extracted = val[signames[0]].extract_epoch(epochname)
    finite_trial = [np.sum(np.isfinite(x)) > 0 for x in extracted]
    occurrences, = np.where(finite_trial)
    
    epochs=val[signames[0]].epochs
    epochs=epochs[epochs['name'] ==  epochname].iloc[occurrences]
    st_mask=val[signames[0]].epochs['name'].str.contains('ST')
    inds=[]
    for index, row in epochs.iterrows():
        matchi = (val[signames[0]].epochs['start'] == row['start']) & (val[signames[0]].epochs['end'] == row['end'])
        matchi = matchi & st_mask
        inds.append(np.where(matchi)[0][0])
         
    names=val[signames[0]].epochs['name'].iloc[inds].tolist()
    
    A=[];B=[];
    for name in names:
        nm=name.split('+')
        A.append(nm[1])
        B.append(nm[2])

    plot_order=['STIM_T+si464+null', 'STIM_T+null+si464', 'STIM_T+si464+si464',
                'STIM_T+si516+null', 'STIM_T+null+si516', 'STIM_T+si516+si516',
                'STIM_T+si464+si516', 'STIM_T+si516+si464']
    plot_order.reverse()
    order = np.array([names.index(nm) for nm in plot_order])
    names_short=[n.replace('STIM_T+','').replace('si464','1').replace('si516','2').replace('null','_') for n in names]
#    names2=sorted(names,key=lambda x: plot_order.index(x))
    
 #   idmap = dict((id,pos) for pos,id in enumerate(plot_order))
    
    nplt=len(occurrences)
    ep=[]
    for i in range(nplt):
        ep.append(val['pred'].extract_epoch(names[order[i]]).squeeze())
    
    ep_=val['resp'].fs*np.array(ep)
    dd='/auto/users/luke/Code/nems/modelspecs/svd_fs_branch/'
    pth=dd+modelspec[0]['meta']['cellid']+'/'+modelspec[0]['meta']['modelname']
    np.save(pth+'.npy',ep_)
    from PyQt5.QtCore import pyqtRemoveInputHook
    from pdb import set_trace
    pyqtRemoveInputHook()
    set_trace() 
    
def calc_psth_metrics(batch,cellid,rec_file=None):
    import nems.db as nd  # NEMS database functions -- NOT celldb
    import nems_lbhb.baphy as nb   # baphy-specific functions
    import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
    import nems.recording as recording
    import numpy as np
    import SPO_helpers as sp
    import nems.preprocessing as preproc
    import nems.metrics.api as nmet
    import nems.metrics.corrcoef
    import copy
    
    options = {}
    #options['cellid']=cellid
    #options['batch']=batch
    #options["stimfmt"] = "envelope"
    #options["chancount"] = 0
    #options["rasterfs"] = 100
    #rec_file=nb.baphy_data_path(options)
    from pdb import set_trace
    set_trace() 
    if rec_file is None:
        rec_file = nw.generate_recording_uri(cellid, batch, loadkey='ns.fs100')  #'was 'env.fs100'
    #uri = nb.baphy_load_recording_uri(cellid=cellid, batch=batch, **options)
    rec=recording.load_recording(rec_file)
    rec['resp'] = rec['resp'].extract_channels([cellid])
    rec['resp'].fs=200
    
    epcs=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    ep2=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PostStimSilence'].iloc[0].copy()
    prestim=epcs.iloc[0]['end']
    poststim=ep2['end']-ep2['start']
    
    spike_times=rec['resp']._data[cellid]
    count=0
    for index, row in epcs.iterrows():
        count+=np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR=count/(epcs['end']-epcs['start']).sum()
    
    resp=rec['resp'].rasterize()
    resp=sp.add_stimtype_epochs(resp)
    ps=resp.select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_rast=ps[ff].mean()*resp.fs
    SR_std=ps[ff].std()*resp.fs
    
    #COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    #smooth and subtract SR
    fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR/resp.fs)
    val['resp']=val['resp'].transform(fn)
    val['resp']=sp.add_stimtype_epochs(val['resp'])
    
    if val['resp'].count_epoch('REFERENCE'):
        epochname = 'REFERENCE'
    else:
        epochname = 'TRIAL'
    sts=val['resp'].epochs['start'].copy()
    nds=val['resp'].epochs['end'].copy()
    sts_rec=rec['resp'].epochs['start'].copy()
    val['resp'].epochs['end']=val['resp'].epochs['start']+prestim
    ps=val['resp'].select_epochs([epochname]).as_continuous()
    ff = np.isfinite(ps)
    SR_av=ps[ff].mean()*resp.fs
    SR_av_std=ps[ff].std()*resp.fs
    val['resp'].epochs['end']=nds
    
    # restore times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    val['resp'].epochs['start']=val['resp'].epochs['start']+prestim
    TotalMax = np.nanmax(val['resp'].as_continuous())
    ps=np.hstack((val['resp'].extract_epoch('A').flatten(), val['resp'].extract_epoch('B').flatten()))
    SinglesMax = np.nanmax(ps)

    # Change epochs to stimulus ss times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    val['resp'].epochs['start']=val['resp'].epochs['start']+prestim+.5
    val['resp'].epochs['end']=val['resp'].epochs['end']-poststim
    types=['A','B','C','I']
    thresh=np.array(((SR + SR_av_std)/resp.fs,
            (SR - SR_av_std)/resp.fs))
    thresh=np.array((SR/resp.fs + 0.1 * (SinglesMax - SR/resp.fs),
            (SR - SR_av_std)/resp.fs))
            #SR/resp.fs - 0.5 * (np.nanmax(val['resp'].as_continuous()) - SR/resp.fs)]

    excitatory_percentage={}
    inhibitory_percentage={}
    Max={}
    Mean={}
    for _type in types:
        ps=val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage[_type]=(ps[ff]>thresh[0]).sum()/ff.sum()
        inhibitory_percentage[_type]=(ps[ff]<thresh[1]).sum()/ff.sum()
        Max[_type]=ps[ff].max()/SinglesMax      
        Mean[_type]=ps[ff].mean()
    
 
    # restore times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    # Change epochs to stimulus onset times
    val['resp'].epochs['start']=val['resp'].epochs['start']+prestim
    val['resp'].epochs['end']=val['resp'].epochs['start']+prestim+.5
    types=['A','B','C','I']
    excitatory_percentage_onset={}
    inhibitory_percentage_onset={}
    Max_onset={}
    for _type in types:
        ps=val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage_onset[_type]=(ps[ff]>thresh[0]).sum()/ff.sum()
        inhibitory_percentage_onset[_type]=(ps[ff]<thresh[1]).sum()/ff.sum()
        Max_onset[_type]=ps[ff].max()/SinglesMax      

    # restore times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    val['resp'].epochs['start']=val['resp'].epochs['start']+prestim
    rec['resp'].epochs['start']=rec['resp'].epochs['start']+prestim
    #over stim on time to end + 0.5
    val['linmodel']=val['resp'].copy()
    val['linmodel']._data = np.full(val['linmodel']._data.shape, np.nan)
    types=['C','I']
    epcs=val['resp'].epochs[val['resp'].epochs['name'].str.contains('STIM')].copy()
    epcs['type']=epcs['name'].apply(sp.parse_stim_type)
    EA=np.array([n.split('+')[1] for n in epcs['name']])
    EB=np.array([n.split('+')[2] for n in epcs['name']])
    r_dual_B={}; r_dual_A={}; r_dual_B_nc={}; r_dual_A_nc={}; r_dual_B_bal={}; r_dual_A_bal={}
    r_lin_B={}; r_lin_A={}; r_lin_B_nc={}; r_lin_A_nc={}; r_lin_B_bal={}; r_lin_A_bal={}
    N_ac=200
    full_resp=rec['resp'].rasterize()
    full_resp=full_resp.transform(fn)
    for _type in types:
        inds=np.nonzero(epcs['type'].values == _type)[0]
        rA_st=[]; rB_st=[]; r_st=[]; rA_rB_st=[];
        init=True
        for ind in inds:
            r=val['resp'].extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                print(epcs.iloc[ind]['name'])
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    #from pdb import set_trace
                    #set_trace() 
                    rA=val['resp'].extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB=val['resp'].extract_epoch(epcs.iloc[indB[0]]['name'])
                    r_st.append(full_resp.extract_epoch(epcs.iloc[ind]['name'])[:,0,:])
                    rA_st_=full_resp.extract_epoch(epcs.iloc[indA[0]]['name'])[:,0,:]
                    rB_st_=full_resp.extract_epoch(epcs.iloc[indB[0]]['name'])[:,0,:]
                    rA_st.append(rA_st_)
                    rB_st.append(rB_st_)
                    minreps=np.min((rA_st_.shape[0],rB_st_.shape[0]))
                    rA_rB_st.append(rA_st_[:minreps,:]+rB_st_[:minreps,:])
                    if init:
                        rA_=rA.squeeze(); rB_=rB.squeeze(); r_=r.squeeze(); rA_rB_=rA.squeeze()+rB.squeeze()
                        init=False
                    else:
                        rA_=np.hstack((rA_,rA.squeeze()))
                        rB_=np.hstack((rB_,rB.squeeze()))
                        r_=np.hstack((r_,r.squeeze()))
                        rA_rB_=np.hstack((rA_rB_,rA.squeeze()+rB.squeeze()))
                    val['linmodel']=val['linmodel'].replace_epoch(epcs.iloc[ind]['name'],rA+rB,preserve_nan=False)
        ff = np.isfinite(r_) & np.isfinite(rA_) & np.isfinite(rB_)
        r_dual_A[_type]=np.corrcoef(rA_[ff],r_[ff])[0,1]
        r_dual_B[_type]=np.corrcoef(rB_[ff],r_[ff])[0,1]
        r_lin_A[_type]=np.corrcoef(rA_[ff],rA_rB_[ff])[0,1]
        r_lin_B[_type]=np.corrcoef(rB_[ff],rA_rB_[ff])[0,1]
        
        minreps = np.min([x.shape[0] for x in r_st])
        r_st = [x[:minreps, :] for x in r_st]
        r_st = np.concatenate(r_st, axis=1)
        rA_st = [x[:minreps, :] for x in rA_st]
        rA_st = np.concatenate(rA_st, axis=1)
        rB_st = [x[:minreps, :] for x in rB_st]
        rB_st = np.concatenate(rB_st, axis=1)
        rA_rB_st = [x[:minreps, :] for x in rA_rB_st]
        rA_rB_st = np.concatenate(rA_rB_st, axis=1)
        
        r_lin_A_bal[_type]=np.corrcoef(rA_st[0::2,ff].mean(axis=0),rA_rB_st[1::2,ff].mean(axis=0))[0,1]
        r_lin_B_bal[_type]=np.corrcoef(rB_st[0::2,ff].mean(axis=0),rA_rB_st[1::2,ff].mean(axis=0))[0,1]
        r_dual_A_bal[_type]=np.corrcoef(rA_st[0::2,ff].mean(axis=0),r_st[:,ff].mean(axis=0))[0,1]
        r_dual_B_bal[_type]=np.corrcoef(rB_st[0::2,ff].mean(axis=0),r_st[:,ff].mean(axis=0))[0,1]
        
        r_dual_A_nc[_type] = r_noise_corrected(rA_st,r_st)
        r_dual_B_nc[_type] = r_noise_corrected(rB_st,r_st)
        r_lin_A_nc[_type] = r_noise_corrected(rA_st,rA_rB_st)
        r_lin_B_nc[_type] = r_noise_corrected(rB_st,rA_rB_st)
        
        

        if _type is 'C':
            r_A_B=np.corrcoef(rA_[ff],rB_[ff])[0,1]
            r_A_B_nc = r_noise_corrected(rA_st,rB_st)
            rAA = nems.metrics.corrcoef._r_single(rA_st,200,0)
            rBB = nems.metrics.corrcoef._r_single(rB_st,200,0)
            rCC = nems.metrics.corrcoef._r_single(r_st,200,0)
            Np=0
            rAA_nc=np.zeros(Np)
            rBB_nc=np.zeros(Np)
            hv=int(minreps/2);
            for i in range(Np):
                inds=np.random.permutation(minreps)
                rAA_nc[i]=sp.r_noise_corrected(rA_st[inds[:hv]],rA_st[inds[hv:]])
                rBB_nc[i]=sp.r_noise_corrected(rB_st[inds[:hv]],rB_st[inds[hv:]])
            ffA=np.isfinite(rAA_nc)
            ffB=np.isfinite(rBB_nc)
            rAAm=rAA_nc[ffA].mean()
            rBBm=rBB_nc[ffB].mean()
            mean_nsA=rA_st.sum(axis=1).mean()
            mean_nsB=rB_st.sum(axis=1).mean()
            min_nsA=rA_st.sum(axis=1).min()
            min_nsB=rB_st.sum(axis=1).min()
        else:
            rII = nems.metrics.corrcoef._r_single(r_st,200,0)
        #rac = _r_single(X, N)
        #r_ceiling = [nmet.r_ceiling(p, rec, 'pred', 'resp') for p in val_copy]
        
    r_fit_linmodel={}             
    r_fit_linmodel_NM={}
    r_ceil_linmodel={}
    mean_enh={}
    mean_supp={}
    EnhP={}
    SuppP={}
    DualAboveZeroP={}
    resp_=copy.deepcopy(rec['resp'].rasterize())
    resp_.epochs['start']=sts_rec
    fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR/val['resp'].fs)
    resp_=resp_.transform(fn)
    for _type in types:
        val_copy=copy.deepcopy(val)
#        from pdb import set_trace
#        set_trace()
        val_copy['resp']=val_copy['resp'].select_epochs([_type])
        r_fit_linmodel_NM[_type] = nmet.corrcoef(val_copy, 'linmodel', 'resp') 
        #r_ceil_linmodel[_type] = nems.metrics.corrcoef.r_ceiling(val_copy,rec,'linmodel', 'resp',exclude_neg_pred=False)[0]
        r_ceil_linmodel[_type] = nems.metrics.corrcoef.r_ceiling(val_copy,rec,'linmodel', 'resp')[0]
        
        pred = val_copy['linmodel'].as_continuous()
        resp = val_copy['resp'].as_continuous()
        ff = np.isfinite(pred) & np.isfinite(resp)
        #cc = np.corrcoef(sp.smooth(pred[ff],3,2), sp.smooth(resp[ff],3,2))
        cc = np.corrcoef(pred[ff], resp[ff])
        r_fit_linmodel[_type] = cc[0, 1]
        
        prdiff= resp[ff] - pred[ff]
        mean_enh[_type] = prdiff[prdiff > 0].mean()*val['resp'].fs
        mean_supp[_type] = prdiff[prdiff < 0].mean()*val['resp'].fs



                
        Njk=10
        if _type is 'C':
            stims=['STIM_T+si464+si464','STIM_T+si516+si516']
        else:
            stims=['STIM_T+si464+si516', 'STIM_T+si516+si464']
        T=int(700+prestim*val['resp'].fs)
        Tps=int(prestim*val['resp'].fs)
        jns=np.zeros((Njk,T,len(stims)))
        for ns in range(len(stims)):
            for njk in range(Njk):
                resp_jn=resp_.jackknife_by_epoch(Njk,njk,stims[ns])
                jns[njk,:,ns]=np.nanmean(resp_jn.extract_epoch(stims[ns]),axis=0)
        jns=np.reshape(jns[:,Tps:,:],(Njk,700*len(stims)),order='F')
        
        lim_models=np.zeros((700,len(stims)))
        for ns in range(len(stims)):
            lim_models[:,ns]=val_copy['linmodel'].extract_epoch(stims[ns])
        lim_models=lim_models.reshape(700*len(stims),order='F')
        
        ff=np.isfinite(lim_models)
        mean_diff=(jns[:,ff]-lim_models[ff]).mean(axis=0)
        std_diff=(jns[:,ff]-lim_models[ff]).std(axis=0)      
        serr_diff=np.sqrt(Njk/(Njk-1))*std_diff
        
        thresh=3
        dual_above_zero = (jns[:,ff].mean(axis=0) > std_diff)
        sig_enh = ((mean_diff/serr_diff) > thresh) & dual_above_zero
        sig_supp = ((mean_diff/serr_diff) < -thresh)
        DualAboveZeroP[_type] = (dual_above_zero).sum()/len(mean_diff)
        EnhP[_type] = (sig_enh).sum()/len(mean_diff)
        SuppP[_type] = (sig_supp).sum()/len(mean_diff)
        
#        time = np.arange(0, lim_models.shape[0])/ val['resp'].fs   
#        plt.figure();
#        plt.plot(time,jns.mean(axis=0),'.-k');
#        plt.plot(time,lim_models,'.-g');
#        plt.plot(time[sig_enh],lim_models[sig_enh],'.r')
#        plt.plot(time[sig_supp],lim_models[sig_supp],'.b')
#        plt.title('Type:{:s}, Enh:{:.2f}, Sup:{:.2f}, Resp_above_zero:{:.2f}'.format(_type,EnhP[_type],SuppP[_type],DualAboveZeroP[_type]))
#        from pdb import set_trace
#        set_trace() 
#        a=2
        #thrsh=5
#        EnhP[_type] = ((prdiff*val['resp'].fs) > thresh).sum()/len(prdiff)
#        SuppP[_type] = ((prdiff*val['resp'].fs) < -thresh).sum()/len(prdiff)
#    return val
#    return {'excitatory_percentage':excitatory_percentage,
#            'inhibitory_percentage':inhibitory_percentage,
#            'r_fit_linmodel':r_fit_linmodel,
#            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}
#    
    return {'thresh':thresh*val['resp'].fs,
            'EP_A':excitatory_percentage['A'],
            'EP_B':excitatory_percentage['B'],
            'EP_C':excitatory_percentage['C'],
            'EP_I':excitatory_percentage['I'],
            'IP_A':inhibitory_percentage['A'],
            'IP_B':inhibitory_percentage['B'],
            'IP_C':inhibitory_percentage['C'],
            'IP_I':inhibitory_percentage['I'],
            'OEP_A':excitatory_percentage_onset['A'],
            'OEP_B':excitatory_percentage_onset['B'],
            'OEP_C':excitatory_percentage_onset['C'],
            'OEP_I':excitatory_percentage_onset['I'],
            'OIP_A':inhibitory_percentage_onset['A'],
            'OIP_B':inhibitory_percentage_onset['B'],
            'OIP_C':inhibitory_percentage_onset['C'],
            'OIP_I':inhibitory_percentage_onset['I'],
            'Max_A':Max['A'],
            'Max_B':Max['B'],
            'Max_C':Max['C'],
            'Max_I':Max['I'],
            'Mean_A':Mean['A'],
            'Mean_B':Mean['B'],
            'Mean_C':Mean['C'],
            'Mean_I':Mean['I'],
            'OMax_A':Max_onset['A'],
            'OMax_B':Max_onset['B'],
            'OMax_C':Max_onset['C'],
            'OMax_I':Max_onset['I'],
            'TotalMax':TotalMax*val['resp'].fs,
            'SinglesMax':SinglesMax*val['resp'].fs,
            'r_lin_C':r_fit_linmodel['C'],
            'r_lin_I':r_fit_linmodel['I'],
            'r_lin_C_NM':r_fit_linmodel_NM['C'],
            'r_lin_I_NM':r_fit_linmodel_NM['I'],
            'r_ceil_C':r_ceil_linmodel['C'],
            'r_ceil_I':r_ceil_linmodel['I'],
            'MEnh_C':mean_enh['C'],
            'MEnh_I':mean_enh['I'],
            'MSupp_C':mean_supp['C'],
            'MSupp_I':mean_supp['I'],
            'EnhP_C':EnhP['C'],
            'EnhP_I':EnhP['I'],
            'SuppP_C':SuppP['C'],
            'SuppP_I':SuppP['I'],
            'DualAboveZeroP_C':DualAboveZeroP['C'],
            'DualAboveZeroP_I':DualAboveZeroP['I'],
            'r_dual_A_C':r_dual_A['C'],
            'r_dual_A_I':r_dual_A['I'],
            'r_dual_B_C':r_dual_B['C'],
            'r_dual_B_I':r_dual_B['I'],
            'r_dual_A_C_nc':r_dual_A_nc['C'],
            'r_dual_A_I_nc':r_dual_A_nc['I'],
            'r_dual_B_C_nc':r_dual_B_nc['C'],
            'r_dual_B_I_nc':r_dual_B_nc['I'],
            'r_dual_A_C_bal':r_dual_A_bal['C'],
            'r_dual_A_I_bal':r_dual_A_bal['I'],
            'r_dual_B_C_bal':r_dual_B_bal['C'],
            'r_dual_B_I_bal':r_dual_B_bal['I'],
            'r_lin_A_C':r_lin_A['C'],
            'r_lin_A_I':r_lin_A['I'],
            'r_lin_B_C':r_lin_B['C'],
            'r_lin_B_I':r_lin_B['I'],
            'r_lin_A_C_nc':r_lin_A_nc['C'],
            'r_lin_A_I_nc':r_lin_A_nc['I'],
            'r_lin_B_C_nc':r_lin_B_nc['C'],
            'r_lin_B_I_nc':r_lin_B_nc['I'],
            'r_lin_A_C_bal':r_lin_A_bal['C'],
            'r_lin_A_I_bal':r_lin_A_bal['I'],
            'r_lin_B_C_bal':r_lin_B_bal['C'],
            'r_lin_B_I_bal':r_lin_B_bal['I'],
            'r_A_B':r_A_B,
            'r_A_B_nc':r_A_B_nc,
            'rAAm':rAAm,'rBBm':rBBm,
            'rAA':rAA,'rBB':rBB,'rCC':rCC,'rII':rII,
            'rAA_nc':rAA_nc,'rBB_nc':rBB_nc,
            'mean_nsA':mean_nsA,'mean_nsB':mean_nsB,'min_nsA':min_nsA,'min_nsB':min_nsB,
            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}    
    
def r_noise_corrected(X,Y,N_ac=200):
    import nems.metrics.corrcoef
    Xac = nems.metrics.corrcoef._r_single(X, N_ac,0)
    Yac = nems.metrics.corrcoef._r_single(Y, N_ac,0)
    repcount = X.shape[0]
    rs = np.zeros((repcount,repcount))
    for nn in range(repcount):
        for mm in range(repcount):
            X_ = X[mm, :]
            Y_ = Y[nn, :]
            # remove all nans from pred and resp
            ff = np.isfinite(X_) & np.isfinite(Y_)

            if (np.sum(X_[ff]) != 0) and (np.sum(Y_[ff]) != 0):
                rs[nn,mm] = np.corrcoef(X_[ff],Y_[ff])[0, 1]
            else:
                rs[nn,mm] = 0
    #rs=rs[np.triu_indices(rs.shape[0],1)]
    #plt.figure(); plt.imshow(rs)
    return np.mean(rs)/(np.sqrt(Xac) * np.sqrt(Yac))
    
    
    
def calc_psth_metrics_orig(batch,cellid):
    import nems.db as nd  # NEMS database functions -- NOT celldb
    import nems_lbhb.baphy as nb   # baphy-specific functions
    import nems_lbhb.xform_wrappers as nw  # wrappers for calling nems code with database stuff
    import nems.recording as recording
    import numpy as np
    import SPO_helpers as sp
    import nems.preprocessing as preproc
    import nems.metrics.api as nmet
    import copy
    
    options = {}
    options['cellid']=cellid
    options['batch']=batch
    options["stimfmt"] = "envelope"
    options["chancount"] = 0
    options["rasterfs"] = 100
    rec_file=nb.baphy_data_path(options)
    rec=recording.load_recording(rec_file)
    rec['resp'].fs=200
    
    epcs=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    spike_times=rec['resp']._data[options['cellid']]
    count=0
    for index, row in epcs.iterrows():
        count+=np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR=count/(epcs['end']-epcs['start']).sum()
    
    resp=rec['resp'].rasterize()
    resp=sp.add_stimtype_epochs(resp)
    ps=resp.select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_rast=ps[ff].mean()*resp.fs
    SR_std=ps[ff].std()*resp.fs
    
    #COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    from pdb import set_trace
    set_trace() 
    #smooth and subtract SR
    fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR/resp.fs)
    val['resp']=val['resp'].transform(fn)
    val['resp']=sp.add_stimtype_epochs(val['resp'])
    
    
    sts=val['resp'].epochs['start'].copy()
    nds=val['resp'].epochs['end'].copy()
    val['resp'].epochs['end']=val['resp'].epochs['start']+1
    ps=val['resp'].select_epochs(['TRIAL']).as_continuous()
    ff = np.isfinite(ps)
    SR_av=ps[ff].mean()*resp.fs
    SR_av_std=ps[ff].std()*resp.fs
    val['resp'].epochs['end']=nds
    
    # restore times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    val['resp'].epochs['start']=val['resp'].epochs['start']+1
    TotalMax = np.nanmax(val['resp'].as_continuous())
    ps=np.hstack((val['resp'].extract_epoch('A').flatten(), val['resp'].extract_epoch('B').flatten()))
    SinglesMax = np.nanmax(ps)

    # Change epochs to stimulus ss times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    val['resp'].epochs['start']=val['resp'].epochs['start']+1.5
    val['resp'].epochs['end']=val['resp'].epochs['end']-0.5
    types=['A','B','C','I']
    thresh=np.array(((SR + SR_av_std)/resp.fs,
            (SR - SR_av_std)/resp.fs))
    thresh=np.array((SR/resp.fs + 0.1 * (SinglesMax - SR/resp.fs),
            (SR - SR_av_std)/resp.fs))
            #SR/resp.fs - 0.5 * (np.nanmax(val['resp'].as_continuous()) - SR/resp.fs)]

    excitatory_percentage={}
    inhibitory_percentage={}
    Max={}
    for _type in types:
        ps=val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage[_type]=(ps[ff]>thresh[0]).sum()/ff.sum()
        inhibitory_percentage[_type]=(ps[ff]<thresh[1]).sum()/ff.sum()
        Max[_type]=ps[ff].max()/SinglesMax      
    
    
    # restore times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    # Change epochs to stimulus onset times
    val['resp'].epochs['start']=val['resp'].epochs['start']+1
    val['resp'].epochs['end']=val['resp'].epochs['start']+1.5
    types=['A','B','C','I']
    excitatory_percentage_onset={}
    inhibitory_percentage_onset={}
    Max_onset={}
    for _type in types:
        ps=val['resp'].extract_epoch(_type).flatten()
        ff = np.isfinite(ps)
        excitatory_percentage_onset[_type]=(ps[ff]>thresh[0]).sum()/ff.sum()
        inhibitory_percentage_onset[_type]=(ps[ff]<thresh[1]).sum()/ff.sum()
        Max_onset[_type]=ps[ff].max()/SinglesMax      

    # restore times
    val['resp'].epochs['end']=nds
    val['resp'].epochs['start']=sts
    val['resp'].epochs['start']=val['resp'].epochs['start']+1
    
    #over stim on time to end + 0.5
    val['linmodel']=val['resp'].copy()
    val['linmodel']._data = np.full(val['linmodel']._data.shape, np.nan)
    types=['C','I']
    epcs=val['resp'].epochs[val['resp'].epochs['name'].str.contains('STIM')].copy()
    epcs['type']=epcs['name'].apply(sp.parse_stim_type)
    EA=np.array([n.split('+')[1] for n in epcs['name']])
    EB=np.array([n.split('+')[2] for n in epcs['name']])
    for _type in types:
        inds=np.nonzero(epcs['type'].values == _type)[0]
        for ind in inds:
            r=val['resp'].extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    rA=val['resp'].extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB=val['resp'].extract_epoch(epcs.iloc[indB[0]]['name'])
                    val['linmodel']=val['linmodel'].replace_epoch(epcs.iloc[ind]['name'],rA+rB,preserve_nan=False)
    
    r_fit_linmodel={}             
    mean_enh={}
    mean_supp={}
    EnhP={}
    SuppP={}
    for _type in types:
        val_copy=copy.deepcopy(val)
        val_copy['resp']=val_copy['resp'].select_epochs(_type)
        #r_fit_linmodel[_type] = nmet.corrcoef(val_copy, 'linmodel', 'resp')
        pred = val_copy['linmodel'].as_continuous()
        resp = val_copy['resp'].as_continuous()
        ff = np.isfinite(pred) & np.isfinite(resp)
        #cc = np.corrcoef(sp.smooth(pred[ff],3,2), sp.smooth(resp[ff],3,2))
        cc = np.corrcoef(pred[ff], resp[ff])
        r_fit_linmodel[_type] = cc[0, 1]
        
        prdiff= resp[ff] - pred[ff]
        mean_enh[_type] = prdiff[prdiff > 0].mean()*val['resp'].fs
        mean_supp[_type] = prdiff[prdiff < 0].mean()*val['resp'].fs


        resp=rec['resp'].rasterize()
        resp_jn=resp.jackknife_by_epoch(10,0,'STIM_T+si464+si464')
        resp_jn=rec.jackknife_by_epoch(10,0,'STIM_T+si464+si464')
        
        
        val['resp'].extract_epoch('STIM_T+si464+si464')
        
        resp_jn.np.zeros(900,10)
        Njk=10
        jns=np.zeros(900,Njk,ken(stims))
        for ns in range(len(stims)):
            for resp_jn in resp.jackknifes_by_epoch(10,stims[ns]):
            
                resp_jn[njk,:,ns]=resp_jn.extract_epoch(stims[ns]).mean(axis=1)
        
        thresh=5
        EnhP[_type] = ((prdiff*val['resp'].fs) > thresh).sum()/len(prdiff)
        SuppP[_type] = ((prdiff*val['resp'].fs) < -thresh).sum()/len(prdiff)
#    return val
#    return {'excitatory_percentage':excitatory_percentage,
#            'inhibitory_percentage':inhibitory_percentage,
#            'r_fit_linmodel':r_fit_linmodel,
#            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}
#    
    return {'thresh':thresh*val['resp'].fs,
            'EP_A':excitatory_percentage['A'],
            'EP_B':excitatory_percentage['B'],
            'EP_C':excitatory_percentage['C'],
            'EP_I':excitatory_percentage['I'],
            'IP_A':inhibitory_percentage['A'],
            'IP_B':inhibitory_percentage['B'],
            'IP_C':inhibitory_percentage['C'],
            'IP_I':inhibitory_percentage['I'],
            'OEP_A':excitatory_percentage_onset['A'],
            'OEP_B':excitatory_percentage_onset['B'],
            'OEP_C':excitatory_percentage_onset['C'],
            'OEP_I':excitatory_percentage_onset['I'],
            'OIP_A':inhibitory_percentage_onset['A'],
            'OIP_B':inhibitory_percentage_onset['B'],
            'OIP_C':inhibitory_percentage_onset['C'],
            'OIP_I':inhibitory_percentage_onset['I'],
            'Max_A':Max['A'],
            'Max_B':Max['B'],
            'Max_C':Max['C'],
            'Max_I':Max['I'],
            'OMax_A':Max_onset['A'],
            'OMax_B':Max_onset['B'],
            'OMax_C':Max_onset['C'],
            'OMax_I':Max_onset['I'],
            'TotalMax':TotalMax*val['resp'].fs,
            'SinglesMax':SinglesMax*val['resp'].fs,
            'r_lin_C':r_fit_linmodel['C'],
            'r_lin_I':r_fit_linmodel['I'],
            'MEnh_C':mean_enh['C'],
            'MEnh_I':mean_enh['I'],
            'MSupp_C':mean_supp['C'],
            'MSupp_I':mean_supp['I'],
            'EnhP_C':EnhP['C'],
            'EnhP_I':EnhP['I'],
            'SuppP_C':SuppP['C'],
            'SuppP_I':SuppP['I'],
            'SR':SR, 'SR_std':SR_std, 'SR_av_std':SR_av_std}    
    
def type_by_psth(row): 
    t=['X','X']
    thresh=.05
    if row['EP_A'] < thresh and row['IP_A'] < thresh:
        t[0] = 'O'
    elif row['EP_A'] >= thresh:
        t[0]='E'
    else:
        t[0]='I'
    if row['EP_B'] < thresh and row['IP_B'] < thresh:
        t[1] = 'O'
    elif row['EP_B'] >= thresh:
        t[1]='E'
    else:
        t[1]='I'
        
    if t.count('E')==2: #EE
        if row['EP_A'] > row['EP_B']:
            inds=np.array((0,1))
        else:
            inds=np.array((1,0))
    elif t.count('I')==2: #II
        if row['IP_A'] > row['IP_B']:
            inds=np.array((0,1))
        else:
            inds=np.array((1,0))
    elif t[0]=='E' and t[1]=='I': #EI
        inds=np.array((0,1))
    elif t[0]=='I' and t[1]=='E': #IE
        inds=np.array((1,0))
        t=['E','I']
    elif t[0]=='E' and t[1]=='O': #EO
        inds=np.array((0,1))
    elif t[0]=='O' and t[1]=='E': #OE
        inds=np.array((1,0))
        t=['E','O']
    elif t.count('O')==2: #OO
        if row['Max_A'] > row['Max_B']:
            inds=np.array((0,1))
        else:
            inds=np.array((1,0))
    else:
        #t = ['ERROR']
        #inds = None
        raise RuntimeError('Unknown type {}'.format(t))
    row['Rtype']=''.join(t)
    row['inds']=inds
    #return pd.Series({'Rtype': ''.join(t), 'inds': inds})
    return row
        
def calc_psth_weights_of_model_responses_list(val,names,signame='pred',do_plot=False,find_mse_confidence=True,get_nrmse_fn=True):
    prestimtime=1;
    duration=3
    post_duration_pad=.5
    time = np.arange(0, val[signame].extract_epoch(names[0][0]).shape[-1]) / val[signame].fs - prestimtime    
    xc_win=(time>0) & (time < (duration + post_duration_pad))
    #names = [ [n[0]] for n in names]
    sig1 = np.concatenate([val[signame].extract_epoch(n).squeeze()[xc_win] for n in names[0]])
    sig2 = np.concatenate([val[signame].extract_epoch(n).squeeze()[xc_win] for n in names[1]])
    #sig_SR=np.ones(sig1.shape)
    sigO=np.concatenate([val[signame].extract_epoch(n).squeeze()[xc_win] for n in names[2]])
      
    #fsigs=np.vstack((sig1,sig2,sig_SR)).T
    fsigs=np.vstack((sig1,sig2)).T
    ff = np.all(np.isfinite(fsigs),axis=1) & np.isfinite(sigO)
    close_to_zero = np.array([np.allclose(fsigs[ff,i], 0, atol=1e-17) for i in (0,1)])
    if any(close_to_zero):
        weights_,residual_sum,rank,singular_values = np.linalg.lstsq(np.expand_dims(fsigs[ff,~close_to_zero],1),sigO[ff],rcond=None)
        weights = np.zeros(2)
        weights[~close_to_zero] = weights_
    else:
        weights,residual_sum,rank,singular_values = np.linalg.lstsq(fsigs[ff,:],sigO[ff],rcond=None)   
    
    #calc CC
    sigF2 = np.dot(weights,fsigs[ff,:].T)
    cc = np.corrcoef(sigF2, sigO[ff])
    r_weight_model = cc[0, 1]

    norm_factor = np.std(sigO[ff])    

    min_nrmse = np.sqrt(residual_sum[0]/ff.sum())/norm_factor
    #create NMSE caclulator for later
    if get_nrmse_fn:
        def get_nrmse(weights=weights):
            sigF2 = np.dot(weights,fsigs[ff,:].T)
            nrmse = np.sqrt(((sigF2-sigO[ff]) ** 2).mean(axis=-1)) / norm_factor
            return nrmse
    else:
        get_nrmse = np.nan
        
    if not find_mse_confidence:
        weights[close_to_zero] = np.nan
        return weights, np.nan, min_nrmse, norm_factor, get_nrmse, r_weight_model
#    sigF=weights[0]*sig1 + weights[1]*sig2 + weights[2]
#    plt.figure();
#    plt.plot(np.vstack((sig1,sig2,sigO,sigF)).T)
#    wA_ = np.linspace(-2, 4, 100)
#    wB_ = np.linspace(-2, 4, 100)
#    wA, wB = np.meshgrid(wA_,wB_)
#    w=np.vstack((wA.flatten(),wB.flatten())).T
#    sigF2=np.dot(w,fsigs[ff,:].T)
#    mse = ((sigF2-sigO[ff].T) ** 2).mean(axis=1)
#    mse = np.reshape(mse,(len(wA_),len(wB_)))
#    plt.figure();plt.imshow(mse,interpolation='none',extent=[wA_[0],wA_[-1],wB_[0],wB_[-1]],origin='lower',vmax=.02,cmap='viridis_r');plt.colorbar()




    def calc_nrmse_matrix(margin,N=60,threshtype='ReChance'):
        #wsearcha=(-2, 4, 100)
        #wsearchb=wsearcha
        #margin=6
        if not hasattr(margin, "__len__"):
            margin = np.float(margin)*np.ones(2)
        wA_ = np.hstack((np.linspace(weights[0]-margin[0],weights[0],N),
                         (np.linspace(weights[0],weights[0]+margin[0],N)[1:])))
        wB_ = np.hstack((np.linspace(weights[1]-margin[1],weights[1],N),
                         (np.linspace(weights[1],weights[1]+margin[1],N)[1:])))
        wA, wB = np.meshgrid(wA_,wB_)
        w=np.stack((wA,wB),axis=2)
        nrmse = get_nrmse(w)
        #range_=mse.max()-mse.min()
        if threshtype is 'Absolute':
            thresh=nrmse.min()*np.array((1.4,1.6))
            thresh=nrmse.min()*np.array((1.02,1.04))
            As=wA[(nrmse<thresh[1]) & (nrmse>thresh[0])]
            Bs=wB[(nrmse<thresh[1]) & (nrmse>thresh[0])]
        elif threshtype is 'ReChance':
            thresh=1-(1-nrmse.min())*np.array((.952,.948))
            As=wA[(nrmse<thresh[1]) & (nrmse>thresh[0])]
            Bs=wB[(nrmse<thresh[1]) & (nrmse>thresh[0])]
        return nrmse, As, Bs, wA_, wB_
    
    if min_nrmse < 1:
        this_threshtype='ReChance'
    else:
        this_threshtype='Absolute'         
    margin=6
    As=np.zeros(0)
    Bs=np.zeros(0)
    attempt=0
    did_estimate=False
    while len(As) < 20:
        attempt+=1
        if (attempt > 1) and (len(As) > 0) and (len(As) > 2) and (not did_estimate):
            margin = np.float(margin)*np.ones(2)
            m = np.abs(weights[0]-As).max()*3
            if m==0: 
                margin[0] = margin[0]/2
            else:
                margin[0] = m
                
            m = np.abs(weights[1]-Bs).max()*3
            if m==0: 
                margin[1] = margin[1]/2
            else:
                margin[1] = m
            did_estimate = True
        elif attempt >1:
            margin = margin/2
        if attempt > 1:
            print('Attempt {}, margin = {}'.format(attempt,margin))
        nrmse, As, Bs, wA_, wB_ = calc_nrmse_matrix(margin,threshtype=this_threshtype)
        
        if attempt == 8:
            print('Too many attempts, break')
            break
        
    try:
        efit = fE.fitEllipse(As,Bs)
        center = fE.ellipse_center(efit)
        phi = fE.ellipse_angle_of_rotation(efit)
        axes = fE.ellipse_axis_length(efit)
    
        epars=np.hstack((center, axes, phi))
    except:
        print('Error fitting ellipse: {}'.format(sys.exc_info()[0]))
        print(sys.exc_info()[0])
        epars=np.full([5], np.nan)
#    idxA = (np.abs(wA_ - weights[0])).argmin()
#    idxB = (np.abs(wB_ - weights[1])).argmin()
    if do_plot:
        plt.figure();plt.imshow(nrmse,interpolation='none',extent=[wA_[0],wA_[-1],wB_[0],wB_[-1]],origin='lower',cmap='viridis_r');plt.colorbar()
        ph=plt.plot(weights[0],weights[1],Color='k',Marker='.')
        plt.plot(As,Bs,'r.')
        
        if not np.isnan(epars).any():
            a, b = axes
            R=np.arange(0,2*np.pi, 0.01)
            xx = center[0] + a*np.cos(R)*np.cos(phi) - b*np.sin(R)*np.sin(phi)
            yy = center[1] + a*np.cos(R)*np.sin(phi) + b*np.sin(R)*np.cos(phi)
            plt.plot(xx,yy,color='k')
    


#    plt.figure();plt.plot(get_nrmse(weights=(xx,yy)))
#    plt.figure();plt.plot(get_nrmse(weights=(As,Bs)))
    weights[close_to_zero] = np.nan
    return weights, epars, nrmse.min(), norm_factor, get_nrmse, r_weight_model

def calc_psth_weights_of_model_responses(val,signame='pred',do_plot=False,find_mse_confidence=True,get_nrmse_fn=True):
    #weights_C=np.ones((2,3))
    #names=['STIM_T+si464+null','STIM_T+null+si464','STIM_T+si464+si464']
    #weights_C[0,:]=calc_psth_weights_of_model_responses_single(val,names)
    #names=['STIM_T+si516+null','STIM_T+null+si516','STIM_T+si516+si516']
    #weights_C[1,:]=calc_psth_weights_of_model_responses_single(val,names)   
    
    names=[['STIM_T+si464+null','STIM_T+si516+null'],
           ['STIM_T+null+si464','STIM_T+null+si516'],
           ['STIM_T+si464+si464','STIM_T+si516+si516']]
    weights_C, Efit_C, nrmse_C, nf_C, get_nrmse_C, r_C = calc_psth_weights_of_model_responses_list(
            val,names,signame,do_plot=do_plot,find_mse_confidence=find_mse_confidence,get_nrmse_fn=get_nrmse_fn)
    if do_plot and find_mse_confidence:
        plt.title('Coherent, signame={}'.format(signame))
    #weights_I=np.ones((2,3))
    #names=['STIM_T+si464+null','STIM_T+null+si516','STIM_T+si464+si516']
    #weights_I[0,:]=calc_psth_weights_of_model_responses_single(val,names)
    #names=['STIM_T+si516+null','STIM_T+null+si464','STIM_T+si516+si464']
    #weights_I[1,:]=calc_psth_weights_of_model_responses_single(val,names)
    
    names=[['STIM_T+si464+null','STIM_T+si516+null'],
           ['STIM_T+null+si516','STIM_T+null+si464'],
           ['STIM_T+si464+si516','STIM_T+si516+si464']]
    weights_I, Efit_I, nrmse_I, nf_I, get_nrmse_I, r_I = calc_psth_weights_of_model_responses_list(
            val,names,signame,do_plot=do_plot,find_mse_confidence=find_mse_confidence,get_nrmse_fn=get_nrmse_fn)
    if do_plot and find_mse_confidence:
        plt.title('Incoherent, signame={}'.format(signame))
    
    D=locals()
    D={k: D[k] for k in ('weights_C', 'Efit_C', 'nrmse_C', 'nf_C', 'get_nrmse_C', 'r_C',
                         'weights_I', 'Efit_I', 'nrmse_I', 'nf_I', 'get_nrmse_I', 'r_I')}
    return D
    #return weights_C, Efit_C, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I

def show_img(cellid,ax=None,ft=1,subset='A+B+C+I',modelspecname='dlog_fir2x15_lvl1_dexp1',
             loader='env.fs100-ld-sev-subset.A+B+C+I',fitter='fit_basic',pth=None,
             ind=None,modelname=None,fignum=0,batch=306):
    ax_=None
    if pth is None:
        print('pth is None')
        if ft==0:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/types/'
            pth=pth + cellid + '_env100_subset_'+subset+'.'+modelspecname+'_all_val+FIR.png'
        elif ft==1:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/'
            pth=pth + cellid + '.png'
        elif ft==11:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/Overlay/'
            pth=pth + cellid + '.png'
        elif ft==2:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/'
            pth=pth + cellid + '.pickle'
        elif ft==3:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/normalization_branch/'
            #subset='A+B+C+I'
            #subset='I'
            pth=pth + cellid + '_env100_subset_'+subset+'.'+modelspecname+'.png'
            if type(ax) == list:
                print('list!')
                ax_=ax
                ax=ax_[0]
                pth2=pth.replace('.png','_all_val.png')
            else:
                pth=pth.replace('.png','_all_val.png')
        elif ft==4:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/svd_fs_branch/'
            #subset='A+B+C+I'
            #subset='I'
            #pth=pth + cellid + '_env.fs100-ld-sev-st.coh-subset.'+subset+'_'+modelspecname+'_all_val.png'
            #pth=pth + cellid + '_'+loader+'_'+modelspecname+'_all_val.png'
            if len(fitter)==0:
                pth=pth + cellid + '_'+loader+'_'+modelspecname+'.png'
            else:
                pth=pth + cellid + '_'+loader+'_'+modelspecname+'_'+fitter+'.png'
            if type(ax) == list:
                print('list!')
                ax_=ax
                ax=ax_[0]
                pth2=pth.replace('.png','_all_val.png')
            else:
                pth=pth.replace('.png','_all_val.png')
        elif ft==5:
            pth=os.path.join(nd.get_results_file(batch,[modelname],[cellid])['modelpath'][0],
                         'figure.{:04d}.png'.format(fignum))
        elif ft==6:
            pth='/auto/users/luke/Projects/SPS/plots/NEMS/types/PSTH/Overlay/{}.png'.format(cellid)
           
    else:
        pth=pth[ind]
        print('pth is {}, ind is {}'.format(pth,ind))
        if type(pth) == list:
            pth_=pth;
            pth=pth_[0]
            pth2=pth_[1]
            print('pth1 is {} in ind {}'.format(pth,ind))
            print('pth2 is {} in ind {}'.format(pth2,ind))
            ax_=ax
            ax=ax_[0]
        else:
            print('{} in ind {}'.format(pth,ind))
            if type(ax) == list:
                print('list!')
                ax_=ax
                ax=ax_[0]
                pth2=pth.replace('.png','_all_val.png')
            else:
                pth=pth.replace('.png','_all_val.png')
    print(pth)
    if pth.split('.')[1] == 'pickle':
        ax.figure=pl.load(open(pth,'rb'))
    elif ax is None:
        ax=display_image_in_actual_size(pth)
    else:
        im_data = plt.imread(pth)
        ax.clear()
        ax.imshow(im_data,interpolation='bilinear')
    ax.figure.canvas.draw()
    ax.figure.canvas.show()
    
    if type(ax_) == list:
        im_data = plt.imread(pth2)
        ax_[1].clear()
        print(pth2);print(ax_[1])
        ax_[1].imshow(im_data,interpolation='bilinear')
        ax_[1].figure.canvas.draw()
        ax_[1].figure.canvas.show()

def display_image_in_actual_size(im_path,ax=None):

    dpi = 150
    im_data = plt.imread(im_path)
    height, width, depth = im_data.shape

    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray',interpolation='bilinear')

    plt.show()
    
    return ax
    
def generate_weighted_model_signals(sig_in,weights,epcs_offsets):
    sig_out=sig_in.copy()
    sig_out._data = np.full(sig_out._data.shape, np.nan)
    types=['C','I']
    epcs=sig_in.epochs[sig_in.epochs['name'].str.contains('STIM')].copy()
    epcs['type']=epcs['name'].apply(parse_stim_type)
    orig_epcs=sig_in.epochs.copy()
    sig_in.epochs['start']=sig_in.epochs['start']+epcs_offsets[0]
    sig_in.epochs['end']=sig_in.epochs['end']+epcs_offsets[1]
    EA=np.array([n.split('+')[1] for n in epcs['name']])
    EB=np.array([n.split('+')[2] for n in epcs['name']])
    corrs={}
    print(epcs)
    print(types)
    print(weights)
    for _weights,_type in zip(weights,types):
        from pdb import set_trace
        set_trace() 
        inds=np.nonzero(epcs['type'].values == _type)[0]
        for ind in inds:
            r=sig_in.extract_epoch(epcs.iloc[ind]['name'])
            if np.any(np.isfinite(r)):
                indA = np.where((EA[ind] == EA) & (EB == 'null'))[0]
                indB = np.where((EB[ind] == EB) & (EA == 'null'))[0]
                if (len(indA) > 0) & (len(indB) > 0):
                    rA=sig_in.extract_epoch(epcs.iloc[indA[0]]['name'])
                    rB=sig_in.extract_epoch(epcs.iloc[indB[0]]['name'])
                    sig_out=sig_out.replace_epoch(epcs.iloc[ind]['name'],_weights[0]*rA+_weights[1]*rB,preserve_nan=False)
                    R=sig_out.extract_epoch(epcs.iloc[ind]['name'])
        
        
        ins=sig_in.extract_epochs(epcs.iloc[inds]['name'])
        ins=np.hstack([ins[k] for k in ins.keys()]).flatten()
        outs=sig_out.extract_epochs(epcs.iloc[inds]['name'])
        outs=np.hstack([outs[k] for k in outs.keys()]).flatten()
        ff = np.isfinite(ins) & np.isfinite(outs)    
        cc = np.corrcoef(ins[ff], outs[ff])
        corrs[_type]=cc[0,1]
    sig_in.epochs=orig_epcs
    sig_out.epochs=orig_epcs.copy()
    return sig_out, corrs

def plot_linear_and_weighted_psths(batch,cellid,weights=None,subset=None,rec_file=None):
    #options = {}
    #options['cellid']=cellid
    #options['batch']=batch
    #options["stimfmt"] = "envelope"
    #options["chancount"] = 0
    #options["rasterfs"] = 100
    #rec_file=nb.baphy_data_path(options)  


    #from pdb import set_trace
    #set_trace() 
    if rec_file is None:
        rec_file = nw.generate_recording_uri(cellid, batch, loadkey='ns.fs100')
    rec=recording.load_recording(rec_file)
    rec['resp'] = rec['resp'].extract_channels([cellid])
    rec['resp'].fs=200
    
    epcs=rec['resp'].epochs[rec['resp'].epochs['name'] == 'PreStimSilence'].copy()
    spike_times=rec['resp']._data[cellid]
    count=0
    for index, row in epcs.iterrows():
        count+=np.sum((spike_times > row['start']) & (spike_times < row['end']))
    SR=count/(epcs['end']-epcs['start']).sum()
     
    #COMPUTE ALL FOLLOWING metrics using smoothed driven rate
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    
    #smooth and subtract SR
    fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - SR/rec['resp'].fs)
    val['resp']=val['resp'].transform(fn)
    val['resp']=sp.add_stimtype_epochs(val['resp'])

    lin_weights=[[1,1],[1,1]]
    epcs_offsets=[epcs['end'].iloc[0], 0]
    val['lin_model'], l_corrs=sp.generate_weighted_model_signals(val['resp'],lin_weights,epcs_offsets)    
    if weights is None:
        sigz=['resp','lin_model']
        plot_singles_on_dual=True
        plot_singles_on_dual=False
        w_corrs=None
    else:
        val['weighted_model'], w_corrs=sp.generate_weighted_model_signals(val['resp'],weights,epcs_offsets)
        sigz=['resp','lin_model','weighted_model']
        plot_singles_on_dual=False
    from pdb import set_trace
    set_trace() 
    fh=plot_all_vals(val,None,signames=sigz,channels=[0,0,0],subset=subset,plot_singles_on_dual=plot_singles_on_dual)
    return fh, w_corrs, l_corrs
    


def plot_psth_weights(df,ax=None,s='R',lengthscale=10,fnargs=None,fn=None,norm_method=None):       
    if type(df) is pd.Series:
        df  = pd.DataFrame(df).T
    if ax is None:
        ax = plt.gca()
        
    def center_on_C(weights):
        w=weights.copy()
        w=w-w[0,:]
        return w
    def center_on_I(weights):
        w=weights.copy()
        w=w-w[1,:]
        return w
    def no_norm(weights):
        return weights.copy()
    if norm_method is None:
        norm=no_norm
    else:
        norm=locals()[norm_method]
    R=np.arange(0,2*np.pi, 0.01)
    for index, row in df.iterrows():
        weights=np.vstack([row['weights_C'+s],row['weights_I'+s]])
        weights=norm(weights)
        ax.plot(weights[:,0],weights[:,1])
        
        try:
            efit=np.vstack((row['Efit_C'+s],row['Efit_I'+s]))
            efit[:,:2]=norm(efit[:,:2])
            
            if row['nrmse_CR'] < 1:
                ls='-'
            else:
                ls='--'
            
            centers=efit[0,0:2]
            lengths=efit[0,2:4].copy()/lengthscale
            phi=efit[0,4]
            xx = centers[0] + lengths[0]*np.cos(R)*np.cos(phi) - lengths[1]*np.sin(R)*np.sin(phi)
            yy = centers[1] + lengths[0]*np.cos(R)*np.sin(phi) + lengths[1]*np.sin(R)*np.cos(phi)
            ax.plot(xx,yy,color='k',linewidth=.5,linestyle=ls)
        
            if row['nrmse_IR'] < 1:
                ls='-'
            else:
                ls='--'
            
            centers=efit[1,0:2]
            lengths=efit[1,2:4].copy()/lengthscale
            phi=efit[1,4]
            xx = centers[0] + lengths[0]*np.cos(R)*np.cos(phi) - lengths[0]*np.sin(R)*np.sin(phi)
            yy = centers[1] + lengths[0]*np.cos(R)*np.sin(phi) + lengths[1]*np.sin(R)*np.cos(phi)
            ax.plot(xx,yy,color='r',linewidth=.5,linestyle=ls)
        except:
            pass
    #w=np.array(df['weights_C'+s].tolist())
    weights=np.swapaxes(np.stack((df['weights_C'+s].tolist(),df['weights_I'+s].tolist()),axis=2),0,2)
    weights=norm(weights)
    if df.index.name is None:
        names=df['cellid'].values.tolist()
    else:
        names=df.index.values.tolist()
    phc=sp.scatterplot_print(weights[0,0,:], weights[0,1,:], names,
                             ax=ax,color='k',markersize=8,fn=fn,fnargs=fnargs)
    #w=np.array(df['weights_I'+s].tolist())
    phi=sp.scatterplot_print(weights[1,0,:], weights[1,1,:], names,
                             ax=ax,color='r',markersize=8,fn=fn,fnargs=fnargs)
    
    ax.plot([0,1],[0,1],'k')
    ax.plot([0,0],[0,1],'k')
    ax.plot([0,1],[0,0],'k')
    
    return phc,phi

def plot_weighted_psths_and_weightplot(row,weights,batch=306):
    weights2=[w[row['inds']] for w in weights]
    fh,w_corrs,l_corrs=plot_linear_and_weighted_psths(batch,row.name,weights2)
    
    ax=fh.axes
    for ax_ in ax:
            pos=ax_.get_position()
            pos.y0=pos.y0-.08
            pos.y1=pos.y1-.08
            ax_.set_position(pos)
    axN=fh.add_axes([.3,.84,.4,.16])
    phcR,phiR=plot_psth_weights(row,ax=axN,lengthscale=1)
    #phcR,phiR=plot_psth_weights(row2,ax=ax,lengthscale=1)
    phc,phi=plot_psth_weights(row,ax=axN,s='')
    phc.set_marker('*');phi.set_marker('*')
    axN.plot(weights[0][0],weights[0][1],'ok')
    axN.plot(weights[1][0],weights[1][1],'or')
    axN.set_xlabel('Voice a')
    axN.set_ylabel('Voice b')
    meta_str='{}\nRType: {}, Pri: {}'.format(row.name,row['Rtype'],['A','B'][row['inds'][0]])
    corr_str='Weighted Correlations:\nC: {:.2f} ({:.2f})\nI : {:.2f} ({:.2f})'.format(
            w_corrs['C'],row['r_CR'],w_corrs['I'],row['r_IR'])
    xv=fh.axes[-1].get_xlim()[1]
    yv=fh.axes[-1].get_ylim()[1]
    fh.axes[-1].text(xv,yv,meta_str+'\n'+corr_str,verticalalignment='top')
    return fh

def calc_psth_weight_model(model,celldf=None,do_plot=False,modelspecs_dir='/auto/users/luke/Code/nems/modelspecs/normalization_branch'):
    cellid=model['cellid']
    cell=celldf.loc[cellid]
    print('load {}, {}'.format(cellid,model['modelspecname']))
    modelspecs,est,val = sp.load_SPO(cellid,
                                      ['A','B','C','I'],
                                      model['modelspecname'],fs=200,
                                      modelspecs_dir=modelspecs_dir)
    #smooth and subtract SR
    fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - cell['SR']/val[0]['resp'].fs)
    
    #fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2)*val[0]['resp'].fs - row['SR'])
    val[0]['resp']=val[0]['resp'].transform(fn)
    
    #calc SR of pred
    ps=est[0]['pred'].select_epochs(['PreStimSilence']).as_continuous()
    ff = np.isfinite(ps)
    SR_model=ps[ff].mean()*val[0]['pred'].fs
    
    fn = lambda x : np.atleast_2d(x.squeeze() - SR_model/val[0]['pred'].fs)
    val[0]['pred']=val[0]['pred'].transform(fn)
    print('calc weights')
    #weights_CR_,weights_IR_=sp.calc_psth_weights_of_model_responses(val[0],signame='resp')
    #weights_CR_,weights_IR_,Efit_CR_,Efit_IR_=sp.calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    #weights_CR_, Efit_C_, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I
    #d=sp.calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    #d={k+'R': v for k, v in d.items()}
    #for k, v in d.items():
    #    row[k]=v
    dat=sp.calc_psth_weights_of_model_responses(val[0],do_plot=do_plot,find_mse_confidence=False,get_nrmse_fn=False)
    for k, v in dat.items():
        model[k]=v
        
    if cell['get_nrmse_IR'] is None:
        raise RuntimeError("Function cell['get_nrmse_IR'] is none.")
    else:
        model['LN_nrmse_ratio_I']=(1-cell['get_nrmse_IR'](model['weights_I'])) / (1 - cell['nrmse_IR'])
        model['LN_nrmse_ratio_C']=(1-cell['get_nrmse_CR'](model['weights_C'])) / (1 - cell['nrmse_CR'])
    return model

def calc_psth_weight_cell(cell,do_plot=False,modelspecs_dir='/auto/users/luke/Code/nems/modelspecs/normalization_branch',get_nrmse_only=False):
    cellid=cell.name
    if get_nrmse_only and (cell['get_nrmse_CR'] is not None) and (cell['get_nrmse_IR'] is not None):
        print('get_nrmse_CR and get_nrmse_IR already exist for {}, skipping'.format(cellid))
        return cell
    print('load {}'.format(cellid))
    
    modelspecs,est,val = sp.load_SPO(cellid,['A','B','C','I'],None,fs=200,get_est=False,get_stim=False)
     
    #smooth and subtract SR
    fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2) - cell['SR']/val[0]['resp'].fs)
    
    #fn = lambda x : np.atleast_2d(sp.smooth(x.squeeze(), 3, 2)*val[0]['resp'].fs - row['SR'])
    val[0]['resp']=val[0]['resp'].transform(fn)
    
    
    print('calc weights')
    #weights_CR_,weights_IR_=sp.calc_psth_weights_of_model_responses(val[0],signame='resp')
    #weights_CR_,weights_IR_,Efit_CR_,Efit_IR_=sp.calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot)
    #weights_CR_, Efit_C_, nmse_C, nf_C, get_mse_C, weights_I, Efit_I, nmse_I, nf_I, get_mse_I
    d=sp.calc_psth_weights_of_model_responses(val[0],signame='resp',do_plot=do_plot,find_mse_confidence = (not get_nrmse_only))
    if get_nrmse_only:
        cell['get_nrmse_CR']=d['get_nrmse_C']
        cell['get_nrmse_IR']=d['get_nrmse_I']
    else:
        d={k+'R': v for k, v in d.items()}
        for k, v in d.items():
            cell[k]=v
    
    return cell