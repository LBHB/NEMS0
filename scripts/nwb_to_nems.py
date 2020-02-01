#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:31:38 2020

@author: ekeppler
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession

from allensdk.brain_observatory.ecephys import nwb  # for compat
import pynwb

import nems
from nems.recording import Recording
from nems.signal import PointProcess, RasterizedSignal, SignalBase
from nems.plots.raster import raster
import nems.epoch as ep


#copied from nems lbhb  - edit for pupil data

#@classmethod
def from_nwb_pupil(nwb_file, nwb_format,fs=20,with_pupil=False,running_speed=False,as_dict=True):
#def from_nwb(cls, nwb_file, nwb_format,with_pupil=False,fs=20):
    """
    The NWB (Neurodata Without Borders) format is a unified data format developed by the Allen Brain Institute.
    Data is stored as an HDF5 file, with the format varying depending how the data was saved.
    
    References:
      - https://nwb.org
      - https://pynwb.readthedocs.io/en/latest/index.html
    :param nwb_file: path to the nwb file
    :param nwb_format: specifier for how the data is saved in the container
    :param int fs: will match for all signals
    :param bool with_pupil, running speed: whether to return pupil, speed signals in recording
    :param bool as_dict: return a dictionary of recording objects, each corresponding to a single unit/neuron
                         else a single recording object w/ each unit corresponding to a channel in pointprocess signal
    :return: a recording object
    """
    #log.info(f'Loading NWB file with format "{nwb_format}" from "{nwb_file}".')

    # add in supported nwb formats here
    assert nwb_format in ['neuropixel'], f'"{nwb_format}" not a supported NWB file format.'

    nwb_filepath = Path(nwb_file)
    if not nwb_filepath.exists():
        raise FileNotFoundError(f'"{nwb_file}" could not be found.')

    if nwb_format == 'neuropixel':
        """
        In neuropixel ecephys nwb files, data is stored in several attributes of the container: 
          - units: individual cell metadata, a dataframe
          - epochs: timing of the stimuli, series of arrays
          - lab_meta_data: metadata about the experiment, such as specimen details
          
        Spike times are saved as arrays in the 'spike_times' column of the units dataframe as xarrays. 
        The frequency defaults to match pupil - if no pupil data retrieved, set to chosen value (previous default 1250).
          
        Refs:
          - https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html
          - https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quickstart.html
          - https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_data_access.html
        """
        try:
            from pynwb import NWBHDF5IO
            from allensdk.brain_observatory.ecephys import nwb  # needed for ecephys format compat
        except ImportError:
            m = 'The "allensdk" library is required to work with neuropixel nwb formats, available on PyPI.'
            #log.error(m)
            raise ImportError(m)

        session_name = nwb_filepath.stem
        with NWBHDF5IO(str(nwb_filepath), 'r') as nwb_io:
            nwbfile = nwb_io.read()

            units = nwbfile.units
            epochs = nwbfile.epochs
           
            spike_times = dict(zip(units.id[:].astype(str), units['spike_times'][:]))

            # extract the metadata and convert to dict
            metadata = nwbfile.lab_meta_data['metadata'].to_dict()
            metadata['uri'] = str(nwb_filepath)  # add in uri
            #add invalid times data to meta as df if exist - includes times and probe id?
            if nwbfile.invalid_times is not None:
                invalid_times = nwbfile.invalid_times
                invalid_times =  np.array([invalid_times[col][:] for col in invalid_times.colnames])
                metadata['invalid_times'] = pd.DataFrame(invalid_times.transpose(),columns=['start_time', 'stop_time', 'tags'])
                
            # build the units metadata
            units_data = {
                col.name: col.data for col in units.columns
                if col.name not in ['spike_times', 'spike_times_index', 'spike_amplitudes',
                                    'spike_amplitudes_index', 'waveform_mean', 'waveform_mean_index']
            }

            # needs to be a dict
            units_meta = pd.DataFrame(units_data, index=units.id[:])
            #add electrode info to units meta
            electrodes=nwbfile.electrodes
            e_data = {col.name: col.data for col in electrodes.columns}
            e_meta = pd.DataFrame(e_data,index=electrodes.id[:])
            units_meta=pd.merge(units_meta,e_meta,left_on=units_meta.peak_channel_id,right_index=True, 
                                suffixes=('_unit','_channel')).drop(['key_0','group'],axis=1).to_dict('index')# needs to be a dict    

            # build the epoch dataframe
            epoch_data = {
                col.name: col.data for col in epochs.columns
                if col.name not in ['tags', 'timeseries', 'tags_index', 'timeseries_index']
            }

            epoch_df = pd.DataFrame(epoch_data, index=epochs.id[:]).rename({
                'start_time': 'start',
                'stop_time': 'end',
                'stimulus_name': 'name'
            }, axis='columns')

 
            #rename epochs to correspond to different nat scene/movie frames - 
            epoch_df.loc[epoch_df['frame'].notna(),'name'] = epoch_df.loc[epoch_df['frame'].notna(),'name'] + '_' + \
            epoch_df[epoch_df['frame'].notna()].iloc[:]['frame'].astype(int).astype(str)
            
            
            #drop extra columns
            metadata['epochs']=epoch_df #save extra stim info to meta
            epoch_df=epoch_df.drop([col for col in epoch_df.columns if col not in ['start','end','name']],axis=1)

#            #rename natural scene epochs to work w/demo
            df_copy = epoch_df[epoch_df.name.str.contains('natural_scene')].copy()
            df_copy.loc[:,'name']='REFERENCE'

            epoch_df=epoch_df.append(df_copy,ignore_index=True)
            #expand epoch bounds epochs will overlap to test evoked potential
#            to_adjust=epoch_df.loc[:,['start','end']].to_numpy()
#            epoch_df.loc[:,['start','end']] = nems.epoch.adjust_epoch_bounds(to_adjust,-0.1,0.1)
            
            
            # save the spike times as a point process signal frequency set to match other signals 
            pp = PointProcess(fs, spike_times, name='resp', recording=session_name, epochs=epoch_df,
                              chans=[str(c) for c in nwbfile.units.id[:]],meta=units_meta)
            #dict to pass to recording
            #signal_dict = {pp.name: pp}

          #  log.info('Successfully loaded nwb file.')
            from scipy.interpolate import interp1d
           #save pupil data as rasterized signal
            if with_pupil:
                try:
                    pupil = nwbfile.modules['eye_tracking'].data_interfaces['pupil_ellipse_fits']
                    t = pupil['timestamps'][:]
                    pupil = pupil['width'][:].reshape(1,-1) #only 1 dimension - or get 'height'
                    
                     #interpolate to set sampling rate
                    f = interp1d(t,pupil,bounds_error=False,fill_value=np.nan)

                    new_t = np.arange(0.0,(t.max()+1/fs),1/fs)#pupil data starting at timepoint 0.0 (nan filler)
                    pupil = f(new_t)
                    
                    pupil_signal = RasterizedSignal(fs=fs,data=pupil,recording=session_name,name='pupil',
                                                    epochs=epoch_df,chans=['pupil']) #for all data list(pupil_data.colnames[0:5])
                    
                #if no pupil data for session - still get spike data
                except KeyError:
                    print(session_name + ' has no pupil data.')

            
            if running_speed:
                running = nwbfile.modules['running'].data_interfaces['running_speed']
                t = running.timestamps[:][1]#data has start and end timestamps, here only end used
                running = running.data[:].reshape(1,-1)

                f = interp1d(t,running)
                #new_t = np.arange(np.min(t),np.max(t),1/fs)
                new_t = np.arange(epoch_df.start.min(),epoch_df.end.max(),1/fs)
                running = f(new_t)
                running=RasterizedSignal(fs=fs,data=running,name='running',recording=session_name,epochs=epoch_df)



            if as_dict:
                #each unit has seperate recording in dict
                rec_dict={}
                for c in pp.chans:
                    unit_signal=pp.extract_channels([c])
                    rec=Recording({'resp':unit_signal},meta=metadata)
                    if with_pupil:
                        rec.add_signal(pupil_signal)
                    if running_speed:
                        rec.add_signal(running)
                    rec_dict[c]=rec
                return rec_dict
            
            else:
                rec=Recording({'resp':pp},meta=metadata)
                if with_pupil:
                    rec.add_signal(pupil_signal)
                if running_speed:
                    rec.add_signal(running)
                return rec
                
#            return cls(signal_dict, meta=metadata)

#file from downloaded session data            
nwb_filepath = Path('/auto/users/tomlinsa/code/allen/data/session_759883607/session_759883607.nwb')
#recording dict 
rec=from_nwb_pupil(nwb_filepath,'neuropixel',with_pupil=True,fs=40)
r=from_nwb_pupil(nwb_filepath,'neuropixel',with_pupil=True,fs=40,as_dict=False)
rec=rec['951762820']
#filter by metadata to get subset of units metadata for each recording in dict is for all units   
meta_df=pd.DataFrame.from_dict(rec['resp'].meta).T
#specific area+allen sdk default filters
vp=meta_df[(meta_df.manual_structure_acronym=='VISp')&(meta_df['isi_violations'] < 0.5) & (meta_df['amplitude_cutoff'] < 0.1) &
        (meta_df['presence_ratio'] > 0.9) & (meta_df['quality']=='good')].index.values
#rasterizesubset of units if >1 unit in each recording
resp=r['resp'].extract_channels([str(c) for c in vp]).rasterize() 

 #raster for single unit over repeated epoch occurances - for recording w/ multiple units       
def epoch_channel_raster(epoch_name,chan_idx,resp):
    chan =np.squeeze(resp.extract_channels([resp.chans[chan_idx]]).extract_epoch(epoch_name),1) 
    #raster of spike rate for single unit accross repeated stim presentation
    raster.raster(times=np.linspace(0,450,chan.shape[1]),values=np.nan_to_num(chan)) #nan as 0.0
    return chan


def mean_evoked_epochs(resp,epoch_regex="^natural_scene"): #avg difference by neuron btwn peak fr and min accross all epochs
    #epoch_regex="^natural_scene"
    all_epochs=all_nwb_epochs(resp,epoch_regex)
    psth_resp=np.nanmean(all_epochs,0)
    evoked=np.nanmax(psth_resp,1)-np.nanmin(psth_resp,1)
    #plt.plot(evoked,'.')
    return evoked

#pupil    
#pupil=np.squeeze(r['pupil'].extract_epoch('natural_scenes_15.0'),1)
#plt.plot(pupil.T,'.') #over each epoch occurance
##plot pupil and firing for single epoch occurance
#channels=resp.chans
##get single channel
#chan =np.squeeze(rras.extract_channels([channels[1]]).extract_epoch('natural_scenes_17.0'),1)
#plt.plot(np.linspace(0,450,chan.shape[1]),chan[10,:],'.',np.linspace(0,450,chan.shape[1]),pupil[10,:],'.')
##pupil v channel
#plt.plot(pupil,chan,'.')
##means
##plot fr avg over epochs
#plt.plot(np.nanmean(chan,0))
#plt.plot(np.nanmean(chan[:,:],1))
#plt.plot(np.nanmean(pupil,1))

#extract epochs for units into array
#epoch_regex="^natural_scene"
def all_nwb_epochs(resp,epoch_regex):
    #return epoch data as array - helpful if recording includes multiple units?
    epoch_list=ep.epoch_names_matching(resp.epochs, epoch_regex)
    epoch_dict=resp.extract_epochs(epoch_list)
    all_epochs=[epoch_dict[key] for key in epoch_dict.keys()]
    e_size=np.max([i.shape[2] for i in all_epochs])
    #merge everything together - resulting array has shape trials*stimsxunitsxsamples
    all_epochs=np.concatenate([np.resize(i,(50,len(resp.chans),e_size)) for i in all_epochs],0) 
    return all_epochs

def spike_dev_by_pupil(rec,rras):
    #mean dev from psth per trial and mean pupil pertrial accross all epochs - return df of correlation coeff
    #will return data for multiple units if 
    epochs=rec.epochs
    if rras is None:
        rras=rec['resp'].rasterize()
    #extract all epochs for pupil
    epoch_regex="^natural_scene"
    epoch_list=ep.epoch_names_matching(epochs, epoch_regex)
    pupil_epochs=rec['pupil'].extract_epochs(epoch_list)
    pupil_epochs=[pupil_epochs[key] for key in pupil_epochs.keys()]
    e_size=np.max([i.shape[2] for i in pupil_epochs])
    pupil_epochs=np.squeeze(np.concatenate([np.resize(i,(50,1,e_size)) for i in pupil_epochs],0),1)
    #mean pupil over time for each stim,trial
    meanpupil=np.nanmean(pupil_epochs,1).reshape(119,50)
    
    #extract all natural_scene epochs then merge dict into array along trial num axis
    epoch_dict=rras.extract_epochs(epoch_list)
    all_epochs=[epoch_dict[key] for key in epoch_dict.keys()]
    e_size=np.max([i.shape[2] for i in all_epochs])
    #merge everything together - resulting array has shape trials*stimsxunitsxsamples
    all_epochs=np.concatenate([np.resize(i,(50,len(rras.chans),e_size)) for i in all_epochs],0) 

    
    from scipy.stats import pearsonr
      
    #iterate over cells in signal
    respdev_dict={}
    for i in range(all_epochs.shape[1]):
        chan=all_epochs[:,i,:].reshape(119,50,-1)
        meanresp_pertrial=np.nanmean(chan,1) #avg accross trials for each epoch
        respdev_dict[rras.chans[i]]=np.nanmean((chan-np.expand_dims(meanresp_pertrial,1)),2)
    
    pupilnan=(~np.isnan(meanpupil)) 
    dev_corr = {}
    #df of pupil, resp dev correlation
    for key in respdev_dict.keys():
        resp = respdev_dict[key]
        corr_index=((~np.isnan(resp))&pupilnan) #remove nans for correlation
        (coeff,pval) = pearsonr(meanpupil[corr_index],resp[corr_index]) 
        dev_corr[key]={'coeff':coeff,'pval':pval}
    dev_corr=pd.DataFrame.from_dict(dev_corr)
    return dev_corr

    


def nwb_resp_psth(rec,epoch_regex):
#intended to give similar output generate_psth_from_resp from preprocessing model, but works better w/ structure
#of neuropixels data
    newrec=rec.copy()

    resp=newrec['resp'].rasterize()

    #epoch_regex="^natural_scene"
    #extract all natural_scene epochs then merge dict and avg - add new signal
    epochs_to_extract = ep.epoch_names_matching(resp.epochs, epoch_regex)
    #epoch_dict=resp.extract_epochs(epochs.loc[epochs.name.str.contains('natural_scene'),'name'])
    epoch_dict=resp.extract_epochs(epochs_to_extract)
    
    #no pre/post stim silence but spontaneous intervals work instead? - or use psth w/out mean subtracted
    spont = resp.extract_epoch('spontaneous')
    spont_rate=np.nanmean(spont)
    
    per_stim_psth_spont = {}
    per_stim_psth = {}
    for k, v in epoch_dict.items():
        per_stim_psth_spont[k] = np.nanmean(v, axis=0)
        per_stim_psth[k] = np.nanmean(v, axis=0) - spont_rate
        
    respavg = resp.replace_epochs(per_stim_psth)
    respavg.name = 'psth'
    respavg_data = respavg.as_continuous().copy()

    respavg_with_spont = resp.replace_epochs(per_stim_psth_spont)
    respavg_with_spont.name = 'psth_sp'
    respavg_spont_data = respavg_with_spont.as_continuous().copy()
    
    respavg = respavg._modified_copy(respavg_data)
    respavg_with_spont = respavg_with_spont._modified_copy(respavg_spont_data)
    
    newrec.add_signal(respavg)
    newrec.add_signal(respavg_with_spont)
    return newrec

