#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 10:20:39 2020

@author: ekeppler
"""

# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging
import random
import copy

import numpy as np
import matplotlib.pyplot as plt

USE_GUI=False

if USE_GUI:
    import nems.gui.editors as gui
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
import nems.uri
from nems import recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
from nems.metrics.state import single_state_mod_index

#from nwb recording fns:
from pathlib import Path
import json

import numpy as np
import pandas as pd

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

def nwb_resp_psth(rec,epoch_regex):
#intended to give similar output generate_psth_from_resp from preprocessing model, but works better w/ structure
#of neuropixels data?
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
# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

nems_dir = os.path.abspath(os.path.dirname(recording.__file__) + '/..')
signals_dir = nems_dir + '/recordings'
modelspecs_dir = nems_dir + '/modelspecs'

#cellid = "TAR010c-06-1"
#recording_uri = os.path.join(signals_dir, cellid + ".tgz")

cellid = '951762820' #allen
# ----------------------------------------------------------------------------
# DATA LOADING

# GOAL: Get your data loaded into memory as a Recording object
logging.info('Loading data...')

# Method #1: Load the data from a local directory
# download demo data if necessary:
#recording.get_demo_recordings(signals_dir)
#rec = recording.load_recording(recording_uri)
nwb_filepath = Path('/auto/users/tomlinsa/code/allen/data/session_759883607/session_759883607.nwb')
rec=from_nwb_pupil(nwb_filepath,'neuropixel',with_pupil=True,fs=40)
rec=rec[cellid]

 #try out allen data

# ----------------------------------------------------------------------------
# PREPROCESSING

# create a new signal that will be used to modulates the output of the linear
# predicted response
logging.info('Generating state signal...')
#rec = preproc.make_state_signal(rec, ['active','pupil'], [''], 'state')
rec = preproc.make_state_signal(rec, state_signals=['pupil'],
                               permute_signals=[''], new_signalname='state')
#rec = preproc.make_state_signal(rec, state_signals=['pupil'],
#                                permute_signals=['pupil'], new_signalname='state') #shuffled pupil

# mask out data from incorrect trials
#rec = preproc.mask_all_but_correct_references(rec)#not for allen?

# calculate a PSTH response for each stimulus, save to a new signal 'psth'
#epoch_regex="^STIM_"
epoch_regex="^natural_scene" #only natural scene stimulus
#rec = preproc.generate_psth_from_resp(rec, epoch_regex=epoch_regex, smooth_resp=False)
rec=nwb_resp_psth(rec,epoch_regex) #should return similar results as above for allen data

# ----------------------------------------------------------------------------
# INSPECT THE DATA

resp = rec['resp'].rasterize()
epochs = resp.epochs
#epoch_regex="^STIM_"
#epoch_regex="^natural_scene" #only natural scene stimulus
epoch_list = ep.epoch_names_matching(epochs, epoch_regex)

# list all stimulus events
print(epochs[epochs['name'].isin(epoch_list)])

# list all events of a single stimulus
epoch = 'natural_scenes_17'#epoch_list[50]
print(epochs[epochs['name'] == epoch])

# extract raster of all these events on correct or passive trials
# use rec['mask'] to remove all incorrect trial data
#raster = resp.extract_epoch(epoch, mask=rec['mask'])[:,0,:]
raster = resp.extract_epoch(epoch)[:,0,:]#no mask for allen
t = np.arange(raster.shape[1]) /resp.fs

plt.figure()
plt.subplot(2,1,1)
plt.imshow(raster, interpolation='none', aspect='auto',
           extent=[t[0], t[-1], raster.shape[0], 0])
plt.title('Raster for {}'.format(epoch))

plt.subplot(2,1,2)
plt.plot(t, np.nanmean(raster, axis=0))
plt.title('PSTH for {}'.format(epoch))

# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

logging.info('Initializing modelspec...')
modelname = 'stategain.S'
meta = {'cellid': cellid, 'modelname': modelname}

# Method #1: create from "shorthand" keyword string
modelspec = nems.initializers.from_keywords(modelname, rec=rec, meta=meta,
                                            input_name='psth_sp')
#don't subtract spont mean w/allen?

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.
logging.info('Generating jackknife datasets for n-fold cross-validation...')

# create all jackknife sets. the single recording, rec, is now turned into
# lists of recordings for estimation (est) and validation (val). Size of
# signals in each set are the same, but the excluded segments are set to nan.
nfolds = 10
est, val, m = preproc.mask_est_val_for_jackknife(rec, modelspecs=None,
                                                 njacks=nfolds)


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

logging.info('Fitting modelspec(s)...')

modelspec.tile_jacks(nfolds)
for jack_index, e in enumerate(est.views()):
    logging.info("Fitting JK {}/{}".format(jack_index+1, nfolds))
    modelspec.jack_index = jack_index
    modelspec = nems.analysis.api.fit_basic(e, modelspec, fitter=scipy_minimize)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

logging.info('Saving Results...')
ms.save_modelspecs(modelspecs_dir, modelspec.fits())

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

logging.info('Generating summary statistics...')

# generate predictions
est, val = nems.analysis.api.generate_prediction(est, val, modelspec)

# evaluate prediction accuracy
modelspec = nems.analysis.api.standard_correlation(est, val, modelspec)
#slow with allen data
s = nems.metrics.api.state_mod_index(val, epoch='REFERENCE',
                                     psth_name='pred',
                                    state_sig='state', state_chan=[])

modelspec.meta['state_mod'] = s
modelspec.meta['state_chans'] = est['state'].chans

logging.info("Performance: r_fit={0:.3f} r_test={1:.3f}".format(
        modelspec.meta['r_fit'][0,0],
        modelspec.meta['r_test'][0,0]))

print(single_state_mod_index(val, modelspec, epoch='REFERENCE', state_chan="pupil"))

# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

logging.info('Generating summary plot...')

# set modelspec to plot psth for big vs. small pupil
modelspec[0]['plot_fn_idx']=5

# Generate a summary plot

modelspec.quickplot(rec=val,epoch='REFERENCE',include_input=False, time_range=(0,10))


