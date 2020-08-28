#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 20:58:16 2020

@author: luke
"""

#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
rec_file_dir='/Users/grego/Downloads/'
# cellid='fre196b-21-1_1417-1233'; rf='fre196b_ec1d319ae74a2d790f3cbda73d46937e588bc791.tgz'
cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
batch=306
rec_file = rec_file_dir + rf


import os
import numpy as np
import matplotlib.pyplot as plt

import nems.recording as recording
import nems.plots.api as nplt
import nems.epoch as ep
import nems
rec=recording.load_recording(rec_file)
rec['resp'] = rec['resp'].extract_channels([cellid])
rec['resp'].fs=200

resp = rec['resp'].rasterize()

stimname = 'STIM_T+si464+null' #HCT A playing env 1
stimname = 'STIM_T+null+si464' #HCT B playing env 1
stimname = 'STIM_T+si464+si464' #HCT A+B both playing env 1
epoch_regex=stimname.replace('+','\+')

stim_epochs = ep.epoch_names_matching(resp.epochs, epoch_regex)
#resp_matrix=resp.extract_epoch('STIM_T+si464+null') * resp.fs

r = resp.as_matrix(stim_epochs) * resp.fs
#s = stim.as_matrix(stim_epochs)
repcount = np.sum(np.isfinite(r[:, :, 0, 0]), axis=1)
max_rep_id, = np.where(repcount == np.max(repcount))

t = np.arange(r.shape[-1]) / resp.fs

plt.figure()

#ax = plt.subplot(3, 1, 1)
#nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
  #                    title="cell {} - stim".format(cellid))

ax = plt.subplot(2, 1, 1)
nplt.raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster')
ax.set_title(stimname)
ax = plt.subplot(2, 1, 2);
nplt.psth_from_raster(t,r[max_rep_id[-1],:,0,:], ax=ax, title='raster',
                      ylabel='spk/s', binsize=10)
ax.set_title('')
plt.tight_layout()


##Make rasters
#Decide which cell:
#UPDATE THE LINE BELOW TO POINT TO THE FILE
rec_file_dir='/Users/grego/Downloads/'
cellid='fre197c-105-1_705-1024'; rf='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz' #Neuron 1 on poster
cellid2='fre197c-68-1_705-1024'; rf2='fre197c_f94fb643b4cb6380b8eb3286fc30d908a1940ea2.tgz'
batch=306
rec_file = rec_file_dir + rf
rec_file2 = rec_file_dir + rf2

import os
import numpy as np
import matplotlib.pyplot as plt

import nems.recording as recording
import nems.plots.api as nplt
import nems.epoch as ep
import nems
rec=recording.load_recording(rec_file)
rec2=recording.load_recording(rec_file2)
rec['resp'] = rec['resp'].extract_channels([cellid])
rec['resp'].fs=200
rec2['resp'] = rec2['resp'].extract_channels([cellid2])
rec2['resp'].fs=200

resp = rec['resp'].rasterize()
resp2 = rec2['resp'].rasterize()

stimnameA = 'STIM_T+si464+null' #HCT A playing env 1
stimnameB = 'STIM_T+null+si464' #HCT B playing env 1
stimnameC = 'STIM_T+si464+si464' #HCT A+B both playing env 1
epoch_regexA=stimnameA.replace('+','\+')
epoch_regexB=stimnameB.replace('+','\+')
epoch_regexC=stimnameC.replace('+','\+')

stim_epochsA1 = ep.epoch_names_matching(resp.epochs, epoch_regexA)
stim_epochsA2 = ep.epoch_names_matching(resp2.epochs, epoch_regexA)
stim_epochsB1 = ep.epoch_names_matching(resp.epochs, epoch_regexB)
stim_epochsB2 = ep.epoch_names_matching(resp2.epochs, epoch_regexB)
stim_epochsC1 = ep.epoch_names_matching(resp.epochs, epoch_regexC)
stim_epochsC2 = ep.epoch_names_matching(resp2.epochs, epoch_regexC)
#resp_matrix=resp.extract_epoch('STIM_T+si464+null') * resp.fs

rA1 = resp.as_matrix(stim_epochsA1) * resp.fs
rA2 = resp2.as_matrix(stim_epochsA2) * resp2.fs
rB1 = resp.as_matrix(stim_epochsB1) * resp.fs
rB2 = resp2.as_matrix(stim_epochsB2) * resp2.fs
rC1 = resp.as_matrix(stim_epochsC1) * resp.fs
rC2 = resp2.as_matrix(stim_epochsC2) * resp2.fs
# s = stim.as_matrix(stim_epochs)
repcountA1 = np.sum(np.isfinite(rA1[:, :, 0, 0]), axis=1)
repcountA2 = np.sum(np.isfinite(rA2[:, :, 0, 0]), axis=1)
repcountB1 = np.sum(np.isfinite(rB1[:, :, 0, 0]), axis=1)
repcountB2 = np.sum(np.isfinite(rB2[:, :, 0, 0]), axis=1)
repcountC1 = np.sum(np.isfinite(rC1[:, :, 0, 0]), axis=1)
repcountC2 = np.sum(np.isfinite(rC2[:, :, 0, 0]), axis=1)


max_rep_idA1, = np.where(repcountA1 == np.max(repcountA1))
max_rep_idA2, = np.where(repcountA2 == np.max(repcountA2))
max_rep_idB1, = np.where(repcountB1 == np.max(repcountB1))
max_rep_idB2, = np.where(repcountB2 == np.max(repcountB2))
max_rep_idC1, = np.where(repcountC1 == np.max(repcountC1))
max_rep_idC2, = np.where(repcountC2 == np.max(repcountC2))

tA1 = np.arange(rA1.shape[-1]) / resp.fs
tA2 = np.arange(rA2.shape[-1]) / resp2.fs
tB1 = np.arange(rB1.shape[-1]) / resp.fs
tB2 = np.arange(rB2.shape[-1]) / resp2.fs
tC1 = np.arange(rC1.shape[-1]) / resp.fs
tC2 = np.arange(rC2.shape[-1]) / resp2.fs

#Unit1
plt.figure()
# ax = plt.subplot(3, 1, 1)
# nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
#                      title="cell {} - stim".format(cellid))

ax = plt.subplot(4, 1, 1)
nplt.raster(tA1,rA1[max_rep_idA1[-1],:,0,:], ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('HCT A ', rotation=0,ha='right')
ax.set_title('Unit 1',fontsize=15)
ax = plt.subplot(4, 1, 2)
nplt.raster(tB1,rB1[max_rep_idB1[-1],:,0,:], ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('HCT B ', rotation=0,ha='right')
ax = plt.subplot(4, 1, 3)
nplt.raster(tC1,rC1[max_rep_idC1[-1],:,0,:], ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('HCT A+B ', rotation=0,ha='right')
# ax.set_title(stimname)
ax = plt.subplot(4, 1, 4);
nplt.psth_from_raster(tA1,rA1[max_rep_idA1[-1],:,0,:], ax=ax,
                      ylabel='spk/sec', binsize=10,facecolor='lightblue')
nplt.psth_from_raster(tB1,rB1[max_rep_idB1[-1],:,0,:], ax=ax,
                      ylabel='spk/sec', binsize=10,facecolor='orangered')
nplt.psth_from_raster(tC1,rC1[max_rep_idC1[-1],:,0,:], ax=ax,
                      ylabel='spk/sec', binsize=10,facecolor='lightgreen')
ax.set_xlabel('Time (s)')
ax.set_xticklabels([-1.0,-0.5,0,0.5,1.0,1.5,2.0,2.5,3.0])
plt.subplots_adjust(hspace=.01)
# ax.set_title('')
plt.tight_layout()


#Unit2
plt.figure()
# ax = plt.subplot(3, 1, 1)
# nplt.plot_spectrogram(s[max_rep_id[-1],0,:,:], fs=stim.fs, ax=ax,
#                      title="cell {} - stim".format(cellid))

ax = plt.subplot(4, 1, 1)
nplt.raster(tA2,rA2[max_rep_idA2[-1],:,0,:], ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('HCT A ', rotation=0,ha='right')
ax.set_title('Unit 2',fontsize=15)
ax = plt.subplot(4, 1, 2)
nplt.raster(tB2,rB2[max_rep_idB2[-1],:,0,:], ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('HCT B ', rotation=0,ha='right')
ax = plt.subplot(4, 1, 3)
nplt.raster(tC2,rC2[max_rep_idC2[-1],:,0,:], ax=ax)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('')
ax.set_ylabel('HCT A+B ', rotation=0,ha='right')
# ax.set_title(stimname)
ax = plt.subplot(4, 1, 4);
nplt.psth_from_raster(tA2,rA2[max_rep_idA2[-1],:,0,:], ax=ax,
                      ylabel='spk/sec', binsize=10,facecolor='lightblue')
nplt.psth_from_raster(tB2,rB2[max_rep_idB2[-1],:,0,:], ax=ax,
                      ylabel='spk/sec', binsize=10,facecolor='orangered')
nplt.psth_from_raster(tC2,rC2[max_rep_idC2[-1],:,0,:], ax=ax,
                      ylabel='spk/sec', binsize=10,facecolor='lightgreen')
ax.set_xlabel('Time (s)')
ax.set_xticklabels([-1.0,-0.5,0,0.5,1.0,1.5,2.0,2.5,3.0])
plt.subplots_adjust(hspace=.01)
# ax.set_title('')
plt.tight_layout()