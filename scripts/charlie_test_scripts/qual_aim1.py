#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:17:05 2018

@author: hellerc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:26:10 2018

@author: hellerc
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append('/auto/users/hellerc/code/NEMS/')
from nems.utilities.baphy import baphy_load_dataset, spike_time_to_raster
from nems.signal import Signal
import dim_reduction_tools as drt
import nems.db as nd
sys.path.append('/auto/users/hellerc/code/local_code/')
from plotting_array_geometery import plot_weights_64D
# load data in first (create a spike/response matrix)

#parmfilepath = '/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'
#options = {'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}

parmfilepath = '/auto/data/daq/Boleto/BOL005/BOL005c05_p_PPS_VOC.m'
options = {'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True, 'runclass': 'VOC'}


### Query database for the correct cellids
cids = list(nd.get_batch_cell_data(batch=294, cellid='BOL005c').index.get_level_values(0).unique())
cids_correct = cids.copy()
for i, cid in enumerate(cids):
    if nd.get_isolation(cellid=cid, batch=294).values < 75:
        cids_correct[i]=0
cids_correct=[x for x in cids_correct if x!=0]     

# load baphy parmfile
out = baphy_load_dataset(parmfilepath,
                           options)
# unpack output
event_times, spike_dict, stim_dict, state_dict = out
event_times['start_index']=[int(x) for x in event_times['start']*100]
event_times['end_index']=[int(x) for x in event_times['end']*100]

# r is response matrix, created from dictionary of spike times
spike_dict_new = spike_dict.copy()
for x in spike_dict.keys():
    if x not in cids_correct:
        del spike_dict_new[x]

r = spike_time_to_raster(spike_dict=spike_dict_new,
                         fs=100,
                         event_times=event_times)

# Load in pupil
p = state_dict['pupiltrace']
nan_inds =  np.argwhere(np.isnan(p.squeeze()))[0][0]
p[:,nan_inds:]=p[:,nan_inds-1] 


# create signals for ease of reshaping/collapsing over stimuli
p_signal = Signal(fs=100,
             matrix=p,
             name='test_name',
             recording='pupil',
             epochs=event_times)
p_folded = p_signal.extract_epochs(stim_dict.keys())

# Create the signal object spike raster (r)
sig = Signal(fs=100,
             matrix=r[0],
             name='test_name',
             recording='spikes',
             epochs=event_times)

# Fold matrix over all stimuli, returning a dictionary where keys are stimuli 
# each value in the dictionary is (reps X cell X bins)
folded_matrix = sig.extract_epochs(stim_dict.keys())

# Choose vocalization for analysis

resp = folded_matrix['STIM_ferretmixed42.wav']
pupil = p_folded['STIM_ferretmixed42.wav']
pred_sig = sig.replace_epochs({'STIM_ferretmixed42.wav': sig.average_epoch('STIM_ferretmixed42.wav')})
pred = pred_sig.extract_epoch('STIM_ferretmixed42.wav')


# make sure to remove all nans (extract_epochs is doing weird stuff at the edges...)
min_ind=resp.shape[-1]
for i in range(resp.shape[0]):
    if np.any(np.isnan(resp[i,0,:])):
        ti = np.argwhere(np.isnan(resp[i,0,:])).min()
        if ti < min_ind:
            min_ind = ti
resp = resp[:,:,:min_ind]
pupil = pupil[:,:,:min_ind]
pred = pred[:,:,:min_ind]        

# Downsample pupil and response (if rasterfs != 100)
rasterfs = 10
bins = int(round(resp.shape[-1]/(100/rasterfs)))
r = ss.resample(resp, bins,axis=2)
pup = ss.resample(pupil, bins,axis=2)
pred = ss.resample(pred, bins, axis=2)
bins = r.shape[2]
cells = r.shape[1]
reps = r.shape[0]

# debugging matrix shapes... everything is good at this point

# Compute principal components of response and look for pupil-correlated PCs
resp_pca = np.transpose(r, [0,2, 1]).reshape(bins*reps, cells)
pred_pca = np.transpose(pred, [0, 2, 1]).reshape(bins*reps, cells)
#resp_pca = resp_pca-pred_pca
p_pca = pup.reshape(bins*reps, 1)
are_nans=False
if np.any(np.isnan(p_pca)):
    are_nans = True
    n = sum(sum(np.isnan(p_pca)))
    resp_pca = resp_pca[:-n,:]
    p_pca = p_pca[:-n]
    pred_pca = pred_pca[:-n]
    
pca_out = drt.PCA(resp_pca.copy(),center=True)
pcs = pca_out['pcs']

# do pca with scikit too
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(resp_pca)
var_exp = []
for i in range(len(pca.explained_variance_ratio_)):
    var_exp.append(sum(pca.explained_variance_ratio_[0:i+1])/sum(pca.explained_variance_ratio_))

# Correlation with pupil
pup_corr = []
for i in range(0, cells):
    pup_corr.append(np.corrcoef(pcs[:,i],p_pca.squeeze())[0][1])

# Plot pca/pupil summary
#### NOTE ON LOADING MATRIX: first dimension of NON-transposed V matrix are loading vectors i.e. V[0,:] 
#### contains the pc1 weights for each neuron    
    
plt.figure()
plt.subplot(131)
plt.plot(pca_out['variance'], '.-')
plt.ylabel('Variance explained')
plt.xlabel('net PCs')
plt.subplot(132)
plt.imshow(pca_out['loading'][:5],aspect='auto')
plt.colorbar()
plt.ylabel('PCs')
plt.xlabel('Neurons')
plt.subplot(133)
plt.bar(np.arange(0,5),abs(np.array(pup_corr[:5])))
plt.xlabel('PCs')
plt.ylabel('r2 (pc vs. pupil)')

# ========= find neurons coupled to PCs that are coupled to pupil =========
plt.figure()
plt.subplot(121)
plt.plot(pup_corr[:10],alpha=0.6)
candidate_pcs = np.argsort([abs(x) for x in pup_corr[:10]])[::-1][:2]
plt.plot(candidate_pcs[0], pup_corr[candidate_pcs[0]], color='red', markersize=10,marker='.')
plt.plot(candidate_pcs[1], pup_corr[candidate_pcs[1]], color='green', markersize=10,marker='.')
# weight matrix
plt.subplot(122)
plt.plot(pca_out['loading'][candidate_pcs[0]],'.-r')
plt.plot(pca_out['loading'][candidate_pcs[1]], '.-g')
pc1_cells = np.argsort([abs(x) for x in pca_out['loading'][candidate_pcs[0]]])[::-1][:3]
pc2_cells = np.argsort([abs(x) for x in pca_out['loading'][candidate_pcs[1]]])[::-1][:3]

plt.figure()
plt.subplot(121)
plot_weights_64D(pca_out['loading'][candidate_pcs[0]],[[x] for x in cids_correct],
                 np.min(pca_out['loading']),np.max(pca_out['loading']))
plt.subplot(122)
plot_weights_64D(pca_out['loading'][candidate_pcs[1]],[[x] for x in cids_correct],
                 np.min(pca_out['loading']),np.max(pca_out['loading']))

# Look at correlations between intra- vs. inter pairs
inter1=[]
inter2=[]
intra=[]
for pc1_cell in pc1_cells:
    for pc2_cell in pc2_cells:
        n1 = resp_pca[:,pc1_cell]
        n2 = resp_pca[:,pc2_cell]
        intra.append(abs(np.corrcoef(n1,n2)[0][1]))
intra = [x for x in intra if x<=0.98]

for i, pc1_cell in enumerate(pc1_cells):
    for j in range(i+1,len(pc1_cells)):
        n1 = resp_pca[:,pc1_cell]
        n2 = resp_pca[:,pc1_cells[j]]
        n3 = resp_pca[:,pc2_cells[i]]
        n4 = resp_pca[:,pc2_cells[j]]
        inter1.append(abs(np.corrcoef(n1,n2)[0][1]))
        inter2.append(abs(np.corrcoef(n4,n3)[0][1]))
print('intra mean:')
print(np.mean(intra))
print('inside pc1 mean:')
print(np.mean(inter1))
print('inside pc2 mean:')
print(np.mean(inter2))


# ================ regress out pupil from these neurons ====================
pc1_1 = resp_pca[:,pc1_cells[0]]
pc1_1_pred = pred_pca[:,pc1_cells[0]]
pc2_1 = resp_pca[:,pc2_cells[0]]
pc2_1_pred = pred_pca[:,pc2_cells[0]]
pc2_2 = resp_pca[:,pc2_cells[1]]
pc2_2_pred = pred_pca[:,pc2_cells[1]]
p = p_pca

n1 = pc1_1#-pc1_1_pred
n2 = pc2_1#-pc2_1_pred
n3 = pc2_2#-pc2_2_pred

regr = linear_model.LinearRegression()
regr.fit(p, n1)
n1_pred = regr.predict(p) + pc1_1_pred
print("MSE's. w/pupil first, then just with psth")
print(mean_squared_error(pc1_1,n1_pred))
print(mean_squared_error(pc1_1, pc1_1_pred))

regr = linear_model.LinearRegression()
regr.fit(p, n2)
n2_pred = regr.predict(p) + pc2_1_pred
print(mean_squared_error(pc2_1,n2_pred))
print(mean_squared_error(pc2_1, pc2_1_pred))

regr = linear_model.LinearRegression()
regr.fit(p, n3)
n3_pred = regr.predict(p) + pc2_2_pred
print(mean_squared_error(pc2_2,n3_pred))
print(mean_squared_error(pc2_2, pc2_2_pred))


# === Prediction with pupil on a trial-by-trial basis ======
n1_error = []
n2_error = []
n3_error = []
reps = int(n1.shape[0]/bins)
n1_pred = n1_pred.reshape(reps, bins)
n2_pred = n2_pred.reshape(reps, bins)
n3_pred = n3_pred.reshape(reps, bins)

pc1_1 = pc1_1.reshape(reps,bins)
pc2_1 = pc2_1.reshape(reps,bins)
pc2_2 = pc2_2.reshape(reps,bins)

for i in range(0, reps):

    n1_error.append(mean_squared_error(pc1_1[i,:],n1_pred[i,:]))

    n2_error.append(mean_squared_error(pc2_1[i,:],n2_pred[i,:]))

    n3_error.append(mean_squared_error(pc2_2[i,:],n3_pred[i,:]))
    
plt.figure();
plt.subplot(121)
plt.scatter(n1_error,n2_error,color='red')
plt.scatter(n1_error,n3_error,color='blue')
plt.legend(['n1 vs. n2', 'n1 vs. n3'])
plt.xlabel('n1')
plt.ylabel('n2/n3')

plt.subplot(122)
plt.scatter(n2_error,n1_error,color='red')
plt.scatter(n2_error,n3_error,color='blue')
plt.legend(['n2 vs. n1', 'n2 vs. n3'])
plt.xlabel('n2')
plt.ylabel('n1/n3')

print("residual correlations")
print('n1 vs n2')
print(np.corrcoef(n1_error,n2_error)[0][1])
print('n1 vs n3')
print(np.corrcoef(n1_error,n3_error)[0][1])
print('n2 vs n3')
print(np.corrcoef(n2_error,n3_error)[0][1])


# overlay big and small pupil traces for the neurons
# big in black
# small in gray
median_pup = np.median(pup,2).squeeze()
ma = np.argsort(median_pup)[-1]
mi = np.argsort(median_pup)[0]
small_inds = np.argsort(median_pup)[int(reps/3):]
big_inds = np.argsort(median_pup)[:int(reps/3)]

# plot the psths for different conditions
n = n4
neuron='n4'
pc = 'pc1'
plt.figure();
plt.plot(np.mean(n.reshape(reps,bins),0),alpha=0.7,lw=3,color='orange')
plt.plot(np.mean(n.reshape(reps,bins)[small_inds,:],0), 'k',lw=3,alpha=0.7)
plt.plot(np.mean(n.reshape(reps,bins)[big_inds,:],0), 'gray',lw=3,alpha=0.7)
plt.title(neuron+ ' ' + pc)
plt.legend(['psth','small pupil','big pupil'])