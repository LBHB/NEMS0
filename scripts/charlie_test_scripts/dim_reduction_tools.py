#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 11:56:40 2018

@author: hellerc
"""

# Functions to perform dimensionality reductions on response matrices

import numpy as np
import sys

def PCA_4D(r, trial_averaged=False, stim_averaged=False, center=True):
    
    # r must be in shape bincount x repcount x stimcount x cellcount
     
    bincount=r.shape[0]; repcount=r.shape[1]; stimcount=r.shape[2];cellcount=r.shape[3]; 
    r_pca = r.reshape(bincount*repcount*stimcount, cellcount);
    
    if trial_averaged is True and stim_averaged is True:
        sys.exit("Cannot average over both stimuli and trials")
    
    if trial_averaged is True:
        r_pca= np.mean(r_pca.reshape(bincount,repcount,stimcount,cellcount),1).reshape(bincount*stimcount,cellcount)
    if stim_averaged is True:
        r_pca = np.mean(r_pca.reshape(bincount,repcount,stimcount,cellcount),2).reshape(bincount*repcount,cellcount)
    
    if center is True:
        for i in range(0,cellcount):
            m = np.mean(r_pca[:,i])
            r_pca[:,i]=(r_pca[:,i]-m);
        
    U,S,V = np.linalg.svd(r_pca,full_matrices=False)
    v = S**2
    step = v;
    var_explained = []
    for i in range(0, cellcount):
        var_explained.append(100*(sum(v[0:(i+1)])/sum(v)));
    loading = V;
    pcs = U*S;
    
    out = {'pcs': pcs,
            'variance': var_explained,
            'step': step,
            'loading':loading
            }
    return out

def PCA(r, center=True):

    r_pca = r
    
    if center is True:
        for i in range(0,r_pca.shape[1]):
            m = np.mean(r_pca[:,i])
            r_pca[:,i]=(r_pca[:,i]-m);
        
    U,S,V = np.linalg.svd(r_pca,full_matrices=False)
    v = S**2
    step = v;
    var_explained = []
    for i in range(0, r_pca.shape[1]):
        var_explained.append(100*(sum(v[0:(i+1)])/sum(v)));
    loading = V;
    pcs = U #*S;
    
    out = {'pcs': pcs,
            'variance': var_explained,
            'step': step,
            'loading':loading
            }
    return out