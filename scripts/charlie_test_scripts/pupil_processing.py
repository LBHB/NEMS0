#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:28:29 2018

@author: hellerc
"""

'''
Signal processing tools for analyzing state based on pupil
'''

from scipy.signal import ellip, freqz,lfilter
from scipy.ndimage.filters import gaussian_filter1d
import numpy as np
import matplotlib.pyplot as plt

def filt(p, rasterfs, low_cut=0.05, high_cut=0.05):
    
    f1 = high_cut/rasterfs*2;
    f2 = 14/rasterfs*2;
    b, a = ellip(N=4, rp=0.5, rs=20, Wn=np.array([f1, f2]), btype='bandpass');
    w, h = freqz(b)    
    hp = lfilter(b, a, p);
    print('applying highpass filter')
    
    f = low_cut/rasterfs*2
    b ,a = ellip(N=4, rp=0.5, rs=20, Wn=f);
    lp = lfilter(b, a, p);
    print('applying lowpass filter')
    
    return hp, lp
    
def derivative(p, rasterfs):
    # first smooth the pupil traces
    
    sigma=rasterfs;
    p = gaussian_filter1d(p, sigma)
    
    pderivative = np.ediff1d(p)
    d_pos = np.zeros(len(pderivative))
    d_neg = np.zeros(len(pderivative))
    
    d_pos[np.argwhere(pderivative>0)] = pderivative[np.argwhere(pderivative>0)]
    d_neg[np.argwhere(pderivative<0)] = pderivative[np.argwhere(pderivative<0)]

    return d_pos, d_neg