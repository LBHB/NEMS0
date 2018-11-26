#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:02:26 2018

@author: svd
"""
import matplotlib.pyplot as plt

def ax_remove_box(ax=None):
    """
    remove right and top lines from plot border
    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

