#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 14:02:26 2018

@author: svd
"""
import matplotlib.pyplot as plt


def ax_remove_box(ax=None, right=True, top=True, left=False, bottom=False, ticks=False):
    """
    remove right and top lines from plot border
    """
    if ax is None:
        ax = plt.gca()

    for remove, loc in [(right, 'right'), (left, 'left'), (top, 'top'), (bottom, 'bottom')]:
        if remove:
            ax.spines[loc].set_visible(False)

    if ticks:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])
