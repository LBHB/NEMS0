#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import nems.plots.api as nplt
import nems.db as nd
import nems.xforms as xforms

cellid='TAR010c-18-1'
batch=271
d=nd.get_results_file(batch=batch, cellids=[cellid])

filepath = d['modelpath'][0] + '/'
xfspec, ctx = xforms.load_analysis(filepath, eval_model=False)

ctx, log_xf = xforms.evaluate(xfspec, ctx)

nplt.quickplot(ctx)