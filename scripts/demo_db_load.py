#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import nems.plots.api as nplt
import nems.db as nd
import nems.xforms as xforms
from nems.gui.recording_browser import browse_recording, browse_context

cellid='TAR010c-18-1'
batch=271
#modelname = 'wc.18x1.g-fir.1x15-lvl.1'
modelname = 'dlog-wc.18x1.g-fir.1x15-lvl.1'
#modelname = 'dlog-wc.18x1.g-stp.1-fir.1x15-lvl.1-dexp.1'

d=nd.get_results_file(batch=batch, cellids=[cellid], modelnames=[modelname])

filepath = d['modelpath'][0] + '/'
xfspec, ctx = xforms.load_analysis(filepath, eval_model=False)

ctx, log_xf = xforms.evaluate(xfspec, ctx)

#nplt.quickplot(ctx)
ctx['modelspec'].quickplot(ctx['val'])

aw = browse_context(ctx, signals=['stim', 'pred', 'resp'])

