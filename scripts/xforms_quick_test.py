import numpy as np
import os
import io
import logging
import matplotlib

matplotlib.use('Qt5Agg')

import nems.modelspec as ms
import nems.xforms as xforms
from nems.xform_helper import fit_model_xform, load_model_xform
from nems.utils import escaped_split, escaped_join
import nems.db as nd
from nems import get_setting
from nems.xform_helper import _xform_exists
from nems.registry import KeywordRegistry, xforms_lib, keyword_lib
from nems.plugins import (default_keywords, default_loaders,
                          default_initializers, default_fitters)
from nems.gui.recording_browser import browse_recording, browse_context
import nems.gui.editors as gui

log = logging.getLogger(__name__)


batch = 271
cellid = "TAR010c-18-1"
recording_uri = '/Users/svd/python/nems_test/recordings/TAR010c_6ae9286b2aafa709114966dadc35082eeb2abb73.tgz'
# MODEL SPEC
modelname = 'ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'

# run and return
xfspec, ctx = fit_model_xform(cellid, batch, modelname, recording_uri=recording_uri, returnModel=True)
ex = gui.browse_xform_fit(ctx, xfspec)

# run and save to database
#saveuri = fit_model_xform(cellid, batch, modelname, recording_uri=recording_uri, saveInDB=True)



# load previously saved
#xfspec, ctx = load_model_xform(cellid, batch, modelname)
#ex = gui.browse_xform_fit(ctx, xfspec)

