import numpy as np
import os
import io
import logging
import matplotlib

#matplotlib.use('Qt5Agg')

from nems.xform_helper import fit_model_xform, load_model_xform
from nems import recording, get_setting
import nems.gui.editors as gui

log = logging.getLogger(__name__)

batch = 271
cellid = "TAR010c-18-1"
recording_file = "TAR010c_6ae9286b2aafa709114966dadc35082eeb2abb73.tgz"

# MODEL SPEC
modelname = 'ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic'

#cellid = "TAR010c-09-2"
#recording_file = "TAR010c.NAT.fs50.tgz"
# MODEL SPEC
#modelname = 'ld-st.pup-tev_dlog-wc.18x1.g-fir.1x15-lvl.1-stategain.2-dexp.1_init.st-basic'

recording.get_demo_recordings(name=recording_file)
recording_uri = os.path.join(get_setting('NEMS_RECORDINGS_DIR'),
                             recording_file)

# run and return
xfspec, ctx = fit_model_xform(cellid, batch, modelname, recording_uri=recording_uri, returnModel=True)

#ex = gui.browse_xform_fit(ctx, xfspec)

# run and save to database
#saveuri = fit_model_xform(cellid, batch, modelname, recording_uri=recording_uri, saveInDB=True)



# load previously saved
#xfspec, ctx = load_model_xform(cellid, batch, modelname)
#ex = gui.browse_xform_fit(ctx, xfspec)

