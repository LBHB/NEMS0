from nems.xform_helper import fit_model_xforms

recording_uri = '/auto/users/jacob/nems/recordings/TAR010c-18-1.tar.gz'
modelname = 'ozgf100ch18_wcg18x1_fir1x15_lvl1_dexp1_'
fitter = 'fititer01-T4-T6-S0x1-S0x1x2x3-ti20-fi20'
modelname += fitter

# Above keyword does the same thing as:
# (see _parse_fititer in xform_helper)
#fitter_kwargs = {
#        'module_sets': [[0, 1], [0, 1, 2, 3]],
#        'tolerances': [1e-4, 1e-6],
#        'tol_iter': 20,
#        'fit_iter': 20,
#        'fitter': coordinate_descent,
#        }

fit_model_xforms(recording_uri, modelname)
