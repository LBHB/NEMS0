from nems.xform_helper import fit_model_xforms

recording_uri = '/auto/users/jacob/nems/recordings/TAR010c-18-1.tar.gz'
modelname = 'ozgf100ch18_wcg18x1_fir1x15_lvl1_dexp1_fititer01'

# TODO: This kind of does the same thing as keywords, so maybe figure out
#       a way to use those instead? But module_sets are so variable it
#       seems clunky to need a diff kw for each one, esp. since the setup
#       of module_sets depends heavily on the other parts of the model.

# Could try to parse something like fititer-T4-T6-S0x1-S0x1x2x3
# to mean Tolerances: [1e-4, 1e-6]; Subsets: [[0,1], [0,1,2,3]],
# but at that point just passing the kwargs seems simpler.
fitter_kwargs = {
        'module_sets': [[0, 1], [0, 1, 2, 3]],
        'tolerances':  [1e-4, 1e-6, 1e-8],
        'invert': False,
        'max_iter': 50,
        }

fit_model_xforms(recording_uri, modelname, fitter_kwargs=fitter_kwargs)
