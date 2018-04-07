from nems.xform_helper import fit_model_xforms_baphy

recording_uri = '/auto/users/jacob/nems/recordings/TAR010c-18-1.tar.gz'
modelname = 'ozgf100ch18_wcg18x1_fir1x15_lvl1_dexp1_fititer01'

fit_model_xforms_baphy(recording_uri, modelname)
