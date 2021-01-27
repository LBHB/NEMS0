import OLP_analysis as olp

parmfile = '/auto/data/daq/Hood/HOD005/HOD005b09_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD006/HOD006b11_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD007/HOD007a10_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD008/HOD008d11_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD009/HOD009a09_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM013/ARM013b32_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM016/ARM016c15_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM018/ARM018a05_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM019/ARM019a07_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM022/ARM022b15_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM023/ARM023a11_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP'

pair, unit = 0, 0

params = olp.load_experiment_params(parmfile)
response = olp.get_response(params, sub_spont=True)
# [2,3] (hBG,fFG) - [2,7] (fBG, hFG) - [0,1,2] fBG/fFG - [1,4,3] (hBG,fFG)
olp.z_heatmaps_allpairs([1,4,3], response, params, 2)

olp.plot_combos(pair, unit, response, params, 2)
olp.psth_allpairs([1,4,3], unit, response, params, sigma=2, sum=False)
olp.z_allpairs([1,4,3], unit, response, params, sigma=2, z_av=False)

params = load_experiment_params(parmfile, 20)
olp.plot_projections(pair, parms)







pca_resp = np.nanmean(response[3,1,:,:,:], axis=0)

import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(pca_resp)
PCA(n_components=2)
print(pca.explained_variance_ratio_)
[0.9924... 0.0075...]
print(pca.singular_values_)
[6.30061... 0.54980...]


