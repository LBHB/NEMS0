import OLP_analysis as olp

parmfile = '/auto/data/daq/Hood/HOD005/HOD005b09_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD006/HOD006b11_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD007/HOD007a10_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD008/HOD008d11_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD009/HOD009a09_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM013/ARM013b32_p_OLP'   #0
parmfile = '/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP'   #1
parmfile = '/auto/data/daq/Armillaria/ARM016/ARM016c15_p_OLP'   #2
parmfile = '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP'   #3
parmfile = '/auto/data/daq/Armillaria/ARM018/ARM018a05_p_OLP'   #4
parmfile = '/auto/data/daq/Armillaria/ARM019/ARM019a07_p_OLP'   #5
parmfile = '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP'   #6
parmfile = '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP'   #7
parmfile = '/auto/data/daq/Armillaria/ARM022/ARM022b15_p_OLP'   #8
parmfile = '/auto/data/daq/Armillaria/ARM023/ARM023a11_p_OLP'   #9
parmfile = '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP'   #10
parmfile = '/auto/data/daq/Armillaria/ARM025/ARM025a10_p_OLP'   #11

pair, unit = 1, 21

params = olp.load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
response = olp.get_response(params, sub_spont=False)
corcoefs = olp._base_reliability(response, params, rep_dim=2, protect_dim=3)
avg_resp = olp._significant_resp(response, params, protect_dim=3, time_dim=-1)
response = olp.find_good_units(response, params,
                               corcoefs=corcoefs, corcoefs_threshold=0.1,
                               avg_resp=avg_resp, avg_threshold=0.2)


import numpy as np
pair = 1

bg_resp, fg_resp = response[pair, 0, :, :, :], response[pair, 1, :, :, :]
combo_resp = response[pair, 2, :, :, :]

bg_len, fg_len = np.count_nonzero(~np.isnan(bg_resp[:, 0, 0])), \
                 np.count_nonzero(~np.isnan(fg_resp[:, 0, 0]))
min_rep = np.min((bg_len, fg_len))
lin_resp = (bg_resp[:min_rep, :, :] + fg_resp[:min_rep, :, :])

mean_bg, mean_fg = np.nanmean(bg_resp, axis=(0,2)), np.nanmean(fg_resp, axis=(0,2))
mean_lin, mean_combo = np.nanmean(lin_resp, axis=(0,2)), np.nanmean(combo_resp, axis=(0,2))

supp = mean_lin - mean_combo

bg_fg = np.concatenate((np.expand_dims(mean_bg, axis=1),
                        np.expand_dims(mean_fg, axis=1)), axis=1)
supp_supp = np.concatenate((np.expand_dims(supp, axis=1),
                        np.expand_dims(supp, axis=1)), axis=1)

fig, ax = plt.subplots()
for pnt in range(supp.shape[0]):
    ax.plot(bg_fg[pnt,:], supp_supp[pnt, :], marker='', color='black', linestyle=':',
            zorder=-1, label=params['good_units'][pnt])
ax.scatter(bg_fg[:,0], supp_supp[:,0], marker='o', color='deepskyblue')
ax.scatter(bg_fg[:,1], supp_supp[:,1], marker='o', color='yellowgreen')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()

fig, ax = plt.subplots()
for pnt in range(supp.shape[0]):
    ax.plot(bg_fg[pnt,:], supp_supp[pnt, :], marker='o', linestyle=':',
            zorder=-1, label=params['good_units'][pnt])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout()

fig, ax = plt.subplots(1, 2, sharey=True)
for pnt in range(supp.shape[0]):
    ax[0].plot(mean_bg[pnt], supp[pnt], marker='o', linestyle='None',
               label=params['good_units'][pnt])
    ax[1].plot(mean_fg[pnt], supp[pnt], marker='o', linestyle='None',
               label=params['good_units'][pnt])
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax[0].set_ylabel('Suppression\n(BG + FG) - BG/FG combo')
ax[0].set_xlabel('BG Response'), ax[1].set_xlabel('FG Response')
ax[0].set_box_aspect(1), ax[1].set_box_aspect(1)
fig.suptitle(f"Experiment {params['experiment']} - Pair {pair}\n"
             f"Background {params['pairs'][pair][0]} - Foreground {params['pairs'][pair][1]}",
             fontweight='bold')
fig.tight_layout()




respA, respB = response[pair, resp_idx[0], :, unit, :], \
               response[pair, resp_idx[1], :, unit, :]
lenA, lenB = np.count_nonzero(~np.isnan(respA[:, 0])), np.count_nonzero(~np.isnan(respB[:, 0]))
min_rep = np.min((lenA, lenB))
resplin = (respA[:min_rep, :] + respB[:min_rep, :])
mean_resplin = resplin.mean(axis=0)
if sigma:
    ax.plot(x, sf.gaussian_filter1d(mean_resplin * params['fs'], sigma), color='dimgray',
            ls='--', label='Linear Sum')
if not sigma:
    ax.plot(x, mean_resplin * params['fs'], color='dimgray', ls='--', label='Linear Sum')



# [2,3] (hBG,fFG) - [2,7] (fBG, hFG) - [0,1,2] fBG/fFG - [1,4,3] (hBG,fFG)
olp.z_heatmaps_allpairs([4,1,3], response, params, 2)
olp.z_heatmaps_allpairs([0,1,2], response, params, 2)
olp.z_heatmaps_allpairs([0,5,7], response, params, 2)



olp.plot_combos(pair, unit, response, params, 2)
olp.psth_allpairs([0,1,2], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([0,1,2], unit, response, params, sigma=2, z_av=False)

olp.psth_allpairs([4,1,3], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([4,1,3], unit, response, params, sigma=2, z_av=False)

olp.psth_allpairs([0,5,7], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([0,5,7], unit, response, params, sigma=2, z_av=False)

olp.psth_allunits([1,4,3], pair, response, params, sigma=2, sum=False)


olp.plot_projections(pair, parms)


olp.psth_fulls_allunits(pair, response, params, 2)

z, zp, z2, zp2 = olp.z_bgfg_compare([0,2], [1,2], 18, response, params, 2)
olp.z_bgfg_compare2(z,zp,z2,zp2,unit,response,params,2)

import matplotlib.pyplot as plt
import numpy as np

pca_resp = np.nanmean(response[3,1,:,:,:], axis=0)

olp.plot_auc_mean([0,1,2], response, params)
olp.plot_auc_mean([0,5,7], response, params)
olp.plot_auc_mean([4,1,3], response, params)



colors = ['deepskyblue', 'yellowgreen', 'dimgray']
pre_bin = int(params['PreStimSilence'] * params['fs'])
post_bin = int((params['Duration'] - params['PostStimSilence']) * params['fs'])
for cnt, cmb in enumerate(combo):
    resp_sub = np.nanmean(response[:, cmb, :, :, int(pre_bin):int(post_bin)], axis=0)
    mean_resp = np.nanmean(resp_sub, axis=0)
    x = np.linspace(0, resp_sub.shape[-1] / params['fs'], resp_sub.shape[-1]) \
        - params['PreStimSilence']

    auc = np.sum(mean_resp, axis=1)
    center = np.sum(mean_resp * x, axis=1) / np.sum(mean_resp, axis=1)
    plt.plot(auc, center, marker='o', linestyle='None', color=colors[cnt],
             label=params['combos'][cmb])
    plt.xlabel('Area Under Curve'), plt.ylabel('Center')
    plt.title(f"Experiment {params['experiment']} - Combos {combo}")
    plt.legend()


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

import matplotlib.pyplot as plt
import numpy as np

##For when I'm doing batch stuff.
parmfiles = ['/auto/data/daq/Armillaria/ARM013/ARM013b32_p_OLP',
             '/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP',
             '/auto/data/daq/Armillaria/ARM016/ARM016c15_p_OLP',
             '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM018/ARM018a05_p_OLP',
             '/auto/data/daq/Armillaria/ARM019/ARM019a07_p_OLP',
             '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP',
             '/auto/data/daq/Armillaria/ARM021/ARM021b14_p_OLP',
             '/auto/data/daq/Armillaria/ARM022/ARM022b15_p_OLP',
             '/auto/data/daq/Armillaria/ARM023/ARM023a11_p_OLP',
             '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM025/ARM025a10_p_OLP']

parmfiles = ['/auto/data/daq/Armillaria/ARM015/ARM015b15_p_OLP',
             '/auto/data/daq/Armillaria/ARM017/ARM017a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM019/ARM019a07_p_OLP',
             '/auto/data/daq/Armillaria/ARM020/ARM020a05_p_OLP',
             '/auto/data/daq/Armillaria/ARM024/ARM024a10_p_OLP']
responses, parameters = olp.load_parms(parmfiles)

olp.multi_exp_auccenter(0, responses, parameters)
olp.multi_exp_auccenter(1, responses, parameters)
olp.multi_exp_auccenter(2, responses, parameters)
olp.multi_exp_auccenter(3, responses, parameters)
olp.multi_exp_auccenter(7, responses, parameters)

import matplotlib.pyplot as plt
import numpy as np

combos = [0,1,2,3,4,5,6,7]

threshold = 3
markers = ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd', '*']
all_sites_arrays = list()
all_units_ids = list()
colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray', 'olive']
all_aucs, all_centers = {}, {}
all_sites_centers = list()

fig, ax = plt.subplots()

for cnt, exp in enumerate(responses.keys()):
    auc_array = np.empty([5, responses[exp].shape[-2]])
    center_array = np.empty([5, responses[exp].shape[-2]])
    for cmbnum, cmb in enumerate(combos):
        pre_bin = int(parameters[exp]['PreStimSilence'] * parameters[exp]['fs'])
        post_bin = int(responses[exp].shape[-1] - (parameters[exp]['PostStimSilence'] * parameters[exp]['fs']))
        resp_sub = np.nanmean(responses[exp][:, cmb, :, :, :], axis=0)
        mean_resp = np.nanmean(resp_sub[..., int(pre_bin):int(post_bin)], axis=0)

        x = np.linspace(0, mean_resp.shape[-1] / parameters[exp]['fs'], mean_resp.shape[-1])
        center = np.sum(np.abs(mean_resp) * x, axis=1) / np.sum(np.abs(mean_resp), axis=1)

        auc = np.sum(mean_resp, axis=1)
        auc_array[cmbnum,:] = auc
        center_array[cmbnum,:] = center

    resp_mask = np.abs(auc_array) > threshold
    resp_mask = np.any(resp_mask, axis=0)
    auc_array = auc_array[:, resp_mask]
    center_array = center_array[:, resp_mask]

    all_units_ids.append(np.asarray(parameters[exp]['good_units'])[resp_mask])
    all_sites_arrays.append(auc_array)
    all_sites_centers.append(center_array)

    all_aucs[exp] = auc_array
    all_centers[exp] = center_array

    ax.plot(auc_array, marker=markers[cnt], linestyle='--', color=colors[cnt])
    ax.scatter([0],auc_array[0,0], color=colors[cnt], label=exp)

all_sites_arrays = np.concatenate(all_sites_arrays, axis=1)
all_units_ids = np.concatenate(all_units_ids, axis=0)
ax.set_ylabel('Area Under Curve')
ax.set_xticks([0,1.0,2.0,3.0,4.0])
ax.set_xticklabels(['Full BG','Full FG','Full BG/Full FG','Half BG/Full FG','Full BG/Half FG'])
ax.set_title(f'Threshold |AUC| > {threshold}')
_ = ax.legend()

plot_combos = [3,1,3]
label_combos = [combos[aa] for aa in plot_combos]
site = 'ARM020'
colors = ['deepskyblue', 'yellowgreen', 'dimgray']

fig, ax = plt.subplots()
for ct, cb in enumerate(plot_combos):
    ax.plot(all_aucs[site][ct], all_centers[site][ct], linestyle='None', marker='o',
            label=parameters[site]['combos'][label_combos[ct]], color=colors[ct])
    ax.legend()
ax.set_xlabel('Area Under Curve'), ax.set_ylabel('Center of Mass')
plt.title(f"Experiment {parameters[site]['experiment']} - Combos {label_combos}")


combo = 4
lab_comb = combos[combo]
fig, ax = plt.subplots()
for ct, site in enumerate(all_aucs.keys()):
    ax.plot(all_aucs[site][ct], all_centers[site][ct], linestyle='None', marker=markers[ct],
            label=site)
    ax.legend()
ax.set_title(f"Combo {lab_comb} {parameters[site]['combos'][lab_comb]}")

# all_sites_arrays = np.concatenate(all_sites_arrays, axis=1)
# plt.plot(all_sites_arrays, marker=markers[cnt], linestyle='--', label=exp)


