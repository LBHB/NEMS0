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
parmfile = '/auto/data/daq/Armillaria/ARM026/ARM026b07_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM027/ARM027a15_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM028/ARM028b13_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM029/ARM029a14_p_OLP'
parmfile = '/auto/data/daq/Armillaria/ARM030/ARM030a12_p_OLP'

params = olp.load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
response = olp.get_response(params, sub_spont=False)
corcoefs = olp._base_reliability(response, rep_dim=2, protect_dim=3)
avg_resp = olp._significant_resp(response, params, protect_dim=3, time_dim=-1)
response = olp._find_good_units(response, params,
                               corcoefs=corcoefs, corcoefs_threshold=0.1,
                               avg_resp=avg_resp, avg_threshold=0.2)

unit = 1
pair = 1

olp.plot_rasters([0,1,2], 1, unit, response, params, 2)



import matplotlib.pyplot as plt
import numpy as np
supp_array = np.empty([len(params['good_units']), len(params['pairs'])])
for nn, pp in enumerate(params['pairs']):
    _, _, _, _, supp, _, _ = olp.get_scatter_resps(nn, response)
    supp_array[:,nn] = supp

fig, ax = plt.subplots()
ax.plot(supp_array.T, marker='o', linestyle=':')
ax.set_xticks([*range(len(params['pairs']))])
ax.set_xticklabels(params['pairs'], ha='right', rotation=40)
ax.set_ylabel('Suppression\n(BG + FG) - BG/FG combo')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
fig.suptitle(f"Experiment {params['experiment']}", fontweight='bold')
fig.tight_layout()


olp.psth_comp([0,1,2], pair, unit, response, params, sigma=2, z=True, sum=True)
pair = 1
#Scatter plots
#bgfg v suppression
olp.bgfg_suppression_scatter(pair, response, params)
#bg v fg
olp.bg_fg_scatter(pair, response, params)
#lin v combo
olp.lin_combo_scatter(pair, response, params)
#bg v fg and lin v combo
olp.bgfg_lincombo_scatter(pair, response, params)


#Big overview of data, plots all full BG and FG responses for all units
olp.psth_fulls_allunits(pair, response, params, 2)


#Plots a bunch of heat maps to get overview of data
# [2,3] (hBG,fFG) - [2,7] (fBG, hFG) - [0,1,2] fBG/fFG - [1,4,3] (hBG,fFG)
olp.z_heatmaps_allpairs([4,1,3], response, params, 2)
olp.z_heatmaps_allpairs([0,1,2], response, params, 2, arranged=True)
olp.z_heatmaps_allpairs([0,1,2], response, params, 2, arranged=False)
olp.z_heatmaps_allpairs([0,5,7], response, params, 2)


pair, unit = 1, 1
olp.plot_combos(pair, unit, response, params, 2)
olp.psth_allpairs([0,1,2], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([0,1,2], unit, response, params, sigma=2, z_av=False)

olp.psth_allpairs([4,1,3], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([4,1,3], unit, response, params, sigma=2, z_av=False)

olp.psth_allpairs([0,5,7], unit, response, params, sigma=2, sum=True)
olp.z_allpairs([0,5,7], unit, response, params, sigma=2, z_av=False)

olp.psth_allunits([0,1,2], pair, response, params, sigma=2, sum=False)




z, zp, z2, zp2 = olp.z_bgfg_compare([0,2], [1,2], 2, response, params, 2)
olp.z_bgfg_compare2(z,zp,z2,zp2,unit,response,params,2)

import matplotlib.pyplot as plt
import numpy as np

pca_resp = np.nanmean(response[3,1,:,:,:], axis=0)

olp.plot_auc_mean([0,1,2], response, params)
olp.plot_auc_mean([0,5,7], response, params)
olp.plot_auc_mean([4,1,3], response, params)



#####################################
#This one does the same as bgfg_suppression_scatter but in subplots rather than on one
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
#####################################



colors = ['deepskyblue', 'yellowgreen', 'dimgray']
pre_bin = int(params['PreStimSilence'] * params['fs'])
post_bin = int((params['stim length'] - params['PostStimSilence']) * params['fs'])
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


#function to see what pairs used

pair_dict = {}
for cnt, parmfile in enumerate(parmfiles):
    params = olp.load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
    bgs = np.expand_dims(np.asarray(params['Background']), axis=1)
    fgs = np.expand_dims(np.asarray(params['Foreground']), axis=1)
    pairs = np.concatenate((bgs, fgs), axis=1)
    pair_dict[f"{params['experiment']}"] = pairs

fig, ax = plt.subplots()
# for pp in range(pair_dict['ARM020'].shape[0]):
for cnt, site in enumerate(pair_dict.keys()):
    ax.plot(pair_dict[site][:,0], pair_dict[site][:,1], marker='o', linestyle='None',
        label=site)


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
             '/auto/data/daq/Armillaria/ARM025/ARM025a10_p_OLP',
             '/auto/data/daq/Armillaria/ARM026/ARM026b07_p_OLP',
             '/auto/data/daq/Armillaria/ARM027/ARM027a15_p_OLP',
             '/auto/data/daq/Armillaria/ARM028/ARM028b13_p_OLP',
             '/auto/data/daq/Armillaria/ARM029/ARM029a14_p_OLP',
             '/auto/data/daq/Armillaria/ARM030/ARM030a12_p_OLP']

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



def z_heatmaps_onepairs(resp_idx, pair, response, params, sigma=None):
    """Plots a two column figure of subplots, one for each sound pair, displaying a heat map
    of the zscore for all the units."""
    zscore, z_params = olp.get_z(resp_idx, response, params)

    if sigma is not None:
        zscore = sf.gaussian_filter1d(zscore, sigma, axis=2)
        zmin, zmax = np.min(np.min(zscore, axis=2)), np.max(np.max(zscore, axis=2))
        abs_max = max(abs(zmin),zmax)
    else:
        zmin, zmax = np.min(np.min(zscore, axis=1)), np.max(np.max(zscore, axis=1))
        abs_max = max(abs(zmin),zmax)

    fig, ax = plt.subplots()

    im = ax.imshow(zscore[pair, :, :], aspect='auto', cmap='bwr',
              extent=[-0.5, (zscore[pair, :, :].shape[1] / params['fs']) -
                      0.5, zscore[pair, :, :].shape[0], 0], vmin=-abs_max, vmax=abs_max)
    ax.set_title(f"Pair {pair}: BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                 fontweight='bold')
    ymin, ymax = ax.get_ylim()
    ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles='--', lw=1)
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin + 0.3, xmax - 0.2)
    ax.set_xticks([0, 0.5, 1.0])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    # fig.text(0.5, 0.03, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.05, 0.5, 'Neurons', ha='center', va='center', rotation='vertical', fontweight='bold')

    # fig.suptitle(f"Experiment {params['experiment']} - Combo Index {z_params['resp_idx']} - "
    #              f"{z_params['idx_names']} - Sigma {sigma}\n"
    #              f"{z_params['label']}", fontweight='bold')



#plot electrode shank
from nems_lbhb.plots import plot_weights_64D
plot_weights_64D(np.zeros(64),[f'AMT001a-{x}-1' for x in range(64)])


