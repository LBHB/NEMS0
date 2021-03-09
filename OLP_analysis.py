import numpy as np
import scipy.ndimage.filters as sf
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import sys
sys.path.extend(['/auto/users/hamersky/olp'])
import helpers as helpers
import scipy.stats as sst
from warnings import warn
import statsmodels.formula.api as smf
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def load_experiment_params(parmfile, rasterfs=100, sub_spont=True):
    """Given a parm file, or if I'm on my laptop, a saved experiment file, it will load the file
    and get relevant parameters about the experiment as well as sort out the sound indexes."""
    params = {}
    expt = BAPHYExperiment(parmfile)
    rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
    resp = rec['resp'].rasterize()
    if sub_spont == True:
        prestimsilence = resp.extract_epoch('PreStimSilence')
        # average over reps(0) and time(-1), preserve neurons
        spont_rate = np.expand_dims(np.nanmean(prestimsilence, axis=(0, -1)), axis=1)
        std_per_neuron = resp._data.std(axis=1, keepdims=True)
        std_per_neuron[std_per_neuron == 0] = 1
        resp = resp._modified_copy(data=(resp._data - spont_rate) / std_per_neuron)

    rec['resp'] = rec['resp'].rasterize()
    e = resp.epochs
    expt_params = expt.get_baphy_exptparams()   #Using Charlie's manager
    ref_handle = expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]

    params['animal'], params['experiment'] = parmfile.split('/')[-3], parmfile.split('/')[-2]
    params['fs'] = resp.fs
    params['PreStimSilence'], params['PostStimSilence'] = ref_handle['PreStimSilence'], ref_handle['PostStimSilence']
    params['Duration'], params['SilenceOnset'] = ref_handle['Duration'], ref_handle['SilenceOnset']
    params['max reps'] = e[e.name.str.startswith('STIM')].pivot_table(index=['name'], aggfunc='size').max()
    params['stim length'] = int(e.loc[e.name.str.startswith('REF')].iloc[0]['end']
                - e.loc[e.name.str.startswith('REF')].iloc[0]['start'])
    params['combos'] = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
          'Half BG/Half FG', 'Full BG/Half FG']
    params['Background'], params['Foreground'] = ref_handle['Background'], ref_handle['Foreground']

    soundies = list(ref_handle['SoundPairs'].values())
    params['pairs'] = [tuple([j for j in (soundies[s]['bg_sound_name'].split('.')[0],
                                  soundies[s]['fg_sound_name'].split('.')[0])])
                                    for s in range(len(soundies))]
    params['units'], params['response'] = resp.chans, resp
    params['rec'] = resp #could be rec, was using for PCA function, might need to fix with spont/std

    return params


def get_response(params, sub_spont=False):
    """A given experiment, pair, and unit will return the 8 sound combos, labeled and in the
    repeat x neuron x time raster. Returns that as well as some basic info about the
    data to pass to other functions.
    This is a pared down version with no plotting best used for the z-scores.
    2/8/2020 sub_spont basically defunct as norm and spontsub added to the loader"""
    full_response = np.empty((len(params['pairs']), len(params['combos']), params['max reps'],
                              len(params['units']), (params['stim length']*params['fs'])))
                    # pair x combo x rep(nan padded to max) x unit x time array
    full_response[:] = np.nan

    for pr, (BG, FG) in enumerate(params['pairs']):
        combo_names = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1',
                 f'STIM_{BG}-0.5-1_{FG}-0-1', f'STIM_{BG}-0.5-1_null', f'STIM_null_{FG}-0.5-1',
                 f'STIM_{BG}-0.5-1_{FG}-0.5-1', f'STIM_{BG}-0-1_{FG}-0.5-1']
        resps_list = [params['response'].extract_epoch(i) for i in combo_names]  # gives you a repeat X neuron X time raster
        for cmb, res in enumerate(resps_list):
            full_response[pr, cmb, :res.shape[0], :, :] = res

    if sub_spont == True:
        full_response = subtract_spont(full_response, params)

    return full_response


def subtract_spont(full_response, params):
    """Takes the raw response data and substracts the unit average during the prestimsilence period.
    Returns a new full response array (pair x combo x rep x unit x time
    2/8/2020 sub_spont basically defunct as norm and spontsub added to the loader"""
    silence_times = int(params['PreStimSilence'] * params['fs'])
    unit_silence_mean = np.nanmean(full_response[..., :silence_times], axis=(0, 1, 2, 4))
    unit_silence_mean = unit_silence_mean[None,None,None,:,None]
    response_nospont = full_response - unit_silence_mean

    return response_nospont


def get_z(resp_idx, full_response, params):
    """Uses a list of two or three sound combo and the responses to generate a
    *z-score* ready for plotting with the label of the component sounds."""
    if 2 < len(resp_idx) > 3:
        raise ValueError(f"resp_idx must be two or three values, {len(resp_idx)} given")

    z_params = {}
    if len(resp_idx) == 3:
        respA, respB = full_response[:, resp_idx[0], :, :, :], \
                       full_response[:, resp_idx[1], :, :, :]
        lenA = [np.count_nonzero(~np.isnan(respA[aa,:,0,0])) for aa in range(len(params['pairs']))]
        lenB = [np.count_nonzero(~np.isnan(respB[aa,:,0,0])) for aa in range(len(params['pairs']))]
        min_rep = np.min(np.stack((lenA,lenB),axis=1),axis=1)
        #add random sampling of repetitions
        resplin = [(respA[nn,:ii,:,:] + respB[nn,:ii,:,:]) for (nn, ii) in enumerate(min_rep)]
        mean_resplin = np.stack([resplin[rr].mean(axis=0) for rr in range(len(resplin))],axis=0)
        semlin = np.stack([stats.sem(resplin[ww], axis=0) for ww in range(len(resplin))])

        respAB = full_response[:, resp_idx[2], :, :, :]
        mean_respAB = np.nanmean(respAB, axis=1)
        semAB = stats.sem(respAB, nan_policy='omit', axis=1).data

        bads = np.logical_and(np.isclose(semlin, 0, rtol=0.04), np.isclose(semAB, 0, rtol=0.04))
        z_no_nan = np.nan_to_num((mean_respAB - mean_resplin) / (semAB + semlin))
        mean_diff = mean_respAB - mean_resplin
        z_no_nan[bads] = mean_diff[bads]

        label = f"{params['combos'][resp_idx[2]]} - Linear Sum"
    if len(resp_idx) == 2:
        respA, respB = full_response[:, resp_idx[0], :, :, :], \
                       full_response[:, resp_idx[1], :, :, :]
        mean_respA, mean_respB = np.nanmean(respA, axis=1), np.nanmean(respB, axis=1)
        semA, semB = stats.sem(respA, nan_policy='omit', axis=1).data, stats.sem(respB, nan_policy='omit', axis=1).data
        z_no_nan = np.nan_to_num((mean_respB - mean_respA) / (semB + semA))
        label = f"{params['combos'][resp_idx[1]]} - {params['combos'][resp_idx[0]]}"
    z_params['resp_idx'], z_params['label'] = resp_idx, label
    z_params['idx_names'] = [params['combos'][cc] for cc in z_params['resp_idx']]

    return z_no_nan, z_params

##plotting functions
def z_heatmaps_allpairs(resp_idx, response, params, sigma=None, arranged=False):
    """Plots a two column figure of subplots, one for each sound pair, displaying a heat map
    of the zscore for all the units."""
    zscore, z_params = get_z(resp_idx, response, params)

    if sigma is not None:
        zscore = sf.gaussian_filter1d(zscore, sigma, axis=2)
        zmin, zmax = np.min(np.min(zscore, axis=2)), np.max(np.max(zscore, axis=2))
        abs_max = max(abs(zmin),zmax)
    else:
        zmin, zmax = np.min(np.min(zscore, axis=1)), np.max(np.max(zscore, axis=1))
        abs_max = max(abs(zmin),zmax)

    fig, axes = plt.subplots(int(np.round(zscore.shape[0]/2)), 2)
    axes = np.ravel(axes, order='F')
    if int(zscore.shape[0]) % 2 != 0:
        axes[-1].spines['top'].set_visible(False), axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False), axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([]), axes[-1].set_xticks([])
        axes = axes[:-1]

    if arranged:
        prebin = int(params['PreStimSilence'] * params['fs'])
        postbin = int((params['stim length'] - params['PostStimSilence']) * params['fs'])
        z_time_avg = np.nanmean(zscore[:,:,prebin:postbin], axis=(0,2))
        idx = np.argsort(z_time_avg)
        zscore = zscore[:, idx, :]

    for cnt, ax in enumerate(axes):
        im = ax.imshow(zscore[cnt, :, :], aspect='auto', cmap='bwr',
                  extent=[-0.5, (zscore[cnt, :, :].shape[1] / params['fs']) -
                          0.5, zscore[cnt, :, :].shape[0], 0], vmin=-abs_max, vmax=abs_max)
        ax.set_title(f"Pair {cnt}: BG {params['pairs'][cnt][0]} - FG {params['pairs'][cnt][1]}",
                     fontweight='bold')
        ymin, ymax = ax.get_ylim()
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles='--', lw=1)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin + 0.3, xmax - 0.2)
        if cnt == int(np.around(zscore.shape[0] / 2) - 1) or cnt == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'Neurons', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - Combo Index {z_params['resp_idx']} - "
                 f"{z_params['idx_names']} - Sigma {sigma}\n"
                 f"{z_params['label']}", fontweight='bold')


def plot_combos(pair, unit, response, params, sigma=None):
    """Uses the response and a defined unit and pair to display psths of all the different
    sound combos, helpful for visualizing side by side."""
    resp_sub = response[pair, :, :, unit, :]
    mean_resp = np.nanmean(resp_sub, axis=1)

    fig, axes = plt.subplots(4,2, sharex=True, sharey=True, squeeze=True)
    axes = np.ravel(axes, order='F')
    x = np.linspace(0, resp_sub.shape[-1] / params['fs'], resp_sub.shape[-1]) - params['PreStimSilence']
    for count, ax in enumerate(axes):
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(mean_resp[count, :], sigma) * params['fs'], color='black')
            sem = stats.sem(resp_sub, axis=1, nan_policy='omit')
            ax.fill_between(x, sf.gaussian_filter1d((mean_resp[count, :] - sem[count, :]) * params['fs'], sigma),
                            sf.gaussian_filter1d((mean_resp[count, :] + sem[count, :]) * params['fs'], sigma),
                            alpha=0.35, color='black')
        else:
            ax.plot(x, mean_resp[count, :],color='black')
        ax.set_title(f"{params['combos'][count]}",fontweight='bold')
    ax.set_xlim(-0.3, (params['Duration'] + 0.2))        # arbitrary window I think is nice
    ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
    ymin, ymax = ax.get_ylim()
    for ax in axes:
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':')

    onset, dur = params['SilenceOnset'], params['Duration']
    seg, height = (dur - onset), (ymax * 0.2)
    for cnt, ax in enumerate(axes):
        if cnt == 0:
            ax.add_patch(matplotlib.patches.Rectangle((0, ymax-height), dur, height,
                                                      color="blue",alpha=0.25,ec=None))
        if cnt == 1:
            ax.add_patch(matplotlib.patches.Rectangle((0, ymax-(height*2)), dur, height,
                                                      color="green",alpha=0.25,ec=None))
        if cnt == 2:
            ax.add_patch(matplotlib.patches.Rectangle((0, ymax-height), dur, height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((0, ymax-(height*2)), dur, height,
                                                      color="green", alpha=0.25, ec=None))
        if cnt == 3:
            ax.add_patch(matplotlib.patches.Rectangle((onset, ymax-height), seg, height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((0, ymax-(height*2)), dur, height,
                                                      color="green",alpha=0.25,ec=None))
        if cnt == 4:
            ax.add_patch(matplotlib.patches.Rectangle((onset, ymax-height), seg, height,
                                                      color="blue",alpha=0.25,ec=None))
        if cnt == 5:
            ax.add_patch(matplotlib.patches.Rectangle((onset, ymax-(height*2)), seg, height,
                                                      color="green", alpha=0.25, ec=None))
        if cnt == 6:
            ax.add_patch(matplotlib.patches.Rectangle((onset, ymax-height), seg, height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((onset, ymax-(height*2)), seg, height,
                                                      color="green", alpha=0.25, ec=None))
        if cnt == 7:
            ax.add_patch(matplotlib.patches.Rectangle((0, ymax-height), dur, height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((onset, ymax-(height*2)), seg, height,
                                                      color="green", alpha=0.25, ec=None))

    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} \nPair {pair} - "
                 f"BG: {params['pairs'][pair][0]} - FG: {params['pairs'][pair][1]}",fontweight='bold')
    fig.tight_layout()


def plot_rasters(resp_idx, pair, unit, response, params, sigma=None):
    """Plots rasters of the specified sound combos vertically above a summary PSTH, can smooth.
    Must pass response without sponts taken out."""
    if len(resp_idx) > 4:
        raise ValueError("resp_idx is too long, this will look terrible. Use 4 or less. Ideally 3.")

    sub_resp = response[pair, :, :, unit, :]
    lens = [np.count_nonzero(~np.isnan(sub_resp[gg, :, 0])) for gg in resp_idx]
    spk_times = [np.where(sub_resp[idx, :ll, :]) for (idx, ll) in zip(resp_idx, lens)]

    colors = ['blue','green','black','red']
    fig, axes = plt.subplots((len(resp_idx) + 1), 1)
    for cnt, ax in enumerate(axes[:-1]):
        ax.plot((spk_times[cnt][1] / params['fs']) , (spk_times[cnt][0] / 1), '|', color='k', markersize=5)
        ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False), ax.spines['bottom'].set_visible(False)
        ax.set_xlim((params['PreStimSilence'] - 0.3), (params['PreStimSilence'] + params['Duration'] + 0.2))
        ax.set_ylabel(params['combos'][cnt],rotation='horizontal',ha='right',va='center')
        ax.set_xticks([]), ax.set_yticks([])
        ymin,ymax = ax.get_ylim()
        ax.add_patch(matplotlib.patches.Rectangle((params['PreStimSilence'],ymin), params['Duration'], ymax,
                                                      color=colors[cnt], alpha=0.2, ec=None))

    x = np.linspace(0, response.shape[-1] / params['fs'], response.shape[-1]) - params['PreStimSilence']
    for col,idx in enumerate(resp_idx):
        if sigma:
            axes[-1].plot(x, sf.gaussian_filter1d(np.nanmean(sub_resp[idx, :, :], axis=0), sigma)
                          * params['fs'], color=colors[col])
        else:
            axes[-1].plot(x, np.nanmean(sub_resp[idx,:,:], axis=0), color=colors[col])
    axes[-1].set_xlim(-0.3, (params['Duration'] + 0.2)), axes[-1].set_xticks([0, 0.5, 1.0])
    axes[-1].spines['top'].set_visible(False), axes[-1].spines['right'].set_visible(False)
    axes[-1].set_ylabel('spk/s'), axes[-1].set_xlabel('Time (s)')
    ymin,ymax = axes[-1].get_ylim()
    axes[-1].vlines([0, params['Duration']], ymin, ymax,colors='black',linestyles=':',lw=0.5)

    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - Pair {pair}\n"
                 f"BG: {params['pairs'][pair][0]} - FG: {params['pairs'][pair][1]}")
    fig.set_figwidth(9), fig.set_figheight(8)


def psth_comp(resp_idx, pair, unit, response, params, sigma=None, z=False, sum=False, ax=None):
    """Produces a single PSTH with however many lines based on the number of indices
    given in resp_idx. Sigma adds smoothing, z adds a second subplot showing the zscore
    between the indices indicated in resp_idx, sum being true shows the linear sum in a
    dotted line on the psth."""
    if len(resp_idx) <= 3:
        colors = ['deepskyblue','yellowgreen','dimgray']
    if len(resp_idx) == 4:
        colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray']
    if len(resp_idx) == 5:
        colors = ['deepskyblue', 'khaki', 'gold', 'lightcoral', 'firebrick']
    edit_fig = False
    if ax == None:
        if z and 1 < len(resp_idx) < 4:
            fig, axes = plt.subplots(2, 1, sharex=True)
            ax = axes[0]
        else:
            fig, ax = plt.subplots()
        edit_fig = True

    for col, idx in zip(colors, resp_idx):
        resp = response[pair, idx, :, unit, :]
        mean_resp = np.nanmean(resp, axis=0)
        x = np.linspace(0,response.shape[-1]/params['fs'],response.shape[-1]) - params['PreStimSilence']
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(mean_resp, sigma) * params['fs'], color=col,
                    label=f"{params['combos'][idx]}")
        if not sigma:
            ax.plot(x, mean_resp * params['fs'], color=col, label=f"{params['combos'][idx]}")
        sem = stats.sem(resp, axis=0, nan_policy='omit')
        ax.fill_between(x, sf.gaussian_filter1d((mean_resp - sem) * params['fs'], sigma),
                        sf.gaussian_filter1d((mean_resp + sem) * params['fs'], sigma),
                        alpha=0.5, color=col)
    if sum:
        respA, respB = response[pair, resp_idx[0], :, unit, :], \
                       response[pair, resp_idx[1], :, unit, :]
        lenA, lenB = np.count_nonzero(~np.isnan(respA[:,0])), np.count_nonzero(~np.isnan(respB[:,0]))
        min_rep = np.min((lenA,lenB))
        resplin = (respA[:min_rep,:] + respB[:min_rep,:])
        mean_resplin = resplin.mean(axis=0)
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(mean_resplin * params['fs'], sigma), color='dimgray',
                    ls='--', label='Linear Sum')
        if not sigma:
            ax.plot(x, mean_resplin * params['fs'], color='dimgray', ls='--', label='Linear Sum')

    ax.set_xlim(-0.3, (params['Duration'] + 0.2))        # arbitrary window I think is nice
    ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
    ymin, ymax = ax.get_ylim()

    ax.set_ylim(ymin,ymax)
    if edit_fig:
        # ax.set_ylim(ysmall,ybig)   #Thing I added to toggle on and off for WIP talk
        # ymin,ymax = ax.get_ylim()  #This too
        ax.set_ylabel('spk/s', fontweight='bold', size=15), ax.legend(loc='upper left')
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':')
        ax.vlines(params['SilenceOnset'], ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)

    if z and 1 < len(resp_idx) < 4:
        ax = axes[1]
        zscore, z_params = get_z(resp_idx, response, params)
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(zscore[pair, unit, :], sigma), color='black', label=z_params['label'])
        if not sigma:
            ax.plot(x, zscore[pair, unit, :], color='black', label=z_params['label'])
        ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin,ymax)
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':', lw=0.5)
        ax.vlines(params['SilenceOnset'],ymax*.9,ymax,colors='black',linestyles='-', lw=0.25)
        ax.legend(loc='upper left')

    ax.set_xlim(-0.3, (params['Duration'] + 0.2))        # arbitrary window I think is nice
    ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
    if edit_fig:
        ax.set_xlabel('Time (s)', fontweight='bold', size=15)
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        fig.set_figheight(6), fig.set_figwidth(15)
        fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - "
                     f"Pair {pair} - BG: {params['pairs'][pair][0]} - "
                     f"FG: {params['pairs'][pair][1]} - {resp_idx}", fontweight='bold', size=15)


def psth_allpairs(resp_idx, unit, response, params, sigma=None, sum=False):
    """Creates subplots for each sound pair where the PSTH is plotted for each sound combo
    indicated by resp_idx. Smooth with sigma and sum adds a dotted line to each psth showing
    the linear sum of the first two resp_idxs"""
    disp_pairs = response.shape[0]
    fig, axes = plt.subplots(int(np.round(disp_pairs/2)), 2, sharey=False)
    axes = np.ravel(axes, order='F')
    mins, maxs = [], []
    if int(disp_pairs) % 2 != 0:
        axes[-1].spines['top'].set_visible(False), axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False), axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([]), axes[-1].set_xticks([])
        axes = axes[:-1]
    for(ax, pair) in zip(axes, range(disp_pairs)):
        psth_comp(resp_idx, pair, unit, response, params, sigma, z=False, sum=sum, ax=ax)
        ax.set_title(f"Pair {pair}: BG {params['pairs'][pair][0]} - "
                     f"FG {params['pairs'][pair][1]}", fontweight='bold')
        if pair == 0:
            ax.legend(loc='upper left', prop={'size': 7})
        if pair == int(np.around(disp_pairs / 2) - 1) or pair == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])
        min, max = ax.get_ylim()
        mins.append(min), maxs.append(max)

    least, most = np.min(mins), np.max(maxs)
    for ax in axes:
        ax.set_ylim(least, most)
        ax.vlines([0, params['Duration']], least, most, colors='black', linestyles=':')
        ax.vlines(params['SilenceOnset'], most * .9, most, colors='black', linestyles='-', lw=0.25)

    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'spk/s', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - resp_idx {resp_idx}",
                 fontweight='bold')


def psth_allunits(resp_idx, pair, response, params, sigma=None, sum=False):

    disp_pairs = response.shape[3]
    dims = int(np.ceil(np.sqrt(disp_pairs)))
    mins, maxs = [], []

    fig, axes = plt.subplots(dims, dims, sharey=False)
    axes = np.ravel(axes, order='F')
    if dims**2 - disp_pairs != 0:
        for aa in range(dims**2 - disp_pairs):
            axes[-(aa+1)].spines['top'].set_visible(False), axes[-(aa+1)].spines['bottom'].set_visible(False)
            axes[-(aa+1)].spines['right'].set_visible(False), axes[-(aa+1)].spines['left'].set_visible(False)
            axes[-(aa+1)].set_xticks([])
        axes = axes[:-(dims**2 - disp_pairs)]
    for(ax, unit) in zip(axes, range(disp_pairs)):
        psth_comp(resp_idx, pair, unit, response, params, sigma, z=False, sum=sum, ax=ax)
        ax.set_title(f"Unit {unit} - {params['units'][unit]}", fontweight='bold')
        if unit == 0:
            ax.legend(loc='upper left', prop={'size': 5})
        if unit in [x-1 for x in (list(range(dims,dims**2,dims)))] or unit == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])

        min, max = ax.get_ylim()
        mins.append(min), maxs.append(max)

    least, most = np.min(mins), np.max(maxs)
    for ax in axes:
        ax.set_ylim(least, most)
        ax.vlines([0, params['Duration']], least, most, colors='black', linestyles=':')
        ax.vlines(params['SilenceOnset'], most * .9, most, colors='black', linestyles='-', lw=0.25)

    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'spk/s', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - resp_idx {resp_idx}\nPair {pair} - "
                 f"BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                 fontweight='bold')

def psth_fulls_allunits(pair, response, params, sigma=None):

    disp_pairs, resp_idx = response.shape[3], [0,1,2]
    dims = int(np.ceil(np.sqrt(disp_pairs)))

    fig, axes = plt.subplots(dims, dims, sharey=False)
    axes = np.ravel(axes, order='F')
    if dims**2 - disp_pairs != 0:
        for aa in range(dims**2 - disp_pairs):
            axes[-(aa+1)].spines['top'].set_visible(False), axes[-(aa+1)].spines['bottom'].set_visible(False)
            axes[-(aa+1)].spines['right'].set_visible(False), axes[-(aa+1)].spines['left'].set_visible(False)
            axes[-(aa+1)].set_xticks([]), axes[-(aa+1)].set_yticks([])
        axes = axes[:-(dims**2 - disp_pairs)]
    for(ax, unit) in zip(axes, range(disp_pairs)):
        psth_comp(resp_idx, pair, unit, response, params, sigma, z=False, sum=True, ax=ax)
        ax.set_title(f"Unit {unit} - {params['units'][unit]}", fontweight='bold')
        ax.hlines(0, -0.5, 1.5, linestyles=':')
        if unit == 0:
            ax.legend(loc='upper left', prop={'size': 5})
        if unit in [x-1 for x in (list(range(dims,dims**2,dims)))] or unit == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)

    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'spk/s', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - resp_idx {resp_idx}\nPair {pair} - "
                 f"BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                 fontweight='bold')

def z_allpairs(resp_idx, unit, response, params, sigma=None, z_av=False):
    """Creates subplots for each sound pair where the zscore is plotted for each sound combo
    indicated by resp_idx. Smooth with sigma and z_av creates a new figure that shows the average
    zscore from all the displayed subplots."""
    disp_pairs = response.shape[0]
    fig, axes = plt.subplots(int(np.round(disp_pairs / 2)), 2, sharey=True)
    axes = np.ravel(axes, order='F')
    zscore, z_params = get_z(resp_idx, response, params)
    x = np.linspace(0, response.shape[-1] / params['fs'], response.shape[-1]) - params['PreStimSilence']
    if int(disp_pairs) % 2 != 0:
        axes[-1].spines['top'].set_visible(False), axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False), axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([]), axes[-1].set_xticks([])
        axes = axes[:-1]
    for(ax, pair) in zip(axes, range(disp_pairs)):
        ax.plot(x, sf.gaussian_filter1d(zscore[pair, unit, :], sigma),
                color='black', label=z_params['label'])
        ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
        ax.set_xlim(-0.3, (params['Duration'] + 0.2))  # arbitrary window I think is nice
        ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        ax.set_title(f"Pair {pair}: BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                     fontweight='bold')
        if pair == 0:
            ax.legend(loc='upper left', prop={'size': 7})
        if pair == int(np.around(disp_pairs / 2) - 1) or pair == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])
    for (ax, pair) in zip(axes, range(disp_pairs)):
        ymin, ymax = ax.get_ylim()
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':', lw=0.5)
        ax.vlines(params['SilenceOnset'], ymax*.9, ymax,colors='black',linestyles='-', lw=0.25)
        ax.set_ylim(ymin, ymax)
    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'spk/s', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - resp_idx {resp_idx}",
                 fontweight='bold')

    if z_av == True:
        z_mean, z_sem = zscore[:, unit, :].mean(axis=0), stats.sem(zscore[:, unit, :], axis=0)
        fig, ax = plt.subplots()
        fig.set_figwidth(15), fig.set_figheight(6)
        ax.plot(x, sf.gaussian_filter1d(z_mean, sigma), color='black', label=z_params['label'])
        ax.fill_between(x, sf.gaussian_filter1d((z_mean - z_sem), sigma),
                        sf.gaussian_filter1d((z_mean + z_sem), sigma), alpha=0.5, color='black')
        ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
        ax.set_xlim(-0.3, (params['Duration'] + 0.2))
        ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':', lw=0.5)
        ax.vlines(params['SilenceOnset'],ymax*.9,ymax,colors='black',linestyles='-', lw=0.25)
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left'), ax.set_xlabel('Time (s)')
        ax.set_title(f"Experiment {params['experiment']} - Unit {unit} - "
                     f"resp_idx {resp_idx}", fontweight='bold')


def z_bgfg_compare(resp_idx, resp_idx2, unit, response, params, sigma=None):
    """Creates subplots for each sound pair where the zscore is plotted for each sound combo
    indicated by resp_idx. Smooth with sigma and z_av creates a new figure that shows the average
    zscore from all the displayed subplots."""
    disp_pairs = response.shape[0]
    fig, axes = plt.subplots(int(np.round(disp_pairs / 2)), 2, sharey=True)
    axes = np.ravel(axes, order='F')
    zscore, z_params = get_z(resp_idx, response, params)
    zscore2, z_params2 = get_z(resp_idx2, response, params)
    x = np.linspace(0, response.shape[-1] / params['fs'], response.shape[-1]) - params['PreStimSilence']
    if int(disp_pairs) % 2 != 0:
        axes[-1].spines['top'].set_visible(False), axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False), axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([]), axes[-1].set_xticks([])
        axes = axes[:-1]
    for(ax, pair) in zip(axes, range(disp_pairs)):
        ax.plot(x, sf.gaussian_filter1d(zscore[pair, unit, :], sigma),
                color='deepskyblue', label=z_params['label'])
        ax.plot(x, sf.gaussian_filter1d(zscore2[pair, unit, :], sigma),
                color='yellowgreen', label=z_params2['label'])
        ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
        ax.set_xlim(-0.3, (params['Duration'] + 0.2))  # arbitrary window I think is nice
        ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        ax.set_title(f"Pair {pair}: BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                     fontweight='bold')
        if pair == 0:
            ax.legend(loc='upper left', prop={'size': 7})
        if pair == int(np.around(disp_pairs / 2) - 1) or pair == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])
    for (ax, pair) in zip(axes, range(disp_pairs)):
        ymin, ymax = ax.get_ylim()
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':', lw=0.5)
        ax.vlines(params['SilenceOnset'], ymax*.9, ymax,colors='black',linestyles='-', lw=0.25)
        ax.set_ylim(ymin, ymax)
    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'spk/s', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - resp_idx {resp_idx}",
                 fontweight='bold')

    return zscore, zscore2, z_params, z_params2


def z_bgfg_compare2(zscore, z_params, zscore2, z_params2, unit, response, params, sigma=None):
    """Creates subplots for each sound pair where the zscore is plotted for each sound combo
    indicated by resp_idx. Smooth with sigma and z_av creates a new figure that shows the average
    zscore from all the displayed subplots."""
    disp_pairs = response.shape[0]
    fig, axes = plt.subplots(int(np.round(disp_pairs / 2)), 2, sharey=True)
    axes = np.ravel(axes, order='F')
    x = np.linspace(0, response.shape[-1] / params['fs'], response.shape[-1]) - params['PreStimSilence']
    if int(disp_pairs) % 2 != 0:
        axes[-1].spines['top'].set_visible(False), axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False), axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([]), axes[-1].set_xticks([])
        axes = axes[:-1]
    for(ax, pair) in zip(axes, range(disp_pairs)):
        ax.plot(x, sf.gaussian_filter1d(zscore[pair, unit, :]/zscore2[pair, unit, :], sigma),
                color='black', label=z_params['label'])
        ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
        ax.set_xlim(-0.3, (params['Duration'] + 0.2))  # arbitrary window I think is nice
        ax.set_xticks([0, (params['Duration'] / 2), params['Duration']])
        ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
        ax.set_title(f"Pair {pair}: BG {params['pairs'][pair][0]} - FG {params['pairs'][pair][1]}",
                     fontweight='bold')
        if pair == 0:
            ax.legend(loc='upper left', prop={'size': 7})
        if pair == int(np.around(disp_pairs / 2) - 1) or pair == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])
    for (ax, pair) in zip(axes, range(disp_pairs)):
        ymin, ymax = ax.get_ylim()
        ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':', lw=0.5)
        ax.vlines(params['SilenceOnset'], ymax*.9, ymax,colors='black',linestyles='-', lw=0.25)
        ax.set_ylim(ymin, ymax)
    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'spk/s', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - resp_idx {resp_idx}",
                 fontweight='bold')


##Bulk loads of parmfiles, right now just useful for histogram metrics below
def load_parms(parmfiles):
    responses, parameters = {}, {}
    for file in parmfiles:
        params = load_experiment_params(file)
        response = get_response(params, sub_spont=False)
        # _, response = _base_reliability(response, params,
        #                                            rep_dim=2, protect_dim=3, threshold=0.1)
        responses[params['experiment']], parameters[params['experiment']] = response, params

    return responses, parameters


def _histogram_metrics(parmfiles, bins=50):
    for cnt, parmfile in enumerate(parmfiles):
        params = load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
        response = get_response(params, sub_spont=False)
        corcoefs = _base_reliability(response, rep_dim=2, protect_dim=3)
        avg_resp = _significant_resp(response, params, protect_dim=3, time_dim=-1)
        if cnt == 0:
            signif, corco = avg_resp, corcoefs
        else:
            signif = np.concatenate((signif, avg_resp), axis=0)
            corco = np.concatenate((corco, corcoefs), axis=0)
    fig, ax = plt.subplots(1,2)
    ax[0].hist(signif, bins=bins)
    ax[0].set_title('Significance'), ax[0].set_xlabel('Time Averaged Normalized Response')
    ax[0].set_ylabel('Number of Cells')
    ax[1].hist(corco, bins=bins)
    ax[1].set_title('Reliability'), ax[1].set_xlabel('Reliability')

    return corco, signif


#Functions for selecting for good units only
def _base_reliability(response, rep_dim, protect_dim):
    '''
    :param raster: ndim array
    :param rep_dim: int. dimension corresponding to repetitions
    :protect_dim: int. dimension to keep outside of calculations
    :return: ndarray. Contain perasons R for each position in the protect_dim.
    '''
    # reorders dimensions, first is repetitions, second is protected_dim
    raster = np.moveaxis(response, [rep_dim, protect_dim], [0, -1])
    R = raster.shape[0]
    P = raster.shape[-1]

    # gets two subsamples across repetitions, and takes the mean across reps
    rep1 = np.nanmean(raster[0:-1:2, ...], axis=0)
    rep2 = np.nanmean(raster[1:R+1:2, ...], axis=0)

    resh1 = np.reshape(rep1,[-1, P])
    resh2 = np.reshape(rep2,[-1, P])

    corcoefs = np.empty(P)
    corcoefs[:] = np.nan
    for pp in range(P):
        r = sst.pearsonr(resh1[:,pp], resh2[:,pp])
        corcoefs[pp] = r[0]

    return corcoefs


def _significant_resp(response, params, protect_dim, time_dim=-1):
    pre_bin = int(params['PreStimSilence'] * params['fs'])
    post_bin = int(response.shape[time_dim] - (params['PostStimSilence'] * params['fs']))

    signal_response = response[...,pre_bin:post_bin]
    raster = np.moveaxis(signal_response, [protect_dim], [-1])
    S = tuple([*range(0, len(raster.shape[:-1]), 1)])
    avg_resp = np.nanmean(np.absolute(raster), axis=S)

    return avg_resp


def _find_good_units(response, params, corcoefs=None, corcoefs_threshold=None,
                    avg_resp=None, avg_threshold=None):
    '''Takes a site response and uses metrics gathered from other functions along
    with a threshold from each and filters so you only have good units left.
    Right now have to manually add additional metrics.'''
    all_mask = (corcoefs > corcoefs_threshold) & (avg_resp > avg_threshold)
    all_goodcells = np.asarray(params['units'])[all_mask]
    good_idx = np.where(all_mask==True)
    params['good_units'], params['good_idx'] = all_goodcells, good_idx

    good_response = response[:, :, :, all_mask, :]

    if len(all_goodcells) == 0:
        warn(f'no reliable cells found')
    print(f"Started with {len(params['units'])} units, found {len(all_goodcells)} reliable units.")

    return good_response
####


def plot_auc_mean(combo, response, params):
    colors = ['deepskyblue', 'yellowgreen', 'dimgray']
    pre_bin = int(params['PreStimSilence'] * params['fs'])
    post_bin = int(response.shape[-1] - (params['PostStimSilence'] * params['fs']))
    for cnt, cmb in enumerate(combo):
        resp_sub = np.nanmean(response[:, cmb, :, :, :], axis=0)
        mean_resp = np.nanmean(resp_sub[...,int(pre_bin):int(post_bin)], axis=0)
        x = np.linspace(0, mean_resp.shape[-1] / params['fs'], mean_resp.shape[-1])

        auc = np.sum(mean_resp, axis=1)
        center = np.sum(np.abs(mean_resp) * x, axis=1) / np.sum(np.abs(mean_resp), axis=1)
        plt.plot(auc, center, marker='o', linestyle='None', color=colors[cnt],
                 label=params['combos'][cmb])
        plt.xlabel('Area Under Curve'), plt.ylabel('Center')
        plt.title(f"Experiment {params['experiment']} - Combos {combo}")
        plt.legend()


def multi_exp_auccenter(combo, responses, parameters):
    markers = ['o', '.', 'v', 'x', '+', 'v', '^', '<', '>', 's', 'd', '*']
    for cnt,exp in enumerate(responses.keys()):
        pre_bin = int(parameters[exp]['PreStimSilence'] * parameters[exp]['fs'])
        post_bin = int(responses[exp].shape[-1] - (parameters[exp]['PostStimSilence'] * parameters[exp]['fs']))
        resp_sub = np.nanmean(responses[exp][:, combo, :, :, :], axis=0)
        mean_resp = np.nanmean(resp_sub[..., int(pre_bin):int(post_bin)], axis=0)
        x = np.linspace(0, mean_resp.shape[-1] / parameters[exp]['fs'], mean_resp.shape[-1])

        auc = np.sum(mean_resp, axis=1)
        center = np.sum(np.abs(mean_resp) * x, axis=1) / np.sum(np.abs(mean_resp), axis=1)

        plt.plot(auc, center, marker=markers[cnt], linestyle='None', label=exp)
        plt.legend()
    plt.xlabel('Area Under Curve'), plt.ylabel('Center')
    plt.title(f"Combos {combo} {parameters[exp]['combos'][combo]}")

##PCA Stuff

def plot_projections(pair, params):
    nbins = params['fs'] * params['stim length']
    time = np.linspace(0, nbins / params['response'].fs, nbins) - params['PreStimSilence']
    bg_name, fg_name = params['pairs'][pair][0], params['pairs'][pair][1]

    bg_epoch = f"{bg_name}-0-{params['Duration']}"
    fg_epoch = f"{fg_name}-0-{params['Duration']}"
    fg_bg_epoch = f"STIM_{bg_name}-{params['SilenceOnset']}-{params['Duration']}_" \
                  f"{fg_name}-0-{params['Duration']}"
    bg_fg_epoch = f"STIM_{bg_name}-0-{params['Duration']}_" \
                  f"{fg_name}-{params['SilenceOnset']}-{params['Duration']}"
    combo_epoch = f"STIM_{bg_name}-0-{params['Duration']}_" \
                  f"{fg_name}-0-{params['Duration']}"
    decoder = helpers.get_decoder(params['rec'], fg_epoch, bg_epoch, collapse=False)

    f = plt.figure(figsize=(11, 9))
    tser1 = plt.subplot2grid((3, 5), (0, 0), rowspan=1, colspan=3)
    tser2 = plt.subplot2grid((3, 5), (1, 0), rowspan=1, colspan=3)
    tser3 = plt.subplot2grid((3, 5), (2, 0), rowspan=1, colspan=3)
    sim = plt.subplot2grid((3, 5), (1, 3), rowspan=1, colspan=2)

    combos, axs = [fg_bg_epoch, bg_fg_epoch, combo_epoch], [tser1, tser2, tser3]
    names = ['half BG, full FG', 'full BG, half FG', 'full BG, full FG']

    bg_resp = params['response'].extract_epoch('STIM_'+bg_epoch+'_null')
    fg_resp = params['response'].extract_epoch('STIM_null_'+fg_epoch)
    # project single trials onto each decoding axis for responses in isolation
    bg_resp = np.concatenate([bg_resp[:, :, i] @ decoder[[i], :].T for i in range(decoder.shape[0])], axis=-1)
    fg_resp = np.concatenate([fg_resp[:, :, i] @ decoder[[i], :].T for i in range(decoder.shape[0])], axis=-1)

    for cnt, (epoch, tser) in enumerate(zip(combos, axs)):
        combo_resp = params['response'].extract_epoch(epoch)
        # project single trials onto each decoding axis for combo responses
        combo_resp = np.concatenate([combo_resp[:, :, i] @ decoder[[i], :].T for i in range(decoder.shape[0])], axis=-1)

        #plot
        tser.plot(time, fg_resp.mean(axis=0), '--', label='fg (isolation)', color='tab:blue', lw=2)
        tser.plot(time, bg_resp.mean(axis=0), '--', label='bg (isolation)', color='tab:orange', lw=2)
        mean, sem = combo_resp.mean(axis=0), combo_resp.std(axis=0)
        tser.plot(time, mean, lw=2, color='grey')
        tser.fill_between(time, mean - sem, mean + sem, lw=0, alpha=0.5, color='grey', label=names[cnt])
        tser.axvline(params['PreStimSilence'], color='k', lw=2, label='transition', zorder=0)

        tser.legend(frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
        tser.set_ylabel('Decoding axis projection')
        tser.set_xlabel('Trial time')
        tser.set_title(combo_epoch)
        tser.set_xlim((time[0], time[-1]))

    sim.imshow(np.abs(decoder @ decoder.T), aspect='auto', cmap='Reds', vmin=0, vmax=1,
               extent=[time[0], time[-1], time[0], time[-1]])
    sim.set_title(f'Decoding axes similarity - Pair {pair}')

    f.tight_layout()
    plt.show()

##############################################
####scatter plots for initial data viewing####

def get_scatter_resps(pair, response):
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

    return mean_bg, mean_fg, mean_lin, mean_combo, supp, bg_fg, supp_supp


def bg_fg_scatter(pair, response, params):
    '''Plots a simple scatter with BG alone response on the x and FG alone response
    on the y with a unity line. Each point is a unit so you can see if at a site
    units respond more to BG or FGs'''
    mean_bg, mean_fg, _, _, _, _, _ = get_scatter_resps(pair, response)
    fig, ax = plt.subplots()
    for pnt in range(len(params['good_units'])):
        ax.plot(mean_bg[pnt], mean_fg[pnt], marker='o', color='black', linestyle='None')

    lims = np.stack((ax.get_xlim(), ax.get_ylim()), axis=0)
    top_lim, bottom_lim = np.min(lims[:, 1]), np.max(lims[:, 0])
    ax.plot((bottom_lim, top_lim), (bottom_lim, top_lim), linestyle=':', color='black')
    ax.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(params['good_units']):
        ax.annotate(f'    {txt}', (mean_bg[i], mean_fg[i]), size=5)
    ax.set_xlabel('BG Response'), ax.set_ylabel('FG Response')
    fig.suptitle(f"Experiment {params['experiment']} - Pair {pair}\n"
                 f"Background {params['pairs'][pair][0]} - Foreground {params['pairs'][pair][1]}",
                 fontweight='bold')
    fig.tight_layout()


def lin_combo_scatter(pair, response, params):
    '''Plots a simple scatter with the linear sum of BG and FG alone on x and BG/FG
    combo response on the y with a unity line. Each point is a unit so you can see
    if at a site units show more suppression or enhancement'''
    _, _, mean_lin, mean_combo, _, _, _ = get_scatter_resps(pair, response)
    fig, ax = plt.subplots()
    for pnt in range(len(params['good_units'])):
        ax.plot(mean_lin[pnt], mean_combo[pnt], marker='o', color='black', linestyle='None')
    lims = np.stack((ax.get_xlim(), ax.get_ylim()), axis=0)
    top_lim, bottom_lim = np.min(lims[:, 1]), np.max(lims[:, 0])
    ax.plot((bottom_lim, top_lim), (bottom_lim, top_lim), linestyle=':', color='black')
    ax.set_aspect('equal', adjustable='box')
    for i, txt in enumerate(params['good_units']):
        ax.annotate(f'    {txt}', (mean_lin[i], mean_combo[i]), size=5)
    ax.set_xlabel('Linear Sum BG Alone + FG Alone'), ax.set_ylabel('Full BG/Full FG Combo Response')
    fig.suptitle(f"Experiment {params['experiment']} - Pair {pair}\n"
                 f"Background {params['pairs'][pair][0]} - Foreground {params['pairs'][pair][1]}",
                 fontweight='bold')
    fig.tight_layout()


def bgfg_lincombo_scatter(pair, response, params):
    '''Combines bg_fg_scatter and lin_combo_scatter into one figure'''
    mean_bg, mean_fg, mean_lin, mean_combo, supp, _, _ = get_scatter_resps(pair, response)
    fig, ax = plt.subplots(1, 2)
    for pnt in range(len(params['good_units'])):
        ax[0].plot(mean_bg[pnt], mean_fg[pnt], marker='o', color='black', linestyle='None')
    lims = np.stack((ax[0].get_xlim(), ax[0].get_ylim()), axis=0)
    top_lim, bottom_lim = np.min(lims[:, 1]), np.max(lims[:, 0])
    ax[0].plot((bottom_lim, top_lim), (bottom_lim, top_lim), linestyle=':', color='black')
    ax[0].set_aspect('equal', adjustable='box')
    for i, txt in enumerate(params['good_units']):
        ax[0].annotate(f'    {txt}', (mean_bg[i], mean_fg[i]), size=5)
    ax[0].set_xlabel('BG Response'), ax[0].set_ylabel('FG Response')

    for pnt in range(len(params['good_units'])):
        ax[1].plot(mean_lin[pnt], mean_combo[pnt], marker='o', color='black', linestyle='None')

    lims = np.stack((ax[1].get_xlim(), ax[1].get_ylim()), axis=0)
    top_lim, bottom_lim = np.min(lims[:, 1]), np.max(lims[:, 0])
    ax[1].plot((bottom_lim, top_lim), (bottom_lim, top_lim), linestyle=':', color='black')
    ax[1].set_aspect('equal', adjustable='box')
    for i, txt in enumerate(params['good_units']):
        ax[1].annotate(f'    {txt}', (mean_lin[i], mean_combo[i]), size=5)
    ax[1].set_xlabel('Linear Sum BG Alone + FG Alone'), ax[1].set_ylabel('Full BG/Full FG Combo Response')

    fig.suptitle(f"Experiment {params['experiment']} - Pair {pair}\n"
                 f"Background {params['pairs'][pair][0]} - Foreground {params['pairs'][pair][1]}",
                 fontweight='bold')
    fig.tight_layout()


def bgfg_suppression_scatter(pair, response, params):
    '''Plot one figure with the normalized response to BG and FG alone on the x axis
    with suppression on the y axis. Points will be connected by a dot from the same
    unit. Units are labeled on the plot to avoid a lot of colors.'''
    _, _, _, _, supp, bg_fg, supp_supp = get_scatter_resps(pair, response)
    fig, ax = plt.subplots()
    for pnt in range(supp.shape[0]):
        ax.plot(bg_fg[pnt, :], supp_supp[pnt, :], marker='', color='black', linestyle=':',
                zorder=-1)
    ax.scatter(bg_fg[:, 0], supp_supp[:, 0], marker='o', color='deepskyblue', label='BG')
    ax.scatter(bg_fg[:, 1], supp_supp[:, 1], marker='o', color='yellowgreen', label='FG')
    ax.legend(loc='upper left')

    for i, txt in enumerate(params['good_units']):
        ax.annotate(txt, (bg_fg[i, 0], supp_supp[i, 0]), size=6)
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('Suppression\n(BG + FG) - BG/FG combo')
    ax.set_xlabel('Normalized Response')
    fig.suptitle(f"Experiment {params['experiment']} - Pair {pair}\n"
                 f"Background {params['pairs'][pair][0]} - Foreground {params['pairs'][pair][1]}",
                 fontweight='bold')
    fig.tight_layout()

##################
#Regression stuff
def _get_suppression(response, params):
    supp_array = np.empty([len(params['good_units']), len(params['pairs'])])
    for nn, pp in enumerate(params['pairs']):
        _, _, _, _, supp, _, _ = get_scatter_resps(nn, response)
        supp_array[:, nn] = supp

    return supp_array

def site_regression(supp_array, params):
    site_results = pd.DataFrame()
    shuffles = [None, 'neuron', 'stimulus']
    for shuf in shuffles:
        reg_results = neur_stim_reg(supp_array, params, shuf)
        site_results = site_results.append(reg_results, ignore_index=True)

    return site_results


def neur_stim_reg(supp_array, params, shuffle=None):
    y = supp_array.reshape(1,-1) #flatten
    stimulus = np.tile(np.arange(0,supp_array.shape[1]), supp_array.shape[0])
    neuron = np.concatenate([np.ones(supp_array.shape[1]) * i for i in
                         range(supp_array.shape[0])], axis=0)

    X = np.stack([neuron,stimulus])
    X = pd.DataFrame(data=X.T,columns=['neuron','stimulus'])
    X = sm.add_constant(X)
    X['suppression'] = y.T

    if not shuffle:
        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=X).fit()

    if shuffle == 'neuron':
        Xshuff = X.copy()
        Xshuff['neuron'] = Xshuff['neuron'].iloc[np.random.choice(
                    np.arange(X.shape[0]),X.shape[0],replace=False)].values
        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    if shuffle == 'stimulus':
        Xshuff = X.copy()
        Xshuff['stimulus'] = Xshuff['stimulus'].iloc[np.random.choice(
            np.arange(X.shape[0]),X.shape[0],replace=False)].values
        results = smf.ols(formula='suppression ~ C(neuron) + C(stimulus) + const', data=Xshuff).fit()

    reg_results = _regression_results(results, shuffle, params)

    return reg_results


def _regression_results(results, shuffle, params):
    intercept = results.params.loc[results.params.index.str.contains('Intercept')].values
    int_err = results.bse.loc[results.bse.index.str.contains('Intercept')].values
    int_conf = results.conf_int().loc[results.conf_int().index.str.contains('Intercept')].values[0]
    neuron_coeffs = results.params.loc[results.params.index.str.contains('neuron')].values
    neuron_coeffs = np.concatenate(([0], neuron_coeffs))
    stim_coeffs = results.params.loc[results.params.index.str.contains('stimulus')].values
    stim_coeffs = np.concatenate(([0], stim_coeffs))
    neur_coeffs = neuron_coeffs + intercept + stim_coeffs.mean()
    stim_coeffs = stim_coeffs + intercept + neuron_coeffs.mean()
    coef_list = np.concatenate((neur_coeffs, stim_coeffs))

    neuron_err = results.bse.loc[results.bse.index.str.contains('neuron')].values
    stim_err = results.bse.loc[results.bse.index.str.contains('stimulus')].values
    neuron_err = np.concatenate((int_err, neuron_err))
    stim_err = np.concatenate((int_err, stim_err))
    err_list = np.concatenate((neuron_err, stim_err))

    neur_low_conf = results.conf_int()[0].loc[results.conf_int().index.str.contains('neuron')].values
    neur_low_conf = np.concatenate(([int_conf[0]], neur_low_conf))
    stim_low_conf = results.conf_int()[0].loc[results.conf_int().index.str.contains('stimulus')].values
    stim_low_conf = np.concatenate(([int_conf[0]], stim_low_conf))
    low_list = np.concatenate((neur_low_conf, stim_low_conf))

    neur_high_conf = results.conf_int()[1].loc[results.conf_int().index.str.contains('neuron')].values
    neur_high_conf = np.concatenate(([int_conf[1]], neur_high_conf))
    stim_high_conf = results.conf_int()[1].loc[results.conf_int().index.str.contains('stimulus')].values
    stim_high_conf = np.concatenate(([int_conf[1]], stim_high_conf))
    high_list = np.concatenate((neur_high_conf, stim_high_conf))

    neur_list = ['neuron'] * len(neur_coeffs)
    stim_list = ['stimulus'] * len(stim_coeffs)
    name_list = np.concatenate((neur_list, stim_list))

    if shuffle == None:
        shuffle = 'full'
    shuff_list = [f"{shuffle}"] * len(name_list)
    site_list = [f"{params['experiment']}"] * len(name_list)
    r_list = [f"{np.round(results.rsquared,4)}"] * len(name_list)

    name_list_actual = list(params['good_units'])
    name_list_actual.extend(params['pairs'])

    reg_results = pd.DataFrame(
        {'name': name_list_actual,
         'id': name_list,
         'site': site_list,
         'shuffle': shuff_list,
         'coeff': coef_list,
         'error': err_list,
         'conf_low': low_list,
         'conf_high': high_list,
         'rsquare': r_list
        })

    return reg_results


def multisite_reg_results(parmfiles):
    regression_results = pd.DataFrame()
    for file in parmfiles:
        params = load_experiment_params(file, rasterfs=100, sub_spont=True)
        response = get_response(params, sub_spont=False)
        corcoefs = _base_reliability(response, rep_dim=2, protect_dim=3)
        avg_resp = _significant_resp(response, params, protect_dim=3, time_dim=-1)
        response = _find_good_units(response, params,
                                    corcoefs=corcoefs, corcoefs_threshold=0.1,
                                    avg_resp=avg_resp, avg_threshold=0.2)
        supp_array = _get_suppression(response, params)
        site_results = site_regression(supp_array, params)

        regression_results = regression_results.append(site_results, ignore_index=True)

    return regression_results

##General load
def _response_params(parmfile):
    params = load_experiment_params(parmfile, rasterfs=100, sub_spont=True)
    response = get_response(params, sub_spont=False)
    corcoefs = _base_reliability(response, rep_dim=2, protect_dim=3)
    avg_resp = _significant_resp(response, params, protect_dim=3, time_dim=-1)
    response = _find_good_units(response, params,
                                    corcoefs=corcoefs, corcoefs_threshold=0.1,
                                    avg_resp=avg_resp, avg_threshold=0.2)
    return response, params