import numpy as np
import scipy.ndimage.filters as sf
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats


def compare_combos(resp_idx,sigma=None,z=True,sum=None,fs=rasterfs,expt_resp=response,expt_ids=ids):
    '''A little function that decides how you will visualize your results based on the
    parameters you give it. If you want z-scores resp_idx must be 2 or 3 indexes long,
    if not, it'll just plot a regular PSTH. Sum being True will indicate that the first
    two given indices should be also shown as a linear sum. Sigma indicates whether
    rasters should be smoothed (probably always will do that).'''
    if z and 1 < len(resp_idx) < 4:
        zscore = psth_comp_z(resp_idx, sigma, fs, expt_resp, expt_ids)
        return zscore
    else:
        if z:
            print(f"List of indices must be 2 or 3, you had {len(resp_idx)}. "
                  f"Only displaying PSTH.")
        psth_comp(resp_idx, sigma, sum, fs, expt_resp, expt_ids)
        return "No z-score to return"




def z_compare(experiment, unit, resp_idx, sigma=2, plot_psth=False, plot_z=False, plot_av_z=True,
              expt_resp=resp, expt_pairs=Pairs, BGs=backgrounds, FGs=foregrounds, fs=rasterfs):
    expt = f'HOD00{experiment}'
    if 0 in resp_idx and 1 in resp_idx:
        sum_type = 'linear'
    if (0 in resp_idx and 5 in resp_idx) or (1 in resp_idx and 4 in resp_idx):
        sum_type = 'combo'

    zs, z_labels, resps, tags, z_list = {}, {}, {}, {}, []
    tags['experiment'], tags['unit'], tags['idx'] = experiment, unit, resp_idx
    for aa in range(len(expt_pairs[expt])):
        response, ids = get_response(int(expt[-1]), aa, unit, expt_resp, expt_pairs, BGs, FGs)
        zscore, label = get_z(resp_idx, response, unit)
        z_labels[ids['pair']], zs[ids['pair']], resps[ids['pair']] = ids['sounds'], zscore, response
        z_list.append(zscore)
    combo_labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    tags['idx_names'] = [combo_labels[i] for i in resp_idx]
    tags['legend'] = label

    if plot_psth == True:
        disp_pairs = len(resps)
        fig, axes = plt.subplots(int(np.round(disp_pairs/2)), 2, sharex=True, sharey=True)
        axes = np.ravel(axes, order='F')
        for (ax, pair) in zip(axes,resps.keys()):
            psth_comp(resp_idx, expt_resp=resps[pair], expt_ids=tags,
                      sigma=sigma, sum=sum_type, fs=fs, ax=ax)
            ax.set_title(f"Pair {pair}: BG {z_labels[pair][0]} - FG {z_labels[pair][1]}", fontweight='bold')
            if pair == 0:
                ax.legend(loc='upper left')
        if disp_pairs % 2 != 0:
            axes[-1].spines['top'].set_visible(False)
            axes[-1].spines['bottom'].set_visible(False)
            axes[-1].spines['right'].set_visible(False)
            axes[-1].spines['left'].set_visible(False)
            axes[-1].set_yticks([])
            axes[-1].set_xticks([])
        fig.suptitle(f"Experiment HOD00{tags['experiment']} - Unit {tags['unit']} - "
                     f"idx {tags['idx']}", fontweight='bold')

    if plot_z == True:
        disp_pairs = len(resps)
        fig, axes = plt.subplots(int(np.round(disp_pairs/2)), 2, sharex=True, sharey=True)
        axes = np.ravel(axes, order='F')
        x = np.linspace(0,zs[0].shape[0]/fs,zs[0].shape[0])-0.5
        for (ax, pair) in zip(axes, zs.keys()):
            ax.plot(x, sf.gaussian_filter1d(zs[pair], sigma), color='black', label=tags['legend'])
            ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
            ax.set_xlim(-0.3, 1.2)
            ymin, ymax = ax.get_ylim()
            ax.set_ylim(ymin, ymax)
            ax.vlines([0, 1], ymin, ymax, colors='black', linestyles=':', lw=0.5)  # unhard code the 1
            ax.vlines(0.5, ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title(f"Pair {pair}: BG {z_labels[pair][0]} - FG {z_labels[pair][1]}", fontweight='bold')
            if pair == 0:
                ax.legend(loc='upper left')
        if disp_pairs % 2 != 0:
            axes[-1].spines['top'].set_visible(False)
            axes[-1].spines['bottom'].set_visible(False)
            axes[-1].spines['right'].set_visible(False)
            axes[-1].spines['left'].set_visible(False)
            axes[-1].set_yticks([])
            axes[-1].set_xticks([])
        fig.suptitle(f"Experiment HOD00{tags['experiment']} - Unit {tags['unit']} - "
                     f"idx {tags['idx']}", fontweight='bold')
    if plot_av_z == True:
        z_scores = np.vstack(z_list)
        z_mean = z_scores.mean(axis=0)
        z_sem = stats.sem(z_scores, axis=0)
        fig, ax = plt.subplots()
        fig.set_figwidth(15)
        fig.set_figheight(6)
        x = np.linspace(0,zs[0].shape[0]/fs,zs[0].shape[0])-0.5
        ax.plot(x, sf.gaussian_filter1d(z_mean, sigma), color='black', label=tags['legend'])
        ax.fill_between(x, sf.gaussian_filter1d((z_mean - z_sem), sigma),
                        sf.gaussian_filter1d((z_mean + z_sem), sigma), alpha=0.5, color='black')
        ax.hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
        ax.set_xlim(-0.3, 1.2)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax)
        ax.vlines([0, 1], ymin, ymax, colors='black', linestyles=':', lw=0.5)  # unhard code the 1
        ax.vlines(0.5, ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(loc='upper left')
        ax.set_xlabel('Time (s)')
        ax.set_title(f"Experiment HOD00{tags['experiment']} - Unit {tags['unit']} - "
                     f"idx {tags['idx']}", fontweight='bold')

    return z_labels, zs, resps, z_list, tags




def heat_map(experiment, pair, combo_idx=None, expt_resp=resp, expt_pairs=Pairs, fs=rasterfs, BGs=backgrounds, FGs=foregrounds):
    response, ids = get_response(experiment, pair, None, expt_resp, expt_pairs, BGs, FGs)

    if combo_idx:
        fig, axes = plt.subplots(1)
        mean_resp = response[combo_idx][1].mean(axis=0)
        axes.imshow(mean_resp, aspect='auto', cmap='inferno',
                  extent=[-0.5,(mean_resp.shape[1]/fs)-0.5,mean_resp.shape[0],0])
        ymin, ymax = axes.get_ylim()
        axes.vlines([0 - (0.5 / fs), 1 - (0.5 / fs)], ymin, ymax, colors='white', linestyles='--',
                  lw=1)  # unhard code the 1
        xmin, xmax = axes.get_xlim()
        axes.set_xlim(xmin+0.3,xmax-0.2)
        axes.set_xticks([0,0.5,1.0])
        axes.set_ylabel('Neurons', fontweight='bold')
        axes.set_xlabel('Time from onset (s)', fontweight='bold')
        fig.suptitle(f"Experiment HOD00{ids['experiment']} - Pair {ids['pair']} - "
                     f"Combo {combo_idx} {ids['labels'][combo_idx]}\n"
                     f"BG: {ids['sounds'][0]} - FG: {ids['sounds'][1]}",fontweight='bold')

    else:
        fig, axes = plt.subplots(4,2, sharex=True, sharey=True, squeeze=True)
        axes = np.ravel(axes, order='F')

        for cnt,ax in enumerate(axes):
            mean_resp = response[cnt][1].mean(axis=0)
            ax.imshow(mean_resp, aspect='auto', cmap='inferno',
                      extent=[-0.5,(mean_resp.shape[1]/fs)-0.5,mean_resp.shape[0],0])
            ax.set_title(f"{ids['labels'][cnt]}", fontweight='bold')
            ymin, ymax = ax.get_ylim()
            ax.vlines([0-(0.5/fs),1-(0.5/fs)], ymin, ymax, colors='white', linestyles='--', lw=1)  # unhard code the 1

        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin+0.3,xmax-0.2)
        ax.set_xticks([0,0.5,1.0])

        fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
        fig.text(0.1, 0.5, 'Neurons', ha='center', va='center', rotation='vertical', fontweight='bold')
        fig.suptitle(f"Experiment HOD00{ids['experiment']} - Pair {ids['pair']} \n"
                     f"BG: {ids['sounds'][0]} - FG: {ids['sounds'][1]}",fontweight='bold')


def heat_map_allpairs(experiment, combo_idx, sigma=None, expt_resp=resp, expt_pairs=Pairs, fs=rasterfs, BGs=backgrounds, FGs=foregrounds):
    expt = f'HOD00{experiment}'
    sound_pairs, resps, tags = {}, {}, {}
    tags['experiment'], tags['idx'] = experiment, combo_idx
    for aa in range(len(expt_pairs[expt])):
        response, ids = get_response(int(expt[-1]), aa, None, expt_resp, expt_pairs, BGs, FGs)
        resps[ids['pair']], sound_pairs[ids['pair']] = response, ids['sounds']
    tags['idx_name'] = ids['labels'][combo_idx]

    mean_resp_list = []
    for aa in range(len(resps)):
        mean = resps[aa][combo_idx][1].mean(axis=0)
        mean_resp_list.append(mean)
    mean_response = np.stack(mean_resp_list,axis=2)   #creates array unit x time x pair
    zmin, zmax = np.min(np.min(mean_response, axis=1)), np.max(np.max(mean_response, axis=1))

    if sigma:
        mean_response = sf.gaussian_filter1d(mean_response, sigma, axis=1)

    fig, axes = plt.subplots(int(np.round(len(resps)/2)), 2)
    axes = np.ravel(axes, order='F')
    if int(len(resps)/2) % 2 != 0:
        axes[-1].spines['top'].set_visible(False)
        axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False)
        axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([])
        axes[-1].set_xticks([])
        axes = axes[:-1]

    for cnt, ax in enumerate(axes):
        # mean_resp = resps[cnt][combo_idx][1].mean(axis=0)
        ax.imshow(mean_response[:,:,cnt], aspect='auto', cmap='inferno',
                  extent=[-0.5, (mean_response[:,:,cnt].shape[1] / fs) -
                          0.5, mean_response[:,:,cnt].shape[0], 0], vmin=zmin, vmax=zmax)
        ax.set_title(f"Pair {cnt}: BG {sound_pairs[cnt][0]} - FG {sound_pairs[cnt][1]}",
                     fontweight='bold')
        ymin, ymax = ax.get_ylim()
        ax.vlines([0 - (0.5 / fs), 1 - (0.5 / fs)], ymin, ymax, colors='white', linestyles='--',
                  lw=1)  # unhard code the 1
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin + 0.3, xmax - 0.2)
        if cnt == int(np.around(len(resps)/2) - 1) or cnt == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])

    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'Neurons', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment HOD00{ids['experiment']} - Combo Index {tags['idx']} - "
                 f"{tags['idx_name']} - Sigma {sigma}", fontweight='bold')




##Doing life better
def load_experiment_params(parmfile):
    """Given a parm file, or if I'm on my laptop, a saved experiment file, it will load the file
    and get relevant parameters about the experiment as well as sort out the sound indexes."""
    rasterfs = 100
    params = {}
    expt = BAPHYExperiment(parmfile)
    rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
    resp = rec['resp'].rasterize()
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


    soundies = list(ref_handle['SoundPairs'].values())
    params['pairs'] = [tuple([j for j in (soundies[s]['bg_sound_name'].split('.')[0],
                                  soundies[s]['fg_sound_name'].split('.')[0])])
                                    for s in range(len(soundies))]
    params['units'], params['response'] = resp.chans, resp

    return params


def get_response(params):
    '''A given experiment, pair, and unit will return the 8 sound combos, labeled and in the
    repeat x neuron x time raster. Returns that as well as some basic info about the
    data to pass to other functions.
    This is a pared down version with no plotting best used for the z-scores.'''
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

    return full_response


def subtract_spont(full_response, params):

    silence_times = int(params['PreStimSilence'] * params['fs'])
    unit_silence_mean = np.nanmean(full_response[..., :silence_times], axis=(0, 1, 2, 4))
    unit_silence_mean = unit_silence_mean[None,None,None,:,None]
    response_nospont = full_response - unit_silence_mean

    return response_nospont


def get_z(resp_idx, full_response, params):
    '''Uses a list of two or three sound combo types and the responses to generate a
    *z-score* ready for plotting with the label of the component sounds.'''
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


def z_heatmaps(zscore, params, z_params, sigma=None):

    if sigma is not None:
        zscore = sf.gaussian_filter1d(zscore, sigma, axis=2)
        zmin, zmax = np.min(np.min(zscore, axis=2)), np.max(np.max(zscore, axis=2))
        abs_max = max(abs(zmin),zmax)
    else:
        zmin, zmax = np.min(np.min(zscore, axis=1)), np.max(np.max(zscore, axis=1))
        abs_max = max(abs(zmin),zmax)

    fig, axes = plt.subplots(int(np.round(zscore.shape[0]/2)), 2)
    axes = np.ravel(axes, order='F')
    if int(zscore.shape[0] / 2) % 2 != 0:
        axes[-1].spines['top'].set_visible(False)
        axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False)
        axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([])
        axes[-1].set_xticks([])
        axes = axes[:-1]

    for cnt, ax in enumerate(axes):
        im = ax.imshow(zscore[cnt, :, :], aspect='auto', cmap='bwr',
                  extent=[-0.5, (zscore[cnt, :, :].shape[1] / params['fs']) -
                          0.5, zscore[cnt, :, :].shape[0], 0], vmin=-abs_max, vmax=abs_max)
        ax.set_title(f"Pair {cnt}: BG {params['pairs'][cnt][0]} - FG {params['pairs'][cnt][1]}",
                     fontweight='bold')
        ymin, ymax = ax.get_ylim()
        ax.vlines([0 - (0.5 / params['fs']), 1 - (0.5 / params['fs'])], ymin, ymax, colors='black', linestyles='--',
                  lw=1)  # unhard code the 1
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
    '''Plots rasters of the specified sound combos vertically above a summary PSTH, can smooth.
    Must pass response without sponts taken out.'''
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


def psth_comp(resp_idx, pair, unit, response, params, sigma=None, z=False, sum=False):
    '''Produces a single PSTH with however many lines based on the number of indices
    given in resp_idx. the rest of the parameters will get passed from raster_comp()
    parameters or could easily be manually inserted.'''
    if len(resp_idx) == 3:
        colors = ['deepskyblue','yellowgreen','dimgray']
    if len(resp_idx) == 4:
        colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray']
    if len(resp_idx) == 5:
        colors = ['deepskyblue', 'khaki', 'gold', 'lightcoral', 'firebrick']

    if z and 1 < len(resp_idx) < 4:
        fig, axes = plt.subplots(2, 1, sharex=True)
        ax = axes[0]
    else:
        fig, ax = plt.subplots()

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
    ax.vlines([0, params['Duration']], ymin, ymax, colors='black', linestyles=':')
    ax.vlines(params['SilenceOnset'], ymax * .9, ymax, colors='black', linestyles='-', lw=0.25)
    ax.set_ylim(ymin,ymax), ax.legend(loc='upper left')
    ax.set_ylabel('spk/s')

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
    ax.set_xlabel('Time (s)')
    fig.set_figheight(6), fig.set_figwidth(15)
    fig.suptitle(f"Experiment {params['experiment']} - Unit {unit} - "
                 f"Pair {pair} - BG: {params['pairs'][pair][0]} - "
                 f"FG: {params['pairs'][pair][1]} - {resp_idx}", fontweight='bold')











parmfile = '/auto/data/daq/Hood/HOD009/HOD009a09_p_OLP'
params = load_experiment_params(parmfile)
response = get_response(params)
resp_nospont = subtract_spont(response, params)
zscore, z_params = get_z([1,4,3], resp_nospont, params)
# [2,3] (hBG,fFG) - [2,7] (fBG, hFG) - [0,1,2] fBG/fFG - [1,4,3] (hBG,fFG)
z_heatmaps(zscore, params, z_params, 2)

zscore, z_params = get_z([1,4,3], nospont, params)
zscore2, z_params2 = get_z([0,1,2], nospont, params)
z_heatmaps(zinput, params, z_params, 2)
