import numpy as np
import scipy.ndimage.filters as sf
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats
import joblib as jl
import pathlib as pl

#lab comp
rec = jl.load(pl.Path('/home/hamersky/Downloads/HOD009a09_p_OLP'))

#joblib.dump()
rec = jl.load(pl.Path('/Users/grego/Downloads/HOD005b09_p_OLP'))
rec = jl.load(pl.Path('/Users/grego/Downloads/HOD009a09_p_OLP'))

expt = jl.load(pl.Path('/Users/grego/Downloads/HOD009a09_p_OLPa'))

rasterfs = 100

# parmfile = '/auto/data/daq/Hood/HOD005/HOD005b09_p_OLP'
# parmfile = '/auto/data/daq/Hood/HOD009/HOD009a09_p_OLP'
# expt = BAPHYExperiment(parmfile)
# rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
resp = rec['resp'].rasterize()
e = resp.epochs
e.loc[e.name.str.startswith('STIM'), 'name'].unique()

backgrounds = ['01Chimes','02Gravel','03Insect_Buzz','04Rain','05Rock_Tumble',
       '06Stream','07Thunder','08Waterfall','09Waves','10Wind']
foregrounds = ['01Alarm','02Chirp','03Loud_Shrill','04Phee','05Seep','06Trill',
       '07Tsik_Ek','08Tsik','09TwitterA','10TwitterB']
expts = ['HOD005','HOD006','HOD007','HOD008','HOD009']
    # Good Units
    #   HOD005: 0,
    #   HOD006:
    #   HOD007:
    #   HOD008:
    #   HOD009: 6, 10, 14
idxs = [([6,10,6,10,7,7,2,4,3,8],[2,2,8,8,1,4,5,7,10,3]),
         ([6,6,1,1,6,10,9,8,7,2],[5,8,8,5,4,10,7,1,9,2]),
         ([7,7,5,4,9,9,1,2,3,10,9,7],[2,7,2,3,7,6,5,10,1,4,2,6]),
         ([3,3,7,5,6,6,1,10],[7,9,7,2,2,8,5,10]),
         ([8,8,1,1,6,6,7,7,5,3,3],[7,2,7,2,5,8,2,7,2,7,9])]
pair_idx = []
for idx in range(len(idxs)):
    list1, list2 = [i - 1 for i in idxs[idx][0]], [j - 1 for j in idxs[idx][1]]  # -1 accounts for MATLAB indexing
    merged = list(zip(list1,list2))
    pair_idx.append(merged)

pair_names = [None] * len(pair_idx)
for expt in range(len(pair_idx)):
    pair_names[expt] = [(backgrounds[b], foregrounds[f]) for b,f in pair_idx[expt]]

Pairs = {}
for tt,ex in enumerate(expts):
    Pairs[ex] = pair_idx[tt]

pairs_named = {}
for tt,ex in enumerate(expts):
    pairs_named[ex] = pair_names[tt]


# def function that makes all the labels (experiment):
#
#
labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
          'Half BG/Half FG', 'Full BG/Half FG']

def get_pair_names(experiment, expt_pairs, BGs, FGs):
    '''Takes given experiment number and returns all the sound pair names. (Figure out
    how to not hard code for Hood).'''
    exp = f'HOD00{experiment}'
    pair_names = [(BGs[b],FGs[f]) for b,f in expt_pairs[exp]]
    return pair_names

def get_names(experiment, pair, expt_pairs, BGs, FGs):
    '''Takes given experiment number and sound pair and produces the sound name of the
    background and foreground. (Figure out how to not hardcode for Hood).'''
    index = expt_pairs[f'HOD00{experiment}'][pair]
    BG, FG = BGs[index[0]], FGs[index[1]]
    return BG, FG

def get_response(experiment, pair, unit, spont=True, expt_resp=resp, expt_pairs=Pairs, BGs=backgrounds, FGs=foregrounds):
    '''A given experiment, pair, and unit will return the 8 sound combos, labeled and in the
    repeat x neuron x time raster. Returns that as well as some basic info about the
    data to pass to other functions.
    This is a pared down version with no plotting best used for the z-scores.'''
    ids = {}
    ids['experiment'], ids['pair'] = experiment, pair
    if unit:
        ids['unit'] = unit
    BG, FG = get_names(experiment, pair, expt_pairs, BGs, FGs)
    names = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1',
             f'STIM_{BG}-0.5-1_{FG}-0-1', f'STIM_{BG}-0.5-1_null', f'STIM_null_{FG}-0.5-1',
             f'STIM_{BG}-0.5-1_{FG}-0.5-1', f'STIM_{BG}-0-1_{FG}-0.5-1']
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    resps = [expt_resp.extract_epoch(i) for i in names]  # gives you a repeat X neuron X time raster
    if spont:
        for cb in range(len(resps)):
            spont_rate = np.mean(resps[cb][:, :, :50], axis=2)[:, :, None]
            resps[cb] = resps[cb] - spont_rate
    ids['sounds'], ids['labels'] = (BG,FG), labels
    resp_names = tuple(zip(labels,resps))

    return resp_names, ids

def get_z(resp_idx, expt_resp, unit):
    '''Uses a list of two or three sound combo types and the responses to generate a
    *z-score* ready for plotting with the label of the component sounds.'''
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    if len(resp_idx) == 3:
        if (0 in resp_idx and 5 in resp_idx) or (1 in resp_idx and 4 in resp_idx):
            resp_ABfull = expt_resp[2][1][:, unit, :]
            resp_ABfull_mean = resp_ABfull.mean(axis=0)
            sem_ABfull = stats.sem(resp_ABfull, axis=0)
            resp_combo = expt_resp[resp_idx[2]][1][:, unit, :]
            mean_combo = resp_combo.mean(axis=0)
            sem_combo = stats.sem(resp_combo, axis=0)
            z_no_nan = np.nan_to_num((mean_combo - resp_ABfull_mean) / (sem_combo + sem_ABfull))
            label = f'{labels[resp_idx[-1]]} - {labels[2]}'
        else:
            respA, respB = expt_resp[resp_idx[0]][1][:, unit, :], \
                           expt_resp[resp_idx[1]][1][:, unit, :]
            trls = np.min([respA.shape[0], respB.shape[0]])
            resplin = (respA[:trls, :] + respB[:trls, :])
            mean_resplin = resplin.mean(axis=0)
            respAB = expt_resp[resp_idx[2]][1][:, unit, :]
            mean_respAB = respAB.mean(axis=0)
            semlin, semAB = stats.sem(resplin, axis=0), stats.sem(respAB, axis=0)
            z_no_nan = np.nan_to_num((mean_respAB - mean_resplin) / (semAB + semlin))
            label = f'{labels[resp_idx[-1]]} - Linear Sum'
    if len(resp_idx) == 2:
        respX, respY = expt_resp[resp_idx[0]][1][:, unit, :], \
                       expt_resp[resp_idx[1]][1][:, unit, :]
        mean_respX, mean_respY = respX.mean(axis=0), respY.mean(axis=0)
        semX, semY = stats.sem(respX, axis=0), stats.sem(respY, axis=0)
        z_no_nan = np.nan_to_num((mean_respY - mean_respX) / (semY + semX))
        label = f'{labels[resp_idx[-1]]} - {labels[resp_idx[0]]}'

    return z_no_nan, label


def plot_combos(experiment, pair, unit, expt_resp=resp, expt_pairs=Pairs, BGs=backgrounds, FGs=foregrounds):
    '''A given experiment, pair, and unit will return the 8 sound combos, labeled and in the
    repeat x neuron x time raster. Returns that as well as some basic info about the
    data to pass to other functions. This function plots the 8 different sound combos
    side by side for comparison, including with colored bars to indicate when each
    BG or FG came on and off in that instance.'''
    ids = {}
    ids['experiment'], ids['pair'], ids['unit'] = experiment, pair, unit
    BG, FG = get_names(experiment,pair,expt_pairs, BGs, FGs)
    fig, axes = plt.subplots(4,2, sharex=True, sharey=True, squeeze=True)
    axes = np.ravel(axes, order='F')
    names = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1',
             f'STIM_{BG}-0.5-1_{FG}-0-1', f'STIM_{BG}-0.5-1_null', f'STIM_null_{FG}-0.5-1',
             f'STIM_{BG}-0.5-1_{FG}-0.5-1', f'STIM_{BG}-0-1_{FG}-0.5-1']
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']

    resps = [expt_resp.extract_epoch(i) for i in names]  # gives you a repeat X neuron X time raster
    for count, (ax, file) in enumerate(zip(axes, resps)):
        ax.plot(resps[count][:, unit, :].mean(axis=0),color='black')
        ax.set_title(f'{labels[count]}',fontweight='bold')
        ax.set_xlim(0, 175)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xticklabels([0, 0.5, 1, 1.5])

    ymin,ymax = ax.get_ylim()
    rec_height = ymax * 0.2
    for cnt, ax in enumerate(axes):
        if cnt == 0:
            ax.add_patch(matplotlib.patches.Rectangle((50, ymax-rec_height), 100, rec_height,
                                                      color="blue",alpha=0.25,ec=None))
        if cnt == 1:
            ax.add_patch(matplotlib.patches.Rectangle((50, ymax-(rec_height*2)), 100, rec_height,
                                                      color="green",alpha=0.25,ec=None))
        if cnt == 2:
            ax.add_patch(matplotlib.patches.Rectangle((50, ymax-rec_height), 100, rec_height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((50, ymax - (rec_height*2)), 100, rec_height,
                                                      color="green", alpha=0.25, ec=None))
        if cnt == 3:
            ax.add_patch(matplotlib.patches.Rectangle((100, ymax-rec_height), 50, rec_height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((50, ymax-(rec_height*2)), 100, rec_height,
                                                      color="green",alpha=0.25,ec=None))
        if cnt == 4:
            ax.add_patch(matplotlib.patches.Rectangle((100, ymax-rec_height), 50, rec_height,
                                                      color="blue",alpha=0.25,ec=None))
        if cnt == 5:
            ax.add_patch(matplotlib.patches.Rectangle((100, ymax -(rec_height*2)), 50, rec_height,
                                                      color="green", alpha=0.25, ec=None))
        if cnt == 6:
            ax.add_patch(matplotlib.patches.Rectangle((100, ymax-rec_height), 50, rec_height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((100, ymax -(rec_height*2)), 50, rec_height,
                                                      color="green", alpha=0.25, ec=None))
        if cnt == 7:
            ax.add_patch(matplotlib.patches.Rectangle((50, ymax-rec_height), 100, rec_height,
                                                      color="blue",alpha=0.25,ec=None))
            ax.add_patch(matplotlib.patches.Rectangle((100, ymax -(rec_height*2)), 50, rec_height,
                                                      color="green", alpha=0.25, ec=None))

    ids['sounds'] = (BG,FG)
    resp_names = tuple(zip(names,resps))
    fig.suptitle(f'Experiment HOD00{experiment} - Unit {unit} \n'
                 f'Pair {pair} - BG: {BG} - FG: {FG}',fontweight='bold')
    fig.tight_layout()
    # fig.savefig(f'/home/hamersky/Documents/Combos/HOD00{experiment}/Unit {unit}/HOD00{experiment}'
    #             f'_Unit {unit}_Pair {pair}.png')

    return resp_names, ids


def plot_rasters(resp_idx, expt_resp=response, expt_ids=ids, fs=rasterfs):
    '''Will generate a figure of the rasters of the three specified sound combos
    vertically arranged above a summary PSTH. !Maybe add sigma option!'''
    spk_times = [np.where(expt_resp[gg][1][:, expt_ids['unit'], :]) for gg in resp_idx]
    colors = ['blue','green','black']

    labels = ['Full BG', 'Full FG', 'Full BG\n+\nFull FG', 'Half BG\n+\nFull FG', 'Half BG', 'Half FG',
              'Half BG\n+\nHalf FG', 'Full BG\n+\nHalf FG']

    fig, ax = plt.subplots(4,1)
    for ii in range(len(ax)-1):
        ax[ii].plot((spk_times[ii][1] / fs) , (spk_times[ii][0] / 1), '|', color='k', markersize=5)
        ax[ii].spines['right'].set_visible(False)
        ax[ii].spines['top'].set_visible(False)
        ax[ii].spines['bottom'].set_visible(False)
        ax[ii].set_xlim(20/fs,160/fs)
        # ax[ii].set_ylabel(response[resp_idx[ii]][0],rotation='horizontal',ha='right')
        ax[ii].set_ylabel(labels[resp_idx[ii]],rotation='horizontal',ha='right',va='center')
        ax[ii].set_xticks([])
        ax[ii].set_yticks([])
        ymin,ymax = ax[ii].get_ylim()
        ax[ii].add_patch(matplotlib.patches.Rectangle((50/fs,ymin), 100/fs, ymax,
                                                      color=colors[ii], alpha=0.2, ec=None))

    for col,idx in enumerate(resp_idx):
        ax[-1].plot(expt_resp[idx][1][:, expt_ids['unit'], :].mean(axis=0),color=colors[col])
    ax[-1].set_xlim(20,160)
    ax[-1].spines['top'].set_visible(False)
    ax[-1].spines['right'].set_visible(False)
    ax[-1].set_xticks([50,100,150])
    ax[-1].set_xticklabels([0,0.5,1.0])
    ax[-1].set_ylabel('spk/s')
    ax[-1].set_xlabel('Time (s)')
    ymin,ymax = ax[-1].get_ylim()
    ax[-1].vlines([50,150],ymin,ymax,colors='black',linestyles=':',lw=0.5)

    fig.suptitle(f"Experiment HOD00{expt_ids['experiment']} - Unit {expt_ids['unit']} - "
                 f"Pair {expt_ids['pair']}\n"
                 f"BG: {expt_ids['sounds'][0]} - FG: {expt_ids['sounds'][1]}")
    # fig.tight_layout()
    fig.set_figwidth(4.5)
    fig.set_figheight(4)


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

def psth_comp(resp_idx, expt_resp, expt_ids, sigma=None, sum=None, fs=rasterfs, ax=None):
    '''Produces a single PSTH with however many lines based on the number of indices
    given in resp_idx. the rest of the parameters will get passed from raster_comp()
    parameters or could easily be manually inserted.'''
    edit_fig = False
    if ax == None:
        fig, ax = plt.subplots()
        edit_fig = True

    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    if len(resp_idx) == 3:
        colors = ['deepskyblue','yellowgreen','dimgray']
    if len(resp_idx) == 4:
        colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'dimgray']
    if len(resp_idx) == 5:
        colors = ['deepskyblue', 'khaki', 'gold', 'lightcoral', 'firebrick']

    for col, idx in enumerate(resp_idx):
        resp = expt_resp[idx][1][:, expt_ids['unit'], :]
        mean_resp = resp.mean(axis=0)
        x = np.linspace(0,mean_resp.shape[0]/fs,mean_resp.shape[0])-0.5
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(mean_resp, sigma) * fs, color=colors[col],label=f'{labels[idx]}')
        if not sigma:
            ax.plot(x, mean_resp * fs,color=colors[col], label=f'{labels[idx]}')
        sem = stats.sem(resp, axis=0)
        ax.fill_between(x, sf.gaussian_filter1d((mean_resp - sem) * fs, sigma),
                        sf.gaussian_filter1d((mean_resp + sem) * fs, sigma), alpha=0.5, color=colors[col])

    if sum == 'linear':
        respAB = expt_resp[resp_idx[0]][1][:, expt_ids['unit'], :].mean(axis=0) +\
                 expt_resp[resp_idx[1]][1][:, expt_ids['unit'], :].mean(axis=0)
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(respAB * fs, sigma), color='dimgray', ls='--', label='Linear Sum')
        if not sigma:
            ax.plot(x, respAB * fs,color='dimgray', ls='--', label='Linear Sum')
    if sum == 'combo':
        resp_ABfull = expt_resp[2][1][:, expt_ids['unit'], :]
        resp_ABfull_mean = resp_ABfull.mean(axis=0)
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(resp_ABfull_mean * fs, sigma), color='dimgray',
                    ls='--', label=f'{labels[2]}')
        if not sigma:
            ax.plot(x, resp_ABfull_mean * fs,color='dimgray', ls='--', label=f'{labels[2]}')
        semAB = stats.sem(resp_ABfull, axis=0)
        ax.fill_between(x, sf.gaussian_filter1d((resp_ABfull_mean - semAB) * fs, sigma),
                        sf.gaussian_filter1d((resp_ABfull_mean + semAB) * fs, sigma),
                        alpha = 0.25, color='dimgray')

    ax.set_xlim(-0.3,1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ymin,ymax = ax.get_ylim()
    ax.vlines([0,1],ymin,ymax,colors='black',linestyles=':') #unhard code the 1
    ax.vlines(0.5,ymax*.9,ymax,colors='black',linestyles='-')
    ax.set_ylim(-0.01,ymax)
    if edit_fig:
        ax.legend(loc='upper left')
        ax.set_ylabel('spk/s')
        ax.set_xlabel('Time (s)')
        fig.tight_layout()
        fig.set_figheight(6)
        fig.set_figwidth(15)
        fig.suptitle(f"Experiment HOD00{expt_ids['experiment']} - Unit {expt_ids['unit']} - "
                     f"Pair {expt_ids['pair']} - BG: {expt_ids['sounds'][0]} - "
                     f"FG: {expt_ids['sounds'][1]} - {resp_idx}")


def psth_comp_z(resp_idx, sigma=None, fs=rasterfs, expt_resp=response, expt_ids=ids):
    '''Produces a single PSTH with however many lines based on the number of indices
    given in resp_idx as well as a plot of zscore line below, the rest of the parameters
    will get passed from raster_comp() parameters or could easily be manually inserted.'''
    fig, ax = plt.subplots(2,1,sharex=True)
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    colors = ['deepskyblue','yellowgreen','dimgray']

    for col, idx in enumerate(resp_idx):
        resp = expt_resp[idx][1][:, expt_ids['unit'], :]
        mean_resp = resp.mean(axis=0)
        x = np.linspace(0,mean_resp.shape[0]/fs,mean_resp.shape[0])-0.5
        if sigma:
            ax[0].plot(x, sf.gaussian_filter1d(mean_resp, sigma) * fs, color=colors[col],label=f'{labels[idx]}')
        if not sigma:
            ax[0].plot(x, mean_resp * fs,color=colors[col], label=f'{labels[idx]}')
        sem = stats.sem(resp, axis=0)
        ax[0].fill_between(x, sf.gaussian_filter1d((mean_resp - sem) * fs, sigma),
                        sf.gaussian_filter1d((mean_resp + sem) * fs, sigma), alpha=0.5, color=colors[col])

    if len(resp_idx) == 3:
        if (0 in resp_idx and 5 in resp_idx) or (1 in resp_idx and 4 in resp_idx):
            resp_ABfull = expt_resp[2][1][:, expt_ids['unit'], :]
            resp_ABfull_mean = resp_ABfull.mean(axis=0)
            if sigma:
                ax[0].plot(x, sf.gaussian_filter1d(resp_ABfull_mean * fs, sigma), color='dimgray',
                        ls='--', label=f'{labels[2]}')
            if not sigma:
                ax[0].plot(x, resp_ABfull_mean * fs, color='dimgray', ls='--', label=f'{labels[2]}')
            sem_ABfull = stats.sem(resp_ABfull, axis=0)
            ax[0].fill_between(x, sf.gaussian_filter1d((resp_ABfull_mean - sem_ABfull) * fs, sigma),
                            sf.gaussian_filter1d((resp_ABfull_mean + sem_ABfull) * fs, sigma),
                            alpha=0.25, color='dimgray')
        else:
            linear_sum = expt_resp[resp_idx[0]][1][:, expt_ids['unit'], :].mean(axis=0) +\
                     expt_resp[resp_idx[1]][1][:, expt_ids['unit'], :].mean(axis=0)
            if sigma:
                ax[0].plot(x, sf.gaussian_filter1d(linear_sum * fs, sigma), color='dimgray', ls='--', label='Linear Sum')
            if not sigma:
                ax[0].plot(x, linear_sum * fs,color='dimgray', ls='--', label='Linear Sum')

    ax[0].legend(loc='upper left')
    ax[0].set_xlim(-0.3,1.2)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ymin,ymax = ax[0].get_ylim()
    ax[0].set_ylim(-0.01,ymax)
    ax[0].set_ylabel('spk/s')
    ax[0].vlines([0,1],ymin,ymax,colors='black',linestyles=':') #unhard code the 1
    ax.vlines(0.5,ymax*.9,ymax,colors='black',linestyles='-')
    fig.tight_layout()
    fig.set_figheight(6)
    fig.set_figwidth(15)
    fig.suptitle(f"Experiment HOD00{expt_ids['experiment']} - Unit {expt_ids['unit']} - "
                 f"Pair {expt_ids['pair']} - BG: {expt_ids['sounds'][0]} - "
                 f"FG: {expt_ids['sounds'][1]} - {resp_idx}")
    #z part
    if len(resp_idx) == 3:
        if (0 in resp_idx and 5 in resp_idx) or (1 in resp_idx and 4 in resp_idx):
            resp_combo = expt_resp[resp_idx[2]][1][:, expt_ids['unit'], :]
            mean_combo = resp_combo.mean(axis=0)
            sem_combo = stats.sem(resp_combo, axis=0)
            z_no_nan = np.nan_to_num((resp_ABfull_mean - mean_combo) / (sem_ABfull + sem_combo))
            label = f'{labels[2]} - {labels[resp_idx[-1]]}'
        else:
            respA, respB = expt_resp[resp_idx[0]][1][:, expt_ids['unit'], :], \
                           expt_resp[resp_idx[1]][1][:, expt_ids['unit'], :]
            trls = np.min([respA.shape[0], respB.shape[0]])
            resplin = (respA[:trls, :] + respB[:trls, :])
            mean_resplin = resplin.mean(axis=0)
            respAB = response[resp_idx[2]][1][:, expt_ids['unit'], :]
            mean_respAB = respAB.mean(axis=0)
            semlin, semAB = stats.sem(resplin, axis=0), stats.sem(respAB, axis=0)
            z_no_nan = np.nan_to_num((mean_resplin - mean_respAB) / (semlin + semAB))
            label = f'Linear Sum - {labels[resp_idx[-1]]}'
    if len(resp_idx) == 2:
        respX, respY = expt_resp[resp_idx[0]][1][:, expt_ids['unit'], :], \
                       expt_resp[resp_idx[1]][1][:, expt_ids['unit'], :]
        mean_respX, mean_respY = respX.mean(axis=0), respY.mean(axis=0)
        semX, semY = stats.sem(respX, axis=0), stats.sem(respY, axis=0)
        z_no_nan = np.nan_to_num((mean_respX - mean_respY) / (semX + semY))
        label = f'{labels[resp_idx[0]]} - {labels[resp_idx[-1]]}'

    if sigma:
        ax[1].plot(x, sf.gaussian_filter1d(z_no_nan, sigma), color='black', label=label)
    if not sigma:
        ax[1].plot(x, z_no_nan, color='black', label=label)

    ax[1].hlines([0], x[0], x[-1], colors='black', linestyles=':', lw=0.5)
    ax[1].set_xlim(-0.3, 1.2)
    ymin, ymax = ax[1].get_ylim()
    ax[1].set_ylim(ymin,ymax)
    ax[1].vlines([0, 1], ymin, ymax, colors='black', linestyles=':', lw=0.5)  # unhard code the 1
    ax.vlines(0.5,ymax*.9,ymax,colors='black',linestyles='-', lw=0.25)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend(loc='upper left')
    ax[1].set_xlabel('Time (s)')

    return z_no_nan




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



response, ids = plot_combos(9,0,6)
resp_idx = [0,1,2]


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


def z_heatmaps(experiment, resp_idx, sigma=None,
              expt_resp=resp, expt_pairs=Pairs, BGs=backgrounds, FGs=foregrounds, fs=rasterfs):
    expt = f'HOD00{experiment}'
    units = len(expt_resp.chans)

    zs, z_labels, resps, tags, all_list = {}, {}, {}, {}, []
    tags['experiment'], tags['idx'] = experiment, resp_idx
    for pair in range(len(expt_pairs[expt])):
        response, ids = get_response(experiment, pair, None, False, expt_resp, expt_pairs, BGs, FGs)
        z_list, mean_response = [], []
        # if spont:
        #     for cb in range(len(response)):
        #         mean_resp = response[cb].mean(axis=0)
        #         spont_rate = np.mean(mean_resp[:, :50], axis=1)[:, None]
        #         mean_response.append(mean_resp - spont_rate)

        for unt in range(units):
            zscore, label = get_z(resp_idx, response, unt)
            z_list.append(zscore)
            zscores = np.stack(z_list, axis=0)    #unitx time array
            z_labels[ids['pair']], zs[ids['pair']], resps[ids['pair']] = ids['sounds'], zscores, response
        all_list.append(zscores)
    all_zs = np.stack(all_list, axis=2)
    if sigma is not None:
        smooth_zs = sf.gaussian_filter1d(all_zs, sigma, axis=1)
        zmin, zmax = np.min(np.min(smooth_zs, axis=1)), np.max(np.max(smooth_zs, axis=1))
        abs_max = max(abs(zmin),zmax)
    else:
        zmin, zmax = np.min(np.min(all_zs, axis=1)), np.max(np.max(all_zs, axis=1))
        abs_max = max(abs(zmin),zmax)

    combo_labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
                    'Half BG/Half FG', 'Full BG/Half FG']
    tags['idx_names'] = [combo_labels[i] for i in resp_idx]
    tags['legend'] = label

    if sigma is not None:
        for pr in range(len(resps)):
            zs[pr] = sf.gaussian_filter1d(zs[pr], sigma, axis=1)


    disp_pairs = len(resps)
    fig, axes = plt.subplots(int(np.round(disp_pairs/2)), 2)
    axes = np.ravel(axes, order='F')
    if int(len(resps) / 2) % 2 != 0:
        axes[-1].spines['top'].set_visible(False)
        axes[-1].spines['bottom'].set_visible(False)
        axes[-1].spines['right'].set_visible(False)
        axes[-1].spines['left'].set_visible(False)
        axes[-1].set_yticks([])
        axes[-1].set_xticks([])
        axes = axes[:-1]

    for cnt, ax in enumerate(axes):
        im = ax.imshow(zs[cnt], aspect='auto', cmap='RdYlBu_r',
                  extent=[-0.5, (zs[cnt].shape[1] / fs) -
                          0.5, zs[cnt].shape[0], 0], vmin=-abs_max, vmax=abs_max)
        ax.set_title(f"Pair {cnt}: BG {z_labels[cnt][0]} - FG {z_labels[cnt][1]}",
                     fontweight='bold')
        ymin, ymax = ax.get_ylim()
        ax.vlines([0 - (0.5 / fs), 1 - (0.5 / fs)], ymin, ymax, colors='white', linestyles='--',
                  lw=1)  # unhard code the 1
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(xmin + 0.3, xmax - 0.2)
        if cnt == int(np.around(len(resps) / 2) - 1) or cnt == int(len(axes) - 1):
            ax.set_xticks([0, 0.5, 1.0])
        else:
            ax.set_xticks([])

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    fig.text(0.5, 0.07, 'Time from onset (s)', ha='center', va='center', fontweight='bold')
    fig.text(0.1, 0.5, 'Neurons', ha='center', va='center', rotation='vertical', fontweight='bold')
    fig.suptitle(f"Experiment HOD00{tags['experiment']} - Combo Index {tags['idx']} - "
                 f"{tags['idx_names']} - Sigma {sigma}\n"
                 f"{tags['legend']}", fontweight='bold')


##Doing life better

def load_experiment_params(parmfile):
    """Given a parm file, or if I'm on my laptop, a saved experiment file, it will load the file
    and get relevant parameters about the experiment as well as sort out the sound indexes."""
    rasterfs = 100
    params = {}
    if parmfile.split('/')[1] == 'auto':
        expt = BAPHYExperiment(parmfile)
    else:
        expt = jl.load(pl.Path('/Users/grego/Downloads/HOD009a09_p_OLPa'))
    rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
    resp = rec['resp'].rasterize()
    e = resp.epochs
    params['animal'], params['experiment'] = parmfile.split('/')[-3], parmfile.split('/')[-2]
    params['fs'] = resp.fs
    params['max reps'] = e[e.name.str.startswith('STIM')].pivot_table(index=['name'], aggfunc='size').max()
    params['stim length'] = int(e.loc[e.name.str.startswith('REF')].iloc[0]['end']
                - e.loc[e.name.str.startswith('REF')].iloc[0]['start'])
    params['combos'] = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
          'Half BG/Half FG', 'Full BG/Half FG']

    expt_params = expt.get_baphy_exptparams()   #Using Charlie's manager
    soundies = list(expt_params[0]['TrialObject'][1]['ReferenceHandle'][1]['SoundPairs'].values())
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






def get_z_nomean(resp_idx, expt_resp, unit):
    '''Uses a list of two or three sound combo types and the responses to generate a
    *z-score* ready for plotting with the label of the component sounds.'''
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    if len(resp_idx) == 3:
        if (0 in resp_idx and 5 in resp_idx) or (1 in resp_idx and 4 in resp_idx):
            resp_ABfull = expt_resp[2][1][:, unit, :]
            resp_ABfull_mean = resp_ABfull.mean(axis=0)
            sem_ABfull = stats.sem(resp_ABfull, axis=0)
            resp_combo = expt_resp[resp_idx[2]][1][:, unit, :]
            mean_combo = resp_combo.mean(axis=0)
            sem_combo = stats.sem(resp_combo, axis=0)
            z_no_nan = np.nan_to_num((mean_combo - resp_ABfull_mean) / (sem_combo + sem_ABfull))
            label = f'{labels[resp_idx[-1]]} - {labels[2]}'
        else:
            respA, respB = expt_resp[resp_idx[0]][1][:, unit, :], \
                           expt_resp[resp_idx[1]][1][:, unit, :]
            trls = np.min([respA.shape[0], respB.shape[0]])
            resplin = (respA[:trls, :] + respB[:trls, :])
            mean_resplin = resplin.mean(axis=0)
            respAB = expt_resp[resp_idx[2]][1][:, unit, :]
            mean_respAB = respAB.mean(axis=0)
            semlin, semAB = stats.sem(resplin, axis=0), stats.sem(respAB, axis=0)
            z_no_nan = np.nan_to_num((mean_respAB - mean_resplin) / (semAB + semlin))
            label = f'{labels[resp_idx[-1]]} - Linear Sum'
    if len(resp_idx) == 2:
        respX, respY = expt_resp[resp_idx[0]][1][:, unit, :], \
                       expt_resp[resp_idx[1]][1][:, unit, :]
        mean_respX, mean_respY = respX.mean(axis=0), respY.mean(axis=0)
        semX, semY = stats.sem(respX, axis=0), stats.sem(respY, axis=0)
        z_no_nan = np.nan_to_num((mean_respY - mean_respX) / (semY + semX))
        label = f'{labels[resp_idx[-1]]} - {labels[resp_idx[0]]}'

    return z_no_nan, label