import numpy as np
import scipy.ndimage.filters as sf
from nems_lbhb.baphy_experiment import BAPHYExperiment
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

rasterfs = 100

parmfile = '/auto/data/daq/Hood/HOD005/HOD005b09_p_OLP'
parmfile = '/auto/data/daq/Hood/HOD009/HOD009a09_p_OLP'
expt = BAPHYExperiment(parmfile)
rec = expt.get_recording(rasterfs=rasterfs, resp=True, stim=False)
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
    list1, list2 = [i - 1 for i in idxs[idx][0]], [j - 1 for j in idxs[idx][1]]
    merged = list(zip(list1,list2))
    pair_idx.append(merged)

Pairs = {}
for tt,ex in enumerate(expts):
    Pairs[ex] = pair_idx[tt]

def get_pair_names(experiment,Pairs=Pairs,backgrounds=backgrounds,foregrounds=foregrounds):
    exp = f'HOD00{experiment}'
    pair_names = [(backgrounds[b],foregrounds[f]) for b,f in Pairs[exp]]
    return pair_names

def get_names(experiment,pair,Pairs=Pairs,backgrounds=backgrounds,foregrounds=foregrounds):
    index = Pairs[f'HOD00{experiment}'][pair]
    BG, FG = backgrounds[index[0]], foregrounds[index[1]]
    return BG, FG

def plot_combos(experiment,pair,unit,resp=resp,Pairs=Pairs,backgrounds=backgrounds,foregrounds=foregrounds):
    ids = {}
    ids['experiment'], ids['pair'], ids['unit'] = experiment, pair, unit
    BG, FG = get_names(experiment,pair)
    fig,axes = plt.subplots(4,2, sharex=True, sharey=True, squeeze=True)
    axes = np.ravel(axes, order='F')
    names = [f'STIM_{BG}-0-1_null', f'STIM_null_{FG}-0-1', f'STIM_{BG}-0-1_{FG}-0-1',
             f'STIM_{BG}-0.5-1_{FG}-0-1', f'STIM_{BG}-0.5-1_null', f'STIM_null_{FG}-0.5-1',
             f'STIM_{BG}-0.5-1_{FG}-0.5-1', f'STIM_{BG}-0-1_{FG}-0.5-1']
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']

    resps = [resp.extract_epoch(i) for i in names]  # gives you a repeat X neuron X time raster
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

    fig.savefig(f'/home/hamersky/Documents/Combos/HOD00{experiment}/Unit {unit}/HOD00{experiment}'
                f'_Unit {unit}_Pair {pair}.png')

    return resp_names, ids

response, ids = plot_combos(9,0,6)
resp_idx = [0,1,2]
# plot_comp(resp_idx)


def plot_comp(resp_idx, fs=1000, response=response,ids=ids):
    spk_times = [np.where(response[gg][1][:,ids['unit'],:]) for gg in resp_idx]
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
        ax[-1].plot(response[idx][1][:,ids['unit'],:].mean(axis=0),color=colors[col])
    ax[-1].set_xlim(20,160)
    ax[-1].spines['top'].set_visible(False)
    ax[-1].spines['right'].set_visible(False)
    ax[-1].set_xticks([50,100,150])
    ax[-1].set_xticklabels([0,0.5,1.0])
    ax[-1].set_ylabel('spk/s')
    ax[-1].set_xlabel('Time (s)')
    ymin,ymax = ax[-1].get_ylim()
    ax[-1].vlines([50,150],ymin,ymax,colors='black',linestyles=':',lw=0.5)

    fig.suptitle(f"Experiment HOD00{ids['experiment']} - Unit {ids['unit']} - Pair {ids['pair']}\n"
                 f"BG: {ids['sounds'][0]} - FG: {ids['sounds'][1]}")
    # fig.tight_layout()
    fig.set_figwidth(4.5)
    fig.set_figheight(4)


def raster_comp(resp_idx,sum=False,sigma=None,fs=rasterfs,response=response,ids=ids):
    fig, ax = plt.subplots()
    labels = ['Full BG', 'Full FG', 'Full BG/Full FG', 'Half BG/Full FG', 'Half BG', 'Half FG',
              'Half BG/Half FG', 'Full BG/Half FG']
    if len(resp_idx) == 3:
        colors = ['deepskyblue','yellowgreen','dimgray']
    if len(resp_idx) == 4:
        colors = ['deepskyblue', 'yellowgreen', 'lightcoral', 'peachpuff','lightseagreen']
    if len(resp_idx) == 5:
        colors = ['deepskyblue', 'khaki', 'gold', 'lightcoral', 'firebrick']

    for col, idx in enumerate(resp_idx):
        mean_resp = response[idx][1][:,ids['unit'],:].mean(axis=0)
        x = np.linspace(0,mean_resp.shape[0]/rasterfs,mean_resp.shape[0])-0.5

        if sigma:
            ax.plot(x, sf.gaussian_filter1d(mean_resp, sigma) * fs, color=colors[col],label=f'{labels[idx]}')
        if not sigma:
            ax.plot(x, mean_resp * fs,color=colors[col], label=f'{labels[idx]}')
        sem = stats.sem(mean_resp, axis=0)
        ax.fill_between(x, sf.gaussian_filter1d((mean_resp - sem) * fs, sigma),
                        sf.gaussian_filter1d((mean_resp + sem) * fs, sigma), alpha=0.5, color=colors[col])

    if sum:
        respA = response[resp_idx[0]][1][:,ids['unit'],:].mean(axis=0)
        respB = response[resp_idx[1]][1][:,ids['unit'],:].mean(axis=0)
        respAB = respA + respB
        if sigma:
            ax.plot(x, sf.gaussian_filter1d(respAB * fs, sigma), color='dimgray', ls='--', label='Linear Sum')
        if not sigma:
            ax.plot(x, respAB * fs,color='dimgray', ls='--', label='Linear Sum')

    ax.legend(loc='upper left')
    ax.set_xlim(-0.3,1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xticks([50,100,150])
    # ax.set_xticklabels([0,0.5,1.0])
    ymin,ymax = ax.get_ylim()
    ax.set_ylim(-0.01,ymax)
    ax.set_ylabel('spk/s')
    ax.set_xlabel('Time (s)')
    ax.vlines([0,1],ymin,ymax,colors='black',linestyles=':') #unhard code the 1
    fig.tight_layout()
    fig.set_figheight(6)
    fig.set_figwidth(15)
    fig.suptitle(f"Experiment HOD00{ids['experiment']} - Unit {ids['unit']} - Pair {ids['pair']} "
                 f"- BG: {ids['sounds'][0]} - FG: {ids['sounds'][1]} - {resp_idx}")


def zscore(array, indie_axis):
    """
    Calculate the z score of each value in the sample, relative to the sample mean and standard deviation, along the
    specified axis
    :param array: ndarray
    :param indie_axis: int, [int,...]. Axis over which to perform the zscore independently, e.g. Cell axis
    :return: z-scored ndarray
    """
    # sanitize the indie_axis valu, it can be either an integer or a list of integers
    if isinstance(indie_axis, int):
        indie_axis = [indie_axis]
    elif isinstance(indie_axis, (list, tuple, set)):
        if all(isinstance(x, int) for x in indie_axis):
            indie_axis = list(indie_axis)
        else:
            raise ValueError('all values in indie_axis must be int')
    elif indie_axis is None:
        indie_axis = []
    else:
        raise ValueError('indie_axis must be an int or a list of ints')
    # reorder axis, first: indie_axis second: shuffle_axis, third: all other axis i.e. protected axis.
    zscore_axis = [x for x in range(array.ndim) if x not in indie_axis]
    new_order = indie_axis + zscore_axis
    array = np.transpose(array, new_order)
    # if multiple axes are being zscored together, reshapes  collapsing across the zscored axis
    # shape of independent chunks of the array, i, o , independent, zscore.
    shape = array.shape
    i_shape = shape[0:len(indie_axis)]
    z_shape = (np.prod(shape[len(indie_axis):], dtype=int),)
    new_shape = i_shape + z_shape
    array = np.reshape(array, new_shape)
    # calcualtes the zscore
    means = np.mean(array, axis=-1)[:, None]
    stds = np.std(array, axis=-1)[:, None]
    zscore = np.nan_to_num((array - means) / stds)
    # reshapes into original dimensions
    zscore = np.reshape(zscore, shape)
    # swap the axis back into original positions
    zscore = np.transpose(zscore, np.argsort(new_order))
    return zscore

mean = np.mean(raster, axis=(0,1,2,4))[None, None, None, :, None]
std = np.std(raster, axis=(0,1,2,4))[None, None, None, :, None]
zscore= np.nan_to_num((raster - mean) / std)