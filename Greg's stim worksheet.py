import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from pydub import AudioSegment
import matplotlib.gridspec as gridspec

from scipy import stats

from nems.analysis.gammatone.gtgram import gtgram
# from nems import recording, signal
# from nems import xforms, get_setting
# import nems.gui.editors as gui
# import nems
# import random
# import nems.tf.cnnlink_new as cnn
# import scipy.stats as stats
# import scipy.stats

factor = 441

kind = 'Textures'
kind = 'Marms'

name = '00cat516_rec1_stream_excerpt1'                                        #50000
name = 'cat287_rec1_rain_excerpt1'                                            #66100
name = 'cat312_rec1_wind_excerpt1'                                            #105000
name = 'cat558_rec1_tire_rolling_on_gravel_excerpt1'                          #22000
name = 'cat301_rec1_envsounds_rock-tumbling_sound-ideas-38-27_2sec_excerpt1'  #99000
name = 'cat368_rec1_thunder_excerpt1'                                         #4000
name = 'cat403_rec1_waves_excerpt1'                                           #55000
name = 'cat565_rec1_insects_buzzing_excerpt1'                                 #132000
name = '00cat78_rec1_chimes_in_the_wind_excerpt1'                             #88200
name = 'cat534_rec1_waterfall_excerpt1'                                       #121200

name = 'cat669_rec1_marmoset_alarm_1_excerpt2'                                #pydub 1140-1640, 1910-2410
name = 'cat669_rec2_marmoset_chirp_excerpt2'                                  #pydub 2560-3560
name = 'cat669_rec9_marmoset_tsik_excerpt2'                                   #pydub 3700-4000, 610-1310
name = 'cat669_rec10_marmoset_tsik_ek_excerpt2'                               #pydub 150-550,950-1450
name = 'cat669_rec7_marmoset_seep_excerpt2'                                   #pydub 650-1650
name = 'cat669_rec3_marmoset_loud_shrill_excerpt1'                            #pydub 2600-3600
name = 'cat669_rec8_marmoset_trill_excerpt1'                                  #pydub 0-500, 2800-3300
name = 'cat669_rec11_marmoset_twitter_excerpt3'                               #70000
name = 'cat669_rec11_marmoset_twitter_excerpt1'                               #86500
name = 'cat669_rec6_marmoset_phee_4_excerpt4'                                 #78500



def multi_chop(name, kind, start, s1len, start2, s2len, little_name):
    GREG_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"
    filepath = GREG_ROOT + name + '.wav'

    sound = AudioSegment.from_file(filepath)
    cut1 = sound[start:start+s1len]
    cut2 = sound[start2:start2+s2len]
    # silence = AudioSegment.silent(duration=200)

    one_sec = cut1 + cut2
    SAVE_PATH = f'/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/{kind}/{little_name}_{start}-{start+s1len}_{start2}-{start2+s2len}.wav'
    one_sec.export(SAVE_PATH, format='wav')

    fs, W = wavfile.read(SAVE_PATH)
    spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(f'{name}, length: {len(one_sec)/fs})')


def save_chop(name, kind, start, little_name):
    GREG_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"
    filepath = GREG_ROOT + name + '.wav'

    fs, W = wavfile.read(filepath)

    one_sec = W[start:start+int((len(W)/4))]

    spec = gtgram(one_sec, fs, 0.02, 0.01, 48, 100, 8000)

    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(f'{name}, length: {len(one_sec)/fs})')

    SAVE_PATH = f'/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/{kind}/{little_name}_{start}.wav'

    wavfile.write(SAVE_PATH,fs,one_sec)

def sound_chop(name, kind, start):
    GREG_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"
    filepath = GREG_ROOT + name + '.wav'

    fs, W = wavfile.read(filepath)

    one_sec = W[start:start+int((len(W)/4))]

    spec = gtgram(one_sec, fs, 0.02, 0.01, 48, 100, 8000)

    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(f'{name}, length: {len(one_sec)/fs})')

    # SAVE_PATH = f'/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/{kind}/{little_name}_{start}.wav'
    #
    # wavfile.write(SAVE_PATH,fs,one_sec)

#wavfile.write(filepath,fs,file)
#wip stuff
name = 'Fight Squeak-0-1'
name = 'Waterfall-0-1'
name = 'Insect Buzz-0-1'
name = 'Waterfall-0-1_Fight Squeak-0-1'
name = 'Insect Buzz-0-1_Fight Squeak-0-1'

def spectro(name, kind):
    GREG_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/WIP Combos/"
    filepath = GREG_ROOT + name + '.wav'

    fs, W = wavfile.read(filepath)

    spec = gtgram(W, fs, 0.02, 0.01, 128, 100, 8000)

    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(name)


TEXTURE_ROOT = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Textures"
MARM_ROOT = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Marms/"
FERRET_ROOT = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Ferrets1/"
ROOTS = ["/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Textures",
         "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Ferrets/"]
chans = 12

def spec_stats(ROOT, idx, chans=48, lfreq=100, hfreq=8000):
    '''makes the three panel of a spectrogram, power spectra, and
    std over time at each freq band, for WIP'''
    dir = os.path.join(ROOT, "*.wav")
    set1 = glob.glob(dir)
    sound = set1[idx]
    fig, axes = plt.subplots(2, 2, figsize=(10,9))
    ax = np.ravel(axes, order='F')
    fs, W = wavfile.read(sound)
    spec = gtgram(W, fs, 0.02, 0.01, chans, lfreq, hfreq)
    basename = os.path.splitext(os.path.basename(sound))[0].split('_')[1]

    ax[0].imshow(spec,aspect='auto',origin='lower')
    ymin, ymax = ax[0].get_ylim()
    xmin, xmax = ax[0].get_xlim()
    ax[0].set_yticks([ymin,ymax]), ax[0].set_xticks([xmin,xmax])
    ax[0].set_yticklabels([lfreq,hfreq]), ax[0].set_xticklabels([0,1])
    ax[0].set_xlabel('Time(s)', fontweight='bold', size=15)
    ax[0].set_ylabel('Frequency (Hz)', fontweight='bold', size=15)
    ax[0].set_title(f'{basename}', fontweight='bold', size=20)

    ax[1].set_yticks([]), ax[1].set_xticks([])
    ax[1].spines['top'].set_visible(False), ax[1].spines['bottom'].set_visible(False)
    ax[1].spines['right'].set_visible(False), ax[1].spines['left'].set_visible(False)

    pow = np.average(spec, axis=1)
    y = np.arange(0, pow.shape[0])
    ax[2].plot(pow, y)
    ymin, ymax = ax[2].get_ylim()
    ax[2].set_yticks([ymin, ymax])
    ax[2].set_yticklabels([lfreq, hfreq])
    ax[2].set_xlabel('Signal Strength', fontweight='bold', size=15)
    ax[2].set_ylabel('Frequency (Hz)', fontweight='bold', size=15)

    spec_std = gtgram(W, fs, 0.02, 0.01, int(chans/4), lfreq, hfreq)
    std_pow = np.std(spec_std, axis=1)
    y = np.arange(0,std_pow.shape[0])
    ax[3].plot(std_pow, y)
    ymin,ymax = ax[3].get_ylim()
    ax[3].set_yticks([ymin, ymax])
    ax[3].set_yticklabels([lfreq, hfreq])
    ax[3].set_xlabel('STD', fontweight='bold', size=15)
    ax[3].set_ylabel('Frequency (Hz)', fontweight='bold', size=15)

    fig.tight_layout()


def spec_collage(ROOT):
    dir = os.path.join(ROOT, "*.wav")
    set1 = glob.glob(dir)
    fig, axes = plt.subplots(5,2, sharey=False, figsize=(6,8))
    axes = np.ravel(axes, order='F')

    for cnt, (ax, ss) in enumerate(zip(axes, set1)):
        fs, W = wavfile.read(ss)
        spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)
        basename = os.path.splitext(os.path.basename(ss))[0].split('_')[1]
        ax.set_title(basename)
        ax.imshow(spec,aspect='auto',origin='lower')
        ax.set_xticks([])
        ax.set_xticklabels([])


    fig.suptitle(f"Ferret Vocalizations - 'Foregrounds'",
                 fontweight='bold', size=15)
    # fig.suptitle(f"Environment Textures - 'Backgrounds'",
    #              fontweight='bold', size=15)

    fig.text(0.5, 0.02, 'Time (s)', ha='center', va='center',
             fontweight='bold', size=15)
    fig.text(0.05, 0.5, 'Frequency (Hz)', ha='center', va='center',
             rotation='vertical', fontweight='bold', size=15)
    fig.set_figheight(12)
    fig.set_figwidth(6)


def spec_collage_power(ROOT):
    dir = os.path.join(ROOT, "*.wav")
    set1 = glob.glob(dir)
    fig, ax = plt.subplots(5,4, sharey=False)
    pow_sum = []

    for nn, ss in enumerate(set1):
        fs, W = wavfile.read(ss)
        spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)
        pow = np.average(spec,axis=1)
        y = np.arange(0, pow.shape[0])
        basename = os.path.splitext(os.path.basename(ss))[0].split('_')[1]
        if nn < int(len(set1)/2):
            ax[nn,0].set_title(basename)
            ax[nn,0].imshow(spec,aspect='auto',origin='lower')
            ax[nn,0].set_xticks([])
            ax[nn,0].set_xticklabels([])
            ax[nn,1].plot(pow,y)
            ax[nn,1].set_yticklabels([])
            ax[nn,1].set_yticks([])
            ax[nn,1].set_xticks([])
        else:
            ax[nn-5,2].set_title(basename)
            ax[nn-5,2].imshow(spec,aspect='auto',origin='lower')
            ax[nn-5,2].set_xticks([])
            ax[nn-5,2].set_xticklabels([])
            ax[nn-5,3].plot(pow, y)
            ax[nn-5,3].set_yticklabels([])
            ax[nn-5,3].set_yticks([])
            ax[nn-5,3].set_xticks([])
        pow_sum.append(pow)

    # fig.suptitle(f"Ferret Vocalizations - 'Foregrounds'", fontweight='bold')
    fig.suptitle(f"Environment Textures - 'Backgrounds'", fontweight='bold')

    fig.text(0.05, 0.5, 'Frequency (Hz)', ha='center', va='center',
             rotation='vertical', fontweight='bold')
    fig.set_figheight(12)
    fig.set_figwidth(10)

    return pow_sum

ROOTS = ["/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Textures",
         "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Ferrets/"]

def time_metrics(ROOTS, chans=12, lfreq=100, hfreq=8000):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12,5))
    axes = np.ravel(axes, order='F')
    files = []
    for root in ROOTS:
        set = glob.glob(os.path.join(root, "*.wav"))
        files.append(len(set))
    longest = np.max(files)
    std_array = np.zeros([chans, longest, len(ROOTS)])

    all_names = []
    for count, (ax, RT) in enumerate(zip(axes,ROOTS)):
        dir = os.path.join(RT, "*.wav")
        set1 = glob.glob(dir)
        if len(set1) != std_array.shape[1]:
            pad = std_array.shape[1] - len(set1)
            std_array[:, -pad:, count] = np.NaN

        names = []
        for nn, ss in enumerate(set1):
            fs, W = wavfile.read(ss)
            spec = gtgram(W, fs, 0.02, 0.01, chans, lfreq, hfreq)
            x = np.linspace(0,chans-1,chans)
            basename = os.path.splitext(os.path.basename(ss))[0].split('_')[1]

            std_pow = np.std(spec, axis=1)
            std_array[:, nn, count] = std_pow
            names.append(basename), all_names.append(basename)

        for cnt in range(len(set1)):
            ax.plot(x, std_array[:,cnt,count], linestyle='-', label=names[cnt])
        leg = ax.legend(handlelength=0, frameon=False)
        for line,text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        ax.set_xticks([0,chans-1])
        ax.set_xticklabels([lfreq,hfreq])
        if count == 0:
            ax.set_title('Backgrounds', fontweight='bold', size=16)
        if count == 1:
            ax.set_title('Foregrounds', fontweight='bold', size=16)

    fig.text(0.5, 0.02, 'Frequency (Hz)', ha='center', va='center',
             fontweight='bold', size=15)
    fig.text(0.07, 0.5, 'STD', ha='center', va='center',
             rotation='vertical', fontweight='bold', size=15)

    return std_array, all_names

#This part makes the summary plot of BG and FG in time
    std_mean = np.nanmean(std_array, axis=0)
    std_sem = stats.sem(std_array, nan_policy='omit', axis=0).data
    sem_rav_mask = std_sem.ravel('F') > 0
    sem_rav = std_sem.ravel('F')[sem_rav_mask]
    rav = std_mean.ravel('F')
    nan_mask = ~np.isnan(rav)
    rav = rav[nan_mask]
    vals = sum(~np.isnan(rav))
    x = np.linspace(0, vals-1, vals)
    fig, ax = plt.subplots(figsize=(6,8))

    ax.bar(x[:10], rav[:10], yerr=sem_rav[:10], color='deepskyblue')
    ax.bar(x[10:], rav[10:], yerr=sem_rav[10:], color='yellowgreen')
    ymin,ymax = ax.get_ylim()
    ax.axvline(9.5,ymin,ymax, linestyle=':', color='black')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=90)
    ax.set_ylabel('Non-stationariness', size=15, fontweight='bold')
    fig.tight_layout()

    kind_mean = np.nanmean(std_mean, axis=0)
    kind_std = np.nanstd(std_mean, axis=0)
    arra = np.asarray([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1])
    data = np.stack((arra,rav), axis=1)
    ax[1].scatter(arra, rav)
    ax[1].set_xlim(-0.5,1.5)
    ax[1].set_xticks([0,1])
    ax[1].set_xticklabels(['BGs','FGs'])


def freq_metrics(ROOTS, chans=48, lfreq=100, hfreq=8000):
    fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
    axes = np.ravel(axes, order='F')
    files = []
    for root in ROOTS:
        set = glob.glob(os.path.join(root, "*.wav"))
        files.append(len(set))
    longest = np.max(files)
    avg_array = np.zeros([chans, longest, len(ROOTS)])

    all_names = []
    for count, (ax, RT) in enumerate(zip(axes, ROOTS)):
        dir = os.path.join(RT, "*.wav")
        set1 = glob.glob(dir)
        if len(set1) != avg_array.shape[1]:
            pad = avg_array.shape[1] - len(set1)
            avg_array[:, -pad:, count] = np.NaN

        names = []
        for nn, ss in enumerate(set1):
            fs, W = wavfile.read(ss)
            spec = gtgram(W, fs, 0.02, 0.01, chans, lfreq, hfreq)
            x = np.linspace(0, chans - 1, chans)
            basename = os.path.splitext(os.path.basename(ss))[0].split('_')[1]

            pow = np.average(spec, axis=1)
            avg_array[:, nn, count] = pow
            names.append(basename), all_names.append(basename)

        for cnt in range(len(set1)):
            ax.plot(x, avg_array[:, cnt, count], linestyle='-', label=names[cnt])
        leg = ax.legend(handlelength=0, frameon=False)
        for line, text in zip(leg.get_lines(), leg.get_texts()):
            text.set_color(line.get_color())
        ax.set_xticks([0, chans - 1])
        ax.set_xticklabels([lfreq, hfreq])
        if count == 0:
            ax.set_title('Backgrounds', fontweight='bold', size=16)
        if count == 1:
            ax.set_title('Foregrounds', fontweight='bold', size=16)

    fig.text(0.5, 0.02, 'Frequency (Hz)', ha='center', va='center',
             fontweight='bold', size=15)
    fig.text(0.07, 0.5, 'Average Strength', ha='center', va='center',
             rotation='vertical', fontweight='bold', size=15)

    return avg_array, all_names

#This part makes the summary plot of BG and FG in time
    avg_mean_idx = np.argmax(avg_array, axis=0)
    x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=chans*2, base=2)
    cf_idx = list(avg_mean_idx.ravel('F'))
    cf_idx = [i for i in cf_idx if i != 0]
    cfs = np.asarray([x_freq[i] for i in cf_idx])
    xplot = np.linspace(0, len(cf_idx)-1, len(cf_idx))
###^^That's for max freq....
    x_freq = np.logspace(np.log2(lfreq), np.log2(hfreq), num=chans*2, base=2)

    reshaped = np.swapaxes(np.reshape(avg_array, (48, 20), 'F'), 0, 1)
    x = np.linspace(0, reshaped.shape[1] - 1, reshaped.shape[1])
    center = np.sum(np.abs(reshaped) * x, axis=1) / np.sum(np.abs(reshaped), axis=1)
    cf_idx = list(np.round(center * 2))
    cf_nums = sum(~np.isnan(cf_idx))
    cfs = np.asarray([x_freq[int(i)] for i in cf_idx[:cf_nums]])

    xplot = np.linspace(0, len(cfs)-1, len(cfs))

    fig, ax = plt.subplots(figsize=(6,8))
    freq_mask = x_freq > np.max(cfs)
    x_freq[freq_mask] = np.NaN
    nonan = sum(~np.isnan(x_freq))
    listy = list(x_freq)
    new_x_freq = np.asarray(listy[:nonan])

    ax.bar(xplot[:10], cfs[:10], color='deepskyblue')
    ax.bar(xplot[10:], cfs[10:], color='yellowgreen')


    ymin,ymax = ax.get_ylim()
    ax.set_yticks([new_x_freq[0], new_x_freq[64], new_x_freq[-1]])

    ax.axvline(9.5,ymin,ymax, linestyle=':', color='black')
    ax.set_xticks(xplot)
    ax.set_xticklabels(all_names, rotation=90)
    ax.set_ylabel('Average Frequency (Hz)', size=15, fontweight='bold')
    fig.tight_layout()







#######################
text_names, marm_names = {}, {}
for kk, ii in texture_dict.items():
    base_text = os.path.splitext(os.path.basename(ii))[0].split('_')[0]
    text_names[kk] = base_text
for mm, nn in marm_dict.items():
    base_marm = os.path.splitext(os.path.basename(nn))[0].split('_')[0]
    marm_names[mm] = base_marm

# texture_set = glob.glob(texture_dir)
# marm_set = glob.glob(marm_dir)
#
# texture_dict, marm_dict = {}, {}
# for rr, tt in enumerate(texture_set):
#     texture_dict[rr] = tt
# for ee, mm in enumerate(marm_set):
#     marm_dict[ee] = mm






import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from nems.analysis.gammatone.gtgram import gtgram

texture_dir = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Textures/*.wav"
marm_dir = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Marms/*.wav"
ferret_dir = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Ferrets/*.wav"
pairs = [(0,7),(1,9),(2,5),(3,4),(4,6),(5,0),(6,3),(7,1),(8,2),(9,8)] # The ones I thought were best


def spec_combos(texture_dir, marm_dir, pairs=None):
    '''Takes two directories of wav files and displays all overlapping combinations with
    individual spectrograms plotted on the outside. Wavs must be same length. Option to plot
    only certain combinations of sounds by passing list of tuples corresponding to which
    wav files you want combined. Leaving it blank or giving it a bad input (not tuples or
    indexes out of range of how many files in the directories) will default to displaying
    all combinations.'''
    texture_dict, marm_dict = sound_dict_maker(texture_dir), sound_dict_maker(marm_dir)

    if pairs:
        for uu in pairs:
            if type(uu) is not tuple:
                print(f"Each element should be a tuple, displaying everything")
                pairs = None
            if uu[0] >= len(texture_dict) or uu[1] >= len(marm_dict):
                print(f"Pair index will be out of range, displaying everything")
                pairs = None

    fig, ax = plt.subplots(len(texture_dict)+1,len(marm_dict)+1)
    for pp in range(ax.shape[0]):
        for oo in range(ax.shape[1]):
            ax[pp,oo].set_xticks([])
            ax[pp,oo].set_yticks([])
            ax[pp,oo].set_xticklabels([])
            ax[pp,oo].set_yticklabels([])
    ax[0,0].axis('off')

    for gg, hh in texture_dict.items():
        fs, W = wavfile.read(hh)
        basename = os.path.splitext(os.path.basename(texture_dict[gg]))[0].split('_')[0]
        spec_text = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)
        ax[gg+1,0].imshow(np.sqrt(spec_text), aspect='auto', origin='lower')
        for jj, kk in marm_dict.items():
            fs2, W2 = wavfile.read(kk)
            basename2 = os.path.splitext(os.path.basename(marm_dict[jj]))[0].split('_')[0]
            spec_marm = gtgram(W2, fs2, 0.02, 0.01, 48, 100, 8000)
            ax[0,jj+1].imshow(np.sqrt(spec_marm), aspect='auto', origin='lower')
            if not pairs:
                spec = gtgram(W+W2, fs, 0.02, 0.01, 48, 100, 8000)
                ax[gg+1,jj+1].imshow(np.sqrt(spec),aspect='auto',origin='lower')
            if jj == 0:
                ax[gg+1,jj].set_ylabel(basename, rotation='horizontal', horizontalalignment='right')
            if gg == 0:
                ax[gg,jj+1].set_title(basename2)

    if pairs:
        for ss in pairs:
            soundA, soundB = texture_dict[ss[0]], marm_dict[ss[1]]
            fs, A = wavfile.read(soundA)
            fs2, B = wavfile.read(soundB)
            spec = gtgram(A+B, fs, 0.02, 0.01, 48, 100, 8000)
            ax[ss[0]+1,ss[1]+1].imshow(np.sqrt(spec), aspect='auto', origin='lower')


def sound_combos(texture_dir, marm_dir, pairs):
    '''Takes two directories of wav files as well as a list of tuples and will go through
    the directory and produce 8 different wav files based on the sounds indicated in the
    tuples. The 8 are
    1. A Alone - 2. B Alone - 3. A + B - 4. A + half B - 5. half A + B - 6. half A -
    7. half B - 8. half A + half B'''
    texture_dict, marm_dict = sound_dict_maker(texture_dir), sound_dict_maker(marm_dir)

    for aa in pairs:
        save_paths = {}
        SAVE_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/WIP Combos/"

        soundA = texture_dict[aa[0]]
        soundB = marm_dict[aa[1]]

        base_text = os.path.splitext(os.path.basename(texture_dict[aa[0]]))[0].split('_')[1]
        base_marm = os.path.splitext(os.path.basename(marm_dict[aa[1]]))[0].split('_')[1]
        fs, A = wavfile.read(soundA)
        fs2, B = wavfile.read(soundB)
        if fs != fs2:
            print('Sampling rates do not match!')

        half_A, half_B = A.copy(), B.copy()
        half_A[:int(len(A) / 2)], half_B[:int(len(B) / 2)] = 0, 0

        save_paths[f'{base_text}-0-1.wav'] = A
        save_paths[f'{base_marm}-0-1.wav'] = B
        save_paths[f'{base_text}-0-1_{base_marm}-0-1.wav'] = A + B
        save_paths[f'{base_text}-0-1_{base_marm}-0.5-1.wav'] = A + half_B
        save_paths[f'{base_text}-0.5-1_{base_marm}-0-1.wav'] = half_A + B
        save_paths[f'{base_text}-0.5-1.wav'] = half_A
        save_paths[f'{base_marm}-0.5-1.wav'] = half_B
        save_paths[f'{base_text}-0.5-1_{base_marm}-0.5-1.wav'] = half_A + half_B

        for dd, ff in save_paths.items():
            wavfile.write(SAVE_ROOT + dd, fs, ff)


def all_sound_combos(texture_dir, marm_dir):
    '''Takes two directories of wav files as well as a list of tuples and will go through
    the directory and produce 8 different wav files based on the sounds indicated in the
    tuples. The 8 are
    1. A Alone - 2. B Alone - 3. A + B - 4. A + half B - 5. half A + B - 6. half A -
    7. half B - 8. half A + half B'''
    texture_dict, marm_dict = sound_dict_maker(texture_dir), sound_dict_maker(marm_dir)
    save_paths = {}
    SAVE_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Sound Combos/"

    for tt in texture_dict.keys():
        sound_A = texture_dict[tt]
        base_text = os.path.splitext(os.path.basename(texture_dict[tt]))[0].split('_')[1]
        fs_A, A = wavfile.read(sound_A)
        half_A = A.copy()
        half_A[:int(len(A) / 2)] = 0

        save_paths[f'{base_text}-0-1.wav'] = A
        save_paths[f'{base_text}-0.5-1.wav'] = half_A

        for mm in marm_dict:
            sound_B = marm_dict[mm]
            base_marm = os.path.splitext(os.path.basename(marm_dict[mm]))[0].split('_')[1]
            fs_B, B = wavfile.read(sound_B)
            half_B = B.copy()
            half_B[:int(len(B) / 2)] = 0

            save_paths[f'{base_marm}-0-1.wav'] = B
            save_paths[f'{base_marm}-0.5-1.wav'] = half_B

            if fs_A != fs_B:
                print(f'Sampling rates of Background: {base_text} and '
                      f'Foreground: {base_marm} do not match!')

            save_paths[f'{base_text}-0-1_{base_marm}-0-1.wav'] = A + B
            save_paths[f'{base_text}-0-1_{base_marm}-0.5-1.wav'] = A + half_B
            save_paths[f'{base_text}-0.5-1_{base_marm}-0-1.wav'] = half_A + B
            save_paths[f'{base_text}-0.5-1_{base_marm}-0.5-1.wav'] = half_A + half_B

    for dd, ff in save_paths.items():
        wavfile.write(SAVE_ROOT + dd, fs_A, ff)

def sound_dict_maker(dir):
    '''Saves some space in the code above, turns the sound directories to be indexable.'''
    set = glob.glob(dir)
    dict = {}
    for rr, tt in enumerate(set):
        dict[rr] = tt
    return dict