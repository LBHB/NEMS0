import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from pydub import AudioSegment
import matplotlib.gridspec as gridspec

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


def spectro(name, kind):
    GREG_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"
    filepath = GREG_ROOT + name + '.wav'

    fs, W = wavfile.read(filepath)

    spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)

    plt.imshow(spec, aspect='auto', origin='lower')
    plt.title(name)


TEXTURE_ROOT = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Textures"
MARM_ROOT = "/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Marms/"

def spec_collage(ROOT):
    dir = os.path.join(ROOT, "*.wav")
    set1 = glob.glob(dir)
    fig, ax = plt.subplots(5,4)

    for nn, ss in enumerate(set1):
        fs, W = wavfile.read(ss)
        spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)
        pow = np.average(spec,axis=1)
        basename = os.path.splitext(os.path.basename(ss))[0].split('_')[0]
        if nn < int(len(set1)/2):
            ax[nn,0].set_title(basename)
            ax[nn,0].imshow(spec,aspect='auto',origin='lower')
            ax[nn,0].set_xticks([])
            ax[nn,0].set_xticklabels([])
            ax[nn,1].plot(pow)
            ax[nn,1].set_yticklabels([])
            ax[nn,1].set_yticks([])
        else:
            ax[nn-5,2].set_title(basename)
            ax[nn-5,2].imshow(spec,aspect='auto',origin='lower')
            ax[nn-5,2].set_xticks([])
            ax[nn-5,2].set_xticklabels([])
            ax[nn-5,3].plot(pow)
            ax[nn-5,3].set_yticklabels([])
            ax[nn-5,3].set_yticks([])





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
pairs = [(0,7),(1,9),(2,5),(3,4),(4,6),(5,0),(6,3),(7,1),(8,2),(9,8)]


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
        SAVE_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Conditions/Pair-{aa[0]}_"

        soundA = texture_dict[aa[0]]
        soundB = marm_dict[aa[1]]

        base_text = os.path.splitext(os.path.basename(texture_dict[aa[0]]))[0].split('_')[0]
        base_marm = os.path.splitext(os.path.basename(marm_dict[aa[1]]))[0].split('_')[0]
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

def sound_dict_maker(dir):
    '''Saves some space in the code above, turns the sound directories to be indexable.'''
    set = glob.glob(dir)
    dict = {}
    for rr, tt in enumerate(set):
        dict[rr] = tt
    return dict
