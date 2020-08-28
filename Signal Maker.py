import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample
from nems.analysis.gammatone.gtgram import gtgram


from nems.analysis.gammatone.gtgram import gtgram
from nems import recording, signal
from nems import xforms, get_setting
import nems.gui.editors as gui
import nems
import random
import nems.tf.cnnlink_new as cnn
import scipy.stats as stats
import scipy.stats

def main_function(length=15, bins=6, min_seg=0.5, max_seg=2):


#Generation of classification set
#Got to load all the wav files first, then chop and save em
for dd in sound_directories:
    splice_gen()
#Sound directories will be list of the directories for the wavs

BAPHY_ROOT = "/Users/grego/baphy"
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
class_labels = ['Ferret', 'Speech','Torc']

dir1 = os.path.join(BAPHY_ROOT, "SoundObjects", "@FerretVocal", "Sounds_set4", "*.wav")
dir2 = os.path.join(BAPHY_ROOT, "SoundObjects", "@Speech", "sounds", "*sa1.wav")
set1 = glob.glob(dir1)
set2 = glob.glob(dir2)
sound_set1 = set1
sound_set2 = set2


sound_classes = (np.zeros(len(set1)) + 1)
sound_files = set1


# generate a label for each wav file
sound_classes = np.concatenate((np.zeros(len(set1)) + 1,
                                np.zeros(len(set2)) + 2,))
sound_files = set1 + set2


def load_wavs(sounds, classes):
    '''This will load all of the wave files and keep it tidy with the name of the sound, its class
    and it will also check to make sure they're all sampled at the same rate, and if not, it will
    resample them.'''
    stim_dict = {}
    fs_list = []
    for nn, (c, s) in enumerate(zip(sound_classes, sound_files)):
        fs, W = wavfile.read(s)
        basename = os.path.splitext(os.path.basename(s))[0]
        tuple = (c, basename, W)
        stim_dict[nn] = tuple
        fs_list.append(fs)

    uniques = np.unique(fs_list)
    print(f'The sampling frequencies in this set are {uniques}')
    # Need to ask Stephen how to up sample, for my dummy data shouldn't be hard.
    # print(f'The length of unique is {len(uniques)}')
    # if len(uniques) > 1:

    return stim_dict, uniques[0]

stim_dict, fs = load_wavs(sound_files,sound_classes)



def empty_trial(length=15,bins=6,fs=48000,overlap=None):
    '''Basic trial structure, you specify the length and how many
    bins for samples you want (with the option of overlap, where you define
    how many bins should have overlap and it'll randomly pick em), and
    it'll create that in seconds before multiplying by the sampling
    rate'''
    trial = np.zeros(length*fs)
    bin_length = int(length / bins * fs)
    bin_start_times = [bin_length * i for i in range(bins)]
    if overlap:
        extra = random.sample(range(0,bins), overlap)
        for bb in extra:
            bin_start_times.append(bin_start_times[bb])
        bin_start_times.sort()
    return trial, bin_start_times

def empty_triad(length=15,bins=6,fs=48000,overlap=None):
    '''Basic trial structure, you specify the length and how many
    bins for samples you want (with the option of overlap, where you define
    how many bins should have overlap and it'll randomly pick em), and
    it'll create that in seconds before multiplying by the sampling
    rate'''
    # length,bins,fs,overlap = 15,6,48000,2
    trial = np.zeros((3, length * fs))
    # trial = [np.zeros(length*fs) * 3 for i in range(3)]

    bin_length = int(length / bins * fs)
    bin_start_times = [bin_length * i for i in range(bins)]
    trial_fills = [[0] * bins for i in range(3)]
    extra_starts = [[] * r for r in range(3)]
    if overlap:
        if (bins/overlap).is_integer():
            extra = random.sample(range(0,bins),bins)
            # n = int(bins/3)
            # overlap_positions = [extra[i * n:(i + 1) * n] for i in range((len(extra) + n - 1) // n)]
            for qq in range(len(extra)):
                # extra_starts[qq % 3].append(bin_start_times[extra[qq]])
                extra_starts[qq % 3].append(extra[qq])
            A_starts = [extra_starts[1],extra_starts[2],extra_starts[0]]
            B_starts = [extra_starts[2],extra_starts[0],extra_starts[1]]
            for tri in range(3):
                for ab in extra_starts[tri]:
                    trial_fills[tri][ab] = f'A{ab+1}+B{ab+1}'
                for aa in A_starts[tri]:
                    trial_fills[tri][aa] = f'A{aa+1}'
                for bb in B_starts[tri]:
                    trial_fills[tri][bb] = f'B{bb+1}'

    return trial, triad, bin_start_times


def segment_gen(len_sec, min_seg=0.5, max_seg=2.0):
    '''Creates a list that gets used as the positions within a wav file to make cuts
    that make use of the entire sound segment. For more universal function I made it so
    the number of pieces it creates corresponds to the number of seconds. I have a version
    where you can define it, but I wasn't yet able to have it handle extreme cases and
    parameters well yet. This version works well for my 3 or 4s stimuli at the moment'''
    if max_seg >= len_sec:
        add_list = []
        print('The longest segment length you want is too long for your sound file!')
    elif min_seg < 0:
        add_list = []
        print('The minimum segment cannot be below zero!')
    elif max_seg - min_seg < 0:
        add_list = []
        print('Minimum segment length must be smaller than the maximum segment length!')
    else:
        pieces = int(len_sec)
        list, add_list = [min_seg] * pieces, []
        ii, time_remaining, total, max_add = 1, len_sec - np.sum(list), 0, max_seg - min_seg
        # print(f'Length {len_sec}, Remaining {time_remaining}, Min {min_seg}, Pieces {pieces}, Max to add {max_add}')
        for aa in range(pieces):
            if time_remaining < max_add and aa == pieces - 1:
                sample = np.round(time_remaining,1)
                # print(f'I used this to add {time_remaining}')
            elif time_remaining > max_add and aa == pieces - 1:
                print('There is an error from the random generator and you might consider rerunning this.')
            if max_add <= time_remaining and aa != pieces - 1:
                sample = np.round(np.random.uniform(0, max_add), 1)
                # print(f'max_add {max_add} <= time_remaining {time_remaining}')
            if max_add > time_remaining and aa != pieces - 1:
                sample = np.round(np.random.uniform(0, time_remaining), 1)
            add_list.append(sample)
            # print(f'On the {aa+1} loop I added {sample} to give {add_list}, tr {time_remaining - sample}')
            time_remaining = time_remaining - sample
        zip_list = zip(list, np.abs(add_list))
        cut_list = [x + y for (x, y) in zip_list]
        print(f'Generated a list of cuts {cut_list} seconds in duration.')

    return cut_list

#how many cuts you want?

cut_dir = []
for bb in range(5):
    listy = segment_gen(3,0.5,2)
    cut_dir.append(listy)

# This makes the list that the function generates exist in all its permutations, this may be useless
# and it does create the problem of making tuples not lists... I would have to work on that.
cut_dir = []
for bb in range(1):
    listy = segment_gen(4,0.5,2)
    perms = permutations(listy)
    for cc in perms:
        cut_dir.append(cc)
    print(f'{listy} : sum is {np.sum(listy)}')


def sound_segmenter(sound, cuts, fs=fs):
    '''This version of the function takes a single list of points to cut, converts them to samples
    based on the sampling rate of the sound, and cuts that sound into those pieces. It also uses
    the id tuples generated with class and wav file source to create a detailed name for each
    generated segment of the original sound file.'''
    start, stop = 0, 1
    samples_list = [0]   # It will need to know later for a couple things that the first cut starts at 9
    segment_list = []
    for vv in cuts:
        seg_samples = vv * fs
        sample_add = samples_list[-1] + seg_samples  # adds the next segment length to the previous segment's
        samples_list.append(int(sample_add))
    while stop <= len(cuts):
        segment = sound[samples_list[start]:samples_list[stop]]  # Counter indicates which segment to take from sample list
        label = f'{ids[1]}_seg-{seconds_list[start]}-{seconds_list[stop]}s_dur-{cc[start]}_CLASS-0{int(ids[0])}'
        seg = (label, segment)
        segment_list.append(seg)
        start += 1
        stop += 1
    return segment_list

sound_segmenter(stim_dict,cut_dir,fs)

def sound_segmenter(stim_dict, cuts, fs, seg_label_list=None, seg_dict=None):
    '''Takes your dictionary of sounds (labeled and categorized) and iterates through chopping
    each sound by however many cuts lists one has. If it is the first time you are running it
    it will make a new output dictionary that contains the label and the segment waveform, as
    well as a list of labels that correspond to the dictionary keys.'''
    if not seg_dict:
        seg_dict = {}
    if not seg_label_list:
        seg_label_list = []

    for c, s in stim_dict.items():
        for nn, cc in enumerate(cuts):
            start, stop = 0, 1
            samples_list = [0]
            for vv in cc:
                seg_samples = vv * fs  # convert seconds to samples
                sample_add = samples_list[-1] + seg_samples
                samples_list.append(int(sample_add))
            seconds_list = [x / fs for x in samples_list]
            while stop <= len(cc):
                segment = stim_dict[c][2][samples_list[start]:samples_list[stop]]
                label = f'{stim_dict[c][1]}_seg-{seconds_list[start]}-{seconds_list[stop]}s_dur-{cc[start]}_cutids-{nn}_CLASS-0{int(stim_dict[c][0])}'
                seg_dict[label] = segment
                seg_label_list.append(label)
                start += 1
                stop += 1
    return seg_dict, seg_label_list




def sound_segmenter(sound, cuts, ids, fs, seg_label_list=None, seg_dict=None):
    '''This version is the original and takes a whole list of all the cuts that will be made to a
    single sound file and goes through all at once, could be more useful ultimately. but simpler
    version is also around.

    This waqy I could add functionality to label which cut list it is using so I know the
    A. source file, B. which cut it is from C. what particular time period of the single cut is'''
    if not seg_dict:
        seg_dict = {}
    if not seg_label_list:
        seg_label_list = []
    for nn, cc in enumerate(cuts):
        start, stop = 0, 1
        samples_list = [0]
        for vv in cc:
            seg_samples = vv * fs  # convert seconds to samples
            sample_add = samples_list[-1] + seg_samples
            samples_list.append(int(sample_add))
        seconds_list = [x / fs for x in samples_list]
        while stop <= len(cc):
            segment = sound[samples_list[start]:samples_list[stop]]
            label = f'{ids[1]}_seg-{seconds_list[start]}-{seconds_list[stop]}s_dur-{cc[start]}_cutids-{nn}_CLASS-0{int(ids[0])}'
            seg_dict[label] = segment
            seg_label_list.append(label)
            start += 1
            stop += 1
    return seg_dict, seg_label_list



segs_dict_cat1 = sound_segmenter(sound,cut_dir,ids,fs)
segs_dict_cat2 = sound_segmenter(sound2,cut_dir,ids2,fs)

for nn in range(10):
    category = random.randrange(1, num_cats+1, 1)
    print(category)





#Time to populate the empty trial with these
def populate_triad(trial, trial_labels, start_times, cats, fs, seg_dict, seg_label_list, bins):
    bin_length = (start_times[1] - start_times[0]) / fs
    start_times_s = start_times / fs

    sound_dict = {}
    epochs = pd.DataFrame(columns=['Name', 'Class', 'On', 'Off', 'Segment'],)
    # sound_list = []
    for bn, tt in enumerate(start_times_s):
        for ab in range(2):
            cat = random.randrange(1, cats+1, 1)
            good_list = []
            for hh in seg_label_list:
                if f'CLASS-0{cat}' in hh:
                    good_list.append(hh)
            random.shuffle(good_list)
            add_label = random.choice(good_list)
            add_wave = seg_dict[add_label]

            seg_len = len(add_wave) / fs
            max_jitter_length = bin_length - seg_len
            jitter = np.round(np.random.uniform(0, max_jitter_length), 1)
            on_time = tt + jitter
            off_time = on_time + seg_len
            on_time_fs = int(on_time * fs)
            #how to get the on and off times associated with these to pass on, new dict?

            pre = np.zeros(int(jitter * fs))
            post = np.zeros(int((bin_length - (jitter + seg_len)) * fs))
            wave = np.concatenate((pre, add_wave, post))
            if len(wave) != int((bin_length * fs)):
                for ab in range(int((bin_length * fs) - len(wave))):
                    wave = np.append(wave,0)

            row = {'Name': 'STIM_' + add_label.split('_')[0], 'Class': f"CLASS_0{cat}", 'On': on_time, 'Off': off_time,'Segment': add_label.split('_')[1]}
            epochs = epochs.append(row, ignore_index=True)

            if ab == 0:
                sound_dict[f'A{bn+1}'] = wave
                # sound_list.append(f'A{bn+1}')
                wave_A = wave
            if ab == 1:
                sound_dict[f'B{bn+1}'] = wave
                # sound_list.append(f'B{bn+1}')
                wave_B = wave
        wave_AB = wave_A + wave_B
        sound_dict[f'A{bn+1}+B{bn+1}'] = wave_AB
        # sound_list.append(f'A{bn+1}+B{bn+1}')

    for trl in range(len(trial_labels)):
        for num, pop in enumerate(trial_labels[trl]):
            trial[trl][start_times[num]:int(start_times[num]+(bin_length*fs))] = sound_dict[pop]


    return trial, epochs


        epochs = pd.DataFrame(columns=['Name', 'On', 'Off'])

        b = "STIM_" + os.path.basename(s)

        row = {'Name': b, 'start': lastendtime, 'end': thisendtime}
        epochs = epochs.append(row, ignore_index=True)
        row = {'name':  f"CLASS_{c:.0f}", 'start': lastendtime + silence, 'end': thisendtime - silence}
        epochs = epochs.append(row, ignore_index=True)
        row = {'name': "REFERENCE", 'start': lastendtime + silence, 'end': thisendtime - silence}
        epochs = epochs.append(row, ignore_index=True)


    return sound_dict, sound_list


    for n, b in enumerate(start_times_s):
        # First pick what category we are goin
        cat = random.randrange(1, cats+1, 1)
        good_list = []
        for hh in seg_label_list:
            if f'CLASS-0{cat}' in hh:
                good_list.append(hh)
        random.shuffle(good_list)
        add_label = random.choice(good_list)
        add_wave = seg_dict[add_label]

        print(f'Bin {n}: File {add_label}')

        seg_len = len(add_wave) / fs
        max_jitter_length = bin_length - seg_len
        jitter = np.round(np.random.uniform(0, max_jitter_length), 1)
        on_time = b + jitter
        off_time = on_time + seg_len
        on_time_fs = int(on_time * fs)

        print(f'It started at {on_time} and ended at {off_time}')

        trial[on_time_fs:on_time_fs + len(add_wave)] = add_wave

        trial_df = pd.DataFrame(columns=['Base File', 'Class', 'Duration', 'Start Time', 'Stop Time'])


    plt.plot()
    plt.imshow(gtgram(trial, fs, 0.02, 0.01, 18, 100, 8000),aspect='auto')




def populate_trial(trial, start_times, seg_dict, seg_label_list, cats, fs):
    bin_length = (start_times[1] - start_times[0]) / fs
    start_times_s = start_times / fs

    for n, b in enumerate(start_times_s):
        # First pick what category we are goin
        cat = random.randrange(1, cats+1, 1)
        good_list = []
        for hh in seg_label_list:
            if f'CLASS-0{cat}' in hh:
                good_list.append(hh)
        random.shuffle(good_list)
        add_label = random.choice(good_list)
        add_wave = seg_dict[add_label]

        print(f'Bin {n}: File {add_label}')

        seg_len = len(add_wave) / fs
        max_jitter_length = bin_length - seg_len
        jitter = np.round(np.random.uniform(0, max_jitter_length), 1)
        on_time = b + jitter
        off_time = on_time + seg_len
        on_time_fs = int(on_time * fs)

        print(f'It started at {on_time} and ended at {off_time}')

        trial[on_time_fs:on_time_fs + len(add_wave)] = add_wave

        trial_df = pd.DataFrame(columns=['Base File', 'Class', 'Duration', 'Start Time', 'Stop Time'])


    plt.plot()
    plt.imshow(gtgram(trial, fs, 0.02, 0.01, 18, 100, 8000),aspect='auto')



    fig, ax = plt.subplots(1)
    _,freq,_,bar = ax.specgram(signalData, Fs=fs)

empty_ar[2000:2000 + 10000] = trial_data
np.round(np.random.uniform(0, max_add), 1)

new = [if f'CLASS-0{cat}' in seg_label_list]
new = [i == f'CLASS-0{cat}' for i in seg_label_list]

newt = [i for i in seg_label_list if i == f'CLASS-0{cat}']

good_list = []
for gg in seg_label_list:
    if f'CLASS-0{cat}' in gg:
        good_list.append(gg)


dfObj.append({'User_ID': 23, 'UserName': 'Riti', 'Action': 'Login'}, ignore_index=True)

    segment_df = pd.DataFrame(columns=['Base File', 'Class', 'Duration', 'Start Time', 'Stop Time'])

    for ff in range(10):
        print(np.random.normal(pieces_avg, .4))




list = [bin_length * i for i in range(bins)]
def splice_gen(sound_set,seg_percent=.25)

#input params: pieces to chop into

for s in sound_set:
    seg_percent = 0.25
    fs, W = wavfile.read(set1[0])    #Load each file in the set
    len_sec = int(W.shape[0]/fs)
    avg_seg = W.shape[0] * seg_percent  #how long with fs each segment should be
    pieces = int(1/seg_percent)      #num pieces to chop into
    pieces_avg = len_sec/pieces
    segment_len_list = []
    sigma = 0.5/pieces_avg
    lower, upper = 0.5, 2
    mu, sigma = 1, 0.4




    for ii in range(pieces):
        while ii <= (pieces - 1):
            if ii != (pieces - 1):
                sample = np.round(scipy.stats.truncnorm.rvs(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=1),1)
                print(sample)
                if ii == 0:
                    total = sample
                    print(f'Total = {total}')
                    remaining = len_sec - sample
                    print(f'{remaining} remaining')
                else:
                    total = total + sample
                    print(f'Total = {total}')
                    remaining = remaining - sample
                    print(f'{remaining} remaining')
                if remaining >= 0.5 and remaining <= 2:
                    sample = remaining
                    print(f'last one {sample}')
seed_list = np.random.uniform(0,10,10)

def segment_gen(len_sec,pieces=4,min_seg=0.5,max_seg=2.0):
    list, ii, remaining, total = [], 1, len_sec, 0
    print(f'len_sec = {len_sec}, pieces = {pieces}')
    while ii <= pieces: #or total <= len_sec:
        # sample = np.asscalar(np.round(scipy.stats.truncnorm.rvs(
        #     (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=1), 1))
        sample = np.round(np.random.uniform(min_seg, max_seg), 1)
        print(f'The {ii} selected number is {sample}')
        if len(list) == pieces:
            final_sample = len_sec - total
            list.append(final_sample)
            print(f'The length of the list now exceeds pieces by 1')
            break
        if ii == 0:
            total = sample
            remaining = len_sec - sample
            list.append(sample)
            print(f'This is the first sample, {total}, with {remaining} remaining from {len_sec}')
            # print(f'first total {total} : remaining {remaining}')
        else:
            total = total + sample
            remaining = remaining - sample
            # print(f'{ii} total: {total} : remaining {remaining}')
            if remaining >= 0.5:
                list.append(sample)
                print(f'The remainder was greater than {min_seg}, so it was added to the list')
                #might be an error here, maybe append remaining not sample?
            else:
                # if total >= len_sec:
                print(f'Remainder was less than {min_seg} and total greater than {len_sec}, generating new sample')
                sample = sample - np.abs(remaining)
                remaining = remaining - sample
                # print(f'new sample:{sample}, remaining {remaining}')
                if 0.5 <= sample <= 2:
                    list.append(sample)
                    print(f'The new sample is the appropriate length of {sample}, it was added')
                # else:
                #     print(f'Remainder was less than {min_seg} and total less than {len_sec}, fixing it')
                #
        ii += 1
    if np.sum(list) < len_sec:
        if (len_sec - np.sum(list)) >= 0.5:
            list.append(len_sec - np.sum(list))
            print("I did something out here in the first one")
        else:
            if np.sum(list) < len_sec:
                if list[-1] + remaining <= 2:
                    list[-1] = list[-1] + remaining
            print('I did something out here in the second one')
                # elif list[-2] + remaining <= 2:
                #     list[-2] = list[-2] + remaining
                # elif list[-3] + remaining <= 2:
                #     list[-3] = list[-3] + remaining
    round_list = [round(num, 1) for num in list]
    return round_list

dict = {}
for aa in range(5):
    listy = segment_gen(3,3)
    dict[aa] = listy
    print(f'{listy} : sum is {np.sum(listy)}')

    import numpy as np


    def slice_segment(total_len=3, min_seg=0.2, max_seg=1):
        lens = np.round(np.random.uniform(min_seg, max_seg, int(total_len / min_seg + 1)), 1)
        cut_points = np.cumsum(lens)
        cut_points = cut_points[cut_points < total_len]
        if total_len - cut_points[-1] < min_seg:
            print('warning: last segment < min_seg, consider excluding')
        return cut_points

for aa in range(10):
    lister = slice_segment(3,0.5,2)
    print(f'{lister} : sum is {np.sum(lister)}')


    #Check to see if any values are above 2 or below .5 at end, and reshuffle if so

##Create list of lists to chop up signals
chop_length = [[1.1,1.0,0.9],[0.8,1.3,0.9],[0.7,1.7,0.6],[
    1.5,0.5,1.0],[1.0,1.2,0.8]]
chop_list = chop_length
def wav_segmenter(sound,ids,chop_list,fs=fs):
    for cc in chop_list:
        start, stop = 0, 1
        samples_list = [0]
        for vv in cc:
            seg_samples = vv * fs #convert seconds to samples
            sample_add = samples_list[-1] + seg_samples
            samples_list.append(int(sample_add))
        seconds_list = [x / fs for x in samples_list]
        while stop <= len(cc):
            segment = sound[samples_list[start]:samples_list[stop]]
            # label = f'{ids[1]}_{seconds_list[start]}-{seconds_list[stop]}_CLASS-0{int(ids[0])}'
            label = f'{ids[1]}_seg-{seconds_list[start]}-{seconds_list[stop]}s_dur-{cc[start]}_CLASS-0{int(ids[0])}'
            start += 1
            stop += 1
            print(label)
dfObj.append({'User_ID': 23, 'UserName': 'Riti', 'Action': 'Login'}, ignore_index=True)

segment_df = pd.DataFrame(columns=['Base File', 'Class', 'Duration', 'Start Time', 'Stop Time'])

    for ff in range(10):
        print(np.random.normal(pieces_avg,.4))


	Resample to desired fs - fs_in →  fs=40000
	Then choose cut points: 1.2 , 1.3, 0.5 sec
	Seg1samples = 1.2 * 40000 (convert seconds to samples)
	Seg2samples = 1.3 * 40000 (convert seconds to samples)

	Seg1 = w[0:seg1samples]
	Seg2 = w[seg1samples:(seg1samples+seg2samples)]

	10 ms ramp: ramp_samples=int(fs * 0.01)
		ramp = np.linspace(0,1,ramp_samples)
		segn[:ramp_samples]=segn[:ramp_samples]*ramp
		Offramp = np.flip(ramp)
		segn[(len(segn)-ramp_samples):len(segn)]  *= offramp








#Make code that generates


#Create parameters for overall trial
#
def stim_shell(length=15,num_bins=6,chans=18,fs=48000):
    empty = np.zeros([chans,length*fs])

    bin_length = length/num_bins
    start_times = []
    for bb in range(num_bins):
        time = 2.5 * bb * fs
        start_times.append(time)

#Create List of start times