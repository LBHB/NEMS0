import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from nems.analysis.gammatone.gtgram import gtgram
from nems import recording, signal
from nems import xforms, get_setting
import nems.gui.editors as gui
import nems
import random
import nems.tf.cnnlink_new as cnn
import scipy.stats as stats
import scipy.stats




BAPHY_ROOT = "/Users/grego/baphy"
RECORDING_PATH = get_setting('NEMS_RECORDINGS_DIR')
recording_file = os.path.join(RECORDING_PATH, "classifier.tgz")
class_labels = ['Ferret']
cats = len(class_labels)

dir1 = os.path.join(BAPHY_ROOT, "SoundObjects", "@FerretVocal", "Sounds_set4", "*.wav")
# dir2 = os.path.join(BAPHY_ROOT, "SoundObjects", "@Speech", "sounds", "*sa1.wav")
set1 = glob.glob(dir1)
# set2 = glob.glob(dir2)
sound_set1 = set1
# sound_set2 = set2

sound_classes = (np.zeros(len(set1)) + 1)
sound_files = set1
# sound_classes = np.concatenate((np.zeros(len(set1)) + 1,
#                                 np.zeros(len(set2)) + 2,))
# sound_files = set1 + set2



unique_triads = 1
num_cut_sets = 1
all_trials, all_epochs = {}, {}

all_trials, all_epochs = {}, {}
#Load all the sound files we may use and gives them class assignments, too
stim_dict, fs, seg_len = load_wavs(sound_files,sound_classes)

#Make a bunch of sets of cut points and then cut all the files you loaded
cut_dir = cut_decider(num_cut_sets, seg_len)
seg_dict, seg_label_list = sound_segmenter(stim_dict, cut_dir, fs)

#Take and select some segments for each segment position and then populate each of three trials and
#add it to the directories to keep track of the trial waveforms and corresponding epochs
for mm in range(unique_triads):
    all_trials, all_epochs = make_three_trials(all_trials, all_epochs, stim_dict,
                                               cut_dir, fs, cats, 15, 6, 2)



# length, bins, fs, overlap = 15, 6, 48000, 2
# trial = np.zeros(length * fs)
# trial = [np.zeros(length * fs)] * 3
#
# bin_length = int(length / bins * fs)
# bin_start_times = [bin_length * i for i in range(bins)]
# triad_starts = [bin_start_times] * 3
# triad = [[0] * bins for i in range(3)]
# extra_starts = [[] * r for r in range(3)]
# if overlap:
#     if (bins / overlap).is_integer():
#         extra = random.sample(range(0, bins), bins)
#         n = int(bins / 3)
#         # overlap_positions = [extra[i * n:(i + 1) * n] for i in range((len(extra) + n - 1) // n)]
#         for qq in range(len(extra)):
#             # extra_starts[qq % 3].append(bin_start_times[extra[qq]])
#             extra_starts[qq % 3].append(extra[qq])
#         A_starts = [extra_starts[1], extra_starts[2], extra_starts[0]]
#         B_starts = [extra_starts[2], extra_starts[0], extra_starts[1]]
#         triads = [0] * 6
#         for tri in range(3):
#             for ab in extra_starts[tri]:
#                 triad[tri][ab] = f'A{ab + 1}+B{ab + 1}'
#             for aa in A_starts[tri]:
#                 triad[tri][aa] = f'A{aa + 1}'
#             for bb in B_starts[tri]:
#                 triad[tri][bb] = f'B{bb + 1}'
#
#
#
# length,bins,fs = 15,6,48000
# trial = np.zeros(length * fs)
# bin_length = int(length / bins * fs)
# bin_start_times = [bin_length * i for i in range(bins)]
# triad_starts = [bin_start_times] * 3
# extra_starts = [[] * r for r in range(3)]
#
# extra = random.sample(range(0, bins), bins)
#
# for qq in range(len(extra)):
#     triad_starts[qq % 3].append(bin_start_times[extra[qq]])
#
# for qq in range(len(extra)):
#     extra_starts[qq % 3].append(bin_start_times[extra[qq]])

# def segment_gen(len_sec, min_seg=0.5, max_seg=2.0):
#     '''Creates a list that gets used as the positions within a wav file to make cuts
#     that make use of the entire sound segment. For more universal function I made it so
#     the number of pieces it creates corresponds to the number of seconds. I have a version
#     where you can define it, but I wasn't yet able to have it handle extreme cases and
#     parameters well yet. This version works well for my 3 or 4s stimuli at the moment'''
#     if max_seg >= len_sec:
#         add_list = []
#         print('The longest segment length you want is too long for your sound file!')
#     elif min_seg < 0:
#         add_list = []
#         print('The minimum segment cannot be below zero!')
#     elif max_seg - min_seg < 0:
#         add_list = []
#         print('Minimum segment length must be smaller than the maximum segment length!')
#     else:
#         pieces = int(len_sec)
#         list, add_list = [min_seg] * pieces, []
#         ii, time_remaining, total, max_add = 1, len_sec - np.sum(list), 0, max_seg - min_seg
#         # print(f'Length {len_sec}, Remaining {time_remaining}, Min {min_seg}, Pieces {pieces}, Max to add {max_add}')
#         for aa in range(pieces):
#             if time_remaining < max_add and aa == pieces - 1:
#                 sample = np.round(time_remaining,1)
#                 # print(f'I used this to add {time_remaining}')
#             elif time_remaining > max_add and aa == pieces - 1:
#                 print('There is an error from the random generator and you might consider rerunning this.')
#             if max_add <= time_remaining and aa != pieces - 1:
#                 sample = np.round(np.random.uniform(0, max_add), 1)
#                 # print(f'max_add {max_add} <= time_remaining {time_remaining}')
#             if max_add > time_remaining and aa != pieces - 1:
#                 sample = np.round(np.random.uniform(0, time_remaining), 1)
#             add_list.append(sample)
#             # print(f'On the {aa+1} loop I added {sample} to give {add_list}, tr {time_remaining - sample}')
#             time_remaining = time_remaining - sample
#         zip_list = zip(list, np.abs(add_list))
#         cut_list = [x + y for (x, y) in zip_list]
#         print(f'Generated a list of cuts {cut_list} seconds in duration.')
#
#     return cut_list
#
#
#
#
# cut_dir = []
# for bb in range(1):
#     listy = segment_gen(4,0.5,2)
#     perms = permutations(listy[0])
#     for cc in perms:
#         cut_dir.append(listy)
#     print(f'{listy} : sum is {np.sum(listy)}')


#
#
# def segment_gen(len_sec, min_seg=0.5, max_seg=2.0):
#     if max_seg >= len_sec:
#         add_list = []
#         print('The longest segment length you want is too long for your sound file!')
#     elif min_seg < 0:
#         add_list = []
#         print('The minimum segment cannot be below zero!')
#     elif max_seg - min_seg < 0:
#         add_list = []
#         print('Minimum segment length must be smaller than the maximum segment length!')
#     else:
#         pieces = int(len_sec)
#         list, add_list = [min_seg] * pieces, []
#         ii, time_remaining, total, max_add = 1, len_sec - np.sum(list), 0, max_seg - min_seg
#         # print(f'Length {len_sec}, Remaining {time_remaining}, Min {min_seg}, Pieces {pieces}, Max to add {max_add}')
#         for aa in range(pieces):
#             if time_remaining < max_add and aa == pieces - 1:
#                 sample = np.round(time_remaining,1)
#                 # print(f'I used this to add {time_remaining}')
#             elif time_remaining > max_add and aa == pieces - 1:
#                 print('There is an error from the random generator and you might consider rerunning this.')
#             if max_add <= time_remaining and aa != pieces - 1:
#                 sample = np.round(np.random.uniform(0, max_add), 1)
#                 # print(f'max_add {max_add} <= time_remaining {time_remaining}')
#             if max_add > time_remaining and aa != pieces - 1:
#                 sample = np.round(np.random.uniform(0, time_remaining), 1)
#             add_list.append(sample)
#             # print(f'On the {aa+1} loop I added {sample} to give {add_list}, tr {time_remaining - sample}')
#             time_remaining = time_remaining - sample
#         zip_list = zip(list, np.abs(add_list))
#         cut_list = [x + y for (x, y) in zip_list]
#         print(f'Generated a list of cuts {cut_list} seconds in duration.')
#
#     return cut_list
#
#
# for ss in range(10):
#     list, max_add = segment_gen(4,0.5,2)
#     print(f'Sum of {np.sum(list)}, it should be {max_add}')
#
#
#
# print(f'{list} with sum of {np.sum(list)}, it should be {max_add}')

#
#
#         list[aa] = list[aa] + sample
#             print(f'I added {sample} to the {aa} position to get {list[aa]}')
#             print(f'The time remaining is {int(time_remaining-sample)}')
#             if int(time_remaining - sample) == 0:
#                 break
#             time_remaining = int(time_remaining - sample)
#     return list
#
# for aa in range(1):
#     listy = segment_gen(3,3)
#     print(f'{listy} : sum is {np.sum(listy)}')
#
#
# def segment_gen(len_sec,pieces=4,min_seg=0.5,max_seg=2.0):
#     list, ii, remaining, total = [], 1, len_sec, 0
#     print(f'len_sec = {len_sec}, pieces = {pieces}')
#     while ii <= pieces: #or total <= len_sec:
#         sample = np.round(np.random.uniform(min_seg, max_seg), 1)
#         print(f'The {ii} selected number is {sample}')
#         if len(list) == pieces:
#             final_sample = len_sec - total
#             list.append(final_sample)
#             print(f'The length of the list now exceeds pieces by 1')
#             break
#         if ii == 0:
#             total = sample
#             remaining = len_sec - sample
#             list.append(sample)
#             print(f'This is the first sample, {total}, with {remaining} remaining from {len_sec}')
#             # print(f'first total {total} : remaining {remaining}')
#         else:
#             total = total + sample
#             remaining = remaining - sample
#             # print(f'{ii} total: {total} : remaining {remaining}')
#             if remaining >= 0.5:
#                 list.append(sample)
#                 print(f'The remainder was greater than {min_seg}, so it was added to the list')
#                 #might be an error here, maybe append remaining not sample?
#             else:
#                 # if total >= len_sec:
#                 print(f'Remainder was less than {min_seg} and total greater than {len_sec}, generating new sample')
#                 sample = sample - np.abs(remaining)
#                 remaining = remaining - sample
#                 # print(f'new sample:{sample}, remaining {remaining}')
#                 if 0.5 <= sample <= 2:
#                     list.append(sample)
#                     print(f'The new sample is the appropriate length of {sample}, it was added')
#                 # else:
#                 #     print(f'Remainder was less than {min_seg} and total less than {len_sec}, fixing it')
#                 #
#         ii += 1
#     if np.sum(list) < len_sec:
#         if (len_sec - np.sum(list)) >= 0.5:
#             list.append(len_sec - np.sum(list))
#             print("I did something out here in the first one")
#         else:
#             if np.sum(list) < len_sec:
#                 if list[-1] + remaining <= 2:
#                     list[-1] = list[-1] + remaining
#             print('I did something out here in the second one')
#                 # elif list[-2] + remaining <= 2:
#                 #     list[-2] = list[-2] + remaining
#                 # elif list[-3] + remaining <= 2:
#                 #     list[-3] = list[-3] + remaining
#     round_list = [round(num, 1) for num in list]
#     return round_list
#
# dict = {}
# for aa in range(5):
#     listy = segment_gen(3,3)
#     dict[aa] = listy
#     print(f'{listy} : sum is {np.sum(listy)}')



def sound_assigner(trial_letter, count, start, cats, seg_label_list,
                   seg_dict, sound_dict, sort_df, bin_length,
                   fs):
    cat = random.randrange(1, cats + 1, 1)
    good_list = []
    for hh in seg_label_list:
        if f'CLASS-0{cat}' in hh:
            good_list.append(hh)
    random.shuffle(good_list)
    add_label = random.choice(good_list)
    add_wave = seg_dict[add_label]

    seg_len = len(add_wave) / fs

    if 'A' in trial_letter:  #The first segment gets randomly placed, sound B will be relative to it to ensure overlap
        max_jitter_length = np.round(bin_length - seg_len, 1)
        jitter = np.round(np.random.uniform(0, max_jitter_length), 1)
        on_time = start + jitter
        pre = np.zeros(int(jitter * fs))
        post = np.zeros(int((bin_length - (jitter + seg_len)) * fs))

    if 'B' in trial_letter:
        df = sort_df.copy().set_index(['Label'])
        A_on, A_off = df.loc[f'A{count + 1}']['On'], df.loc[f'A{count + 1}']['Off']

        max_start = np.round(A_off - 0.5, 1)
        if max_start + seg_len > start + bin_length:
            max_start = bin_length * (count + 1) - seg_len
        min_start = np.round((A_on + 0.5) - seg_len, 1)
        if min_start < 0:
            min_start = 0
        max_range = np.round(max_start - A_on, 1)
        min_range = np.round(min_start - A_on, 1)

        jitter_values, jit = [i for i in np.arange(-bin_length,bin_length,0.25)], 0
        jitter_list = [i for i in jitter_values if min_range < i < max_range]
        jitter = random.choice(jitter_list)
        on_time = A_on + jitter
        pre = np.zeros(int((on_time - start) * fs))
        post = np.zeros(int(((bin_length + start) - (on_time + seg_len)) * fs))

    off_time = on_time + seg_len
    wave = np.concatenate((pre, add_wave, post))
    if len(wave) != int((bin_length * fs)):
        for ab in range(int((bin_length * fs) - len(wave))):
            wave = np.append(wave, 0)

    sound_dict[f'{trial_letter}{count + 1}'] = wave
    # sound_list.append(f'A{bn+1}')
    row = {'Label': f'{trial_letter}{count + 1}', 'Name': 'STIM_' + add_label.split('_')[0], 'Class': f"CLASS_0{cat}", 'On': np.round(on_time, 2),
           'Off': np.round(off_time, 2), 'Segment': add_label.split('_')[1]}
    sort_df = sort_df.append(row, ignore_index=True)

    return wave, sound_dict, sort_df
