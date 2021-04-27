import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import resample

from nems.analysis.gammatone.gtgram import gtgram

factor = 441

kind = 'Textures'
kind = 'Marms'
kind = 'Transients'

##########Foreground3####################
#Threshold == 0.15 #z
#Threshold = 0.025 #norm
kind = "Ferrets"
ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"

name = 'cat668_rec2_ferret_fights_Jasmine-Violet001_excerpt2'
# little_name = '01_Fight Squeak'
# start = 180
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[22050:] = one_sec[21609:-441]

# name = '00cat668_rec1_ferret_fights_Athena-Violet001_excerpt1'  #02Fight 610-1610 Fight
#02Fight - start:66
#OLD WAYspec[:,38:] = spec[:,35:-3] #spec[:,90:] = 0
#FINAL    one_sec[21609:] = one_sec[17199:-4410]
    # one_sec[17640:22050] = 0

name = 'cat668_rec7_ferret_oxford_male_chopped_excerpt2'  #0-1000 Gobble
# little_name = '04_Gobble'
# start = 9
# one_sec = W[start:start + int((len(W) / 4)) + (3 * factor)]
# one_sec = one_sec.copy()
# one_sec[22050:-1323] = one_sec[23373:]
# one_sec = one_sec[:-1323]

name = 'cat668_rec5_ferret_kits_0p1-2-3-4-5-6_excerpt2'
# little_name = '05_Kit Groan'
# start = 262
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat668_rec3_ferret_kits_51p9-8-10-11-12-13-14_excerpt4'  #50-550, 1150-1650 AND Low 1990-1490, 2450-2950, Mid, Low
# little_name = '07_Kit High'
# start, start2 = 59, 118
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]

name = 'cat668_rec6_ferret_kits_18p1-2-3-5-7-8_excerpt7'    #730-1730 Whine
# little_name = '08_Kit Whine'
# start, start2 = 76, 129
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]
################END FERRET########################

#######TRANSIENTS#####
kind = "Transients"
ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"

name = 'cat24_rec1_desk_bell_freesound_burkay_bell_hotel_desk_excerpt1'
# little_name = '11_Bell'
# start, start2 = 6, 196
# start2 = 196 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]

name = 'cat30_rec1_blacksmith_hammering_excerpt1'
# little_name = '12_Blacksmith'
# start, start2 = 185, 273
# start2 = 273 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[15435:] = 0
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]
# one_sec[39690:] = 0

name = 'cat33_rec1_bluegrass_excerpt1'
# little_name = '13_Banjo'
# start = 149
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat39_rec1_branch_snapping_sound-ideas-37-96_excerpt1'
# little_name = '14_Branch'
# start, start2 = 71, 140
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[15435:] = 0
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]
# one_sec[39690:] = 0

name = 'cat66_rec1_cash_register_excerpt1'
# little_name = '15_Cash Register'
# start, start2 = 168, 211
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[15435:] = 0
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]
# one_sec[39690:] = 0

name = 'cat536_rec1_fire_crackers_excerpt1'
# little_name = "16_Fire Cracker"
# start, start2 = 51, 322
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[15435:] = 0
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]
# one_sec[39690:] = 0

name = 'cat387_rec1_typing_excerpt1'
# little_name = '17_Typing'
# start, start2 = 155, 193
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]
# one_sec[20727:22050] = one_sec[20286]
# one_sec[22050:] = one_sec[21609:-441]

name = '00cat401_rec1_walking_with_heels_excerpt1'
# little_name = "18_Heels"
# start, start2, start3, start4 = 57, 113, 170, 221
# start2, start3, start4 = start2 * factor, start3*factor, start4*factor
# one_sec = W[start:start + int((len(W) / 16))]
# q2 = W[start2:start2 + int((len(W) / 16))]
# q3 = W[start3:start3 + int((len(W) / 16))]
# q4 = W[start4:start4 + int((len(W) / 16))]
# one_sec = np.concatenate([one_sec, q2,q3,q4])

name = 'cat211_rec1_keys_jingling_excerpt1'
# little_name = "19_Keys"
# start = 226
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[9261:] = one_sec[8820:-441]

name = 'cat77_rec1_chickens_clucking_excerpt1'
# little_name = "20_Chickens"
# start = 172
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[22050:] = one_sec[21609:-441]

name = '00cat172_rec1_geese_excerpt1'
# little_name = "21_Geese"
# start, start2 = 12, 87
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]

name = 'cat413_rec1_wolves_howling_excerpt1'
# little_name = "22_Wolf"
# start = 0
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat560_rec1_chipmunk_excerpt1'
# little_name = "23_Chipmunk"
# start, start2 = 5, 79
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]

name = 'cat561_rec1_dolphin_excerpt1'
# little_name = "24_Dolphin"
# start = 271
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat604_rec1_xylophone_excerpt1'
# little_name = "25_Xylophone"
# start = 163
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[17199:] = one_sec[16758:-441]

name = 'cat603_rec1_woodblock_excerpt1'
# little_name = "26_Woodblock"
# start, start2 = 125, 172
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[17640:] = 0
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]

name = 'cat361_rec1_tambourine_excerpt1'
# little_name = "27_Tambourine"
# start = 139
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[41895:] = 0

name = 'cat44_rec1_bugle_excerpt1'
# little_name = "28_Bugle"
# start = 234
# one_sec = W[start:start + int((len(W) / 4) + (2*factor))]
# one_sec = one_sec.copy()
# one_sec[7938:-882] = one_sec[8820:]
# one_sec = one_sec[:-882]
# # one_sec[40572:] = 0

name = 'cat67_rec1_castenets_excerpt1'
# little_name = "29_Castinets"
# start = 31
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[40572:] = 0

name = 'cat111_rec1_dice_excerpt1'
# little_name = "30_Dice"
# start = 151
# one_sec = W[start:start + int((len(W) / 4) + (factor*1))]
# one_sec = one_sec.copy()
# one_sec[14553:-441] = one_sec[14994:]
# one_sec = one_sec[:-441]

name = 'cat272_rec1_ping_pong_excerpt1'
# little_name = "31_Pingpong"
# start, start2 = 87, 127
# start2 = start2 * factor
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
# one_sec[15435:] = 0
# half_sec = W[start2:start2 + int((len(W) / 8))]
# one_sec[22050:] = half_sec[:]

name = '00cat414_rec1_woman_speaking_excerpt1'
# little_name = "32_Woman1"
# start = 204
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat10093_rec1_48u_c0303_excerpt1'
# little_name = "33_Man1"
# start = 5
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat10089_rec1_48q_c021e_excerpt1'
# little_name = "34_Man2"
# start = 67
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat10088_rec1_48p_c021d_excerpt1'
# little_name = "35_Woman2"
# start = 65
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat15000_rec1_speech-russian_0sec_excerpt1'
# little_name = "36_Russian"
# start = 116
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
###################################

#####Backgrounds - TEXTURES###############
kind = 'Textures'
ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"

name = '00cat78_rec1_chimes_in_the_wind_excerpt1'                             #88200
# little_name = "01_Chimes"
# start = 201
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat558_rec1_tire_rolling_on_gravel_excerpt1'                          #22000
# little_name = "02_Gravel"
# start = 49
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat565_rec1_insects_buzzing_excerpt1'                                 #132000
# little_name = "03_Insect Buzz"
# start = 75
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat287_rec1_rain_excerpt1'                                            #66100
# little_name = "04_Rain"
# start = 74
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat301_rec1_envsounds_rock-tumbling_sound-ideas-38-27_2sec_excerpt1'  #99000
# little_name = "05_Rock Tumble"
# start = 237
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = '00cat516_rec1_stream_excerpt1'                                        #50000
# little_name = "06_Stream"
# start = 165
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat368_rec1_thunder_excerpt1'                                         #4000
# little_name = "07_Thunder"
# start = 18
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat534_rec1_waterfall_excerpt1'                                       #121200
# little_name = "08_Waterfall"
# start = 269
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat403_rec1_waves_excerpt1'                                           #55000
# little_name = "09_Waves"
# start = 198
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat312_rec1_wind_excerpt1'                                            #105000
# little_name = "10_Wind"
# start = 0
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()
##############################

#########Potential Backgrounds##########
kind = 'Background'
ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/{kind}/"

dir = os.path.join(ROOT, "*.wav")
set = glob.glob(dir)
names = [nn.split('\\')[-1].split('.')[0] for nn in set]

name = 'cat140_rec1_hand_drill_sound-ideas-12-45_excerpt1'
# little_name = "11_Drill"
# start = 161
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat149_rec1_film_reel_freesounds_bone666138_excerpt1'
# little_name = "12_Film"
# start = 214
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat185_rec1_hairdryer_excerpt1'
# little_name = "13_Hairdryer"
# start = 59
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat23_rec1_bees_buzzing_excerpt1'
# little_name = "14_Bees"
# start = 49
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat254_rec1_classicalsymphony_mozart_symphony48-allegro_80sec_excerpt1'
# little_name = "15_OrchestraA"
# start = 155
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat254_rec2_classicalsymphony_schumann_symphony3-lebhaft_0sec_excerpt1'
# little_name = "16_OrchestraB"
# start = 130
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat255_rec1_orchestra_tuning_up_excerpt1'
# little_name = "17_Tuning"
# start = 52
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat256_rec1_classical-organ_handel_messiah-hallelujah-chorus_2sec_excerpt1'
# little_name = "18_Organ"
# start = 294
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat31_rec1_blender_excerpt1'
# little_name = "19_Bike"
# start = 261
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat291_rec1_rattlesnake_excerpt1'
# little_name = "20_Rattlesnake"
# start = 267
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat313_rec1_electric_sander_excerpt1'
# little_name = "21_Sander"
# start = 75
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat329_rec1_electric_shaver_excerpt1'
# little_name = "22_Shaver"
# start = 252
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat45_rec1_bulldozer_sound-ideas-11-41_excerpt1'
# little_name = "23_Bulldozer"
# start = 118
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat47_rec1_bus_excerpt1'
# little_name = "24_Bus"
# start = 157
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat581_rec1_tractor_excerpt1'
# little_name = "25_Tractor"
# start = 85
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat61_rec1_idling_sound-ideas-5-22_excerpt1'
# little_name = "26_Idling"
# start = 100
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat73_rec1_chainsaw_excerpt1'
# little_name = "27_Chainsaw"
# start = 204
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat87_rec1_coffee_machine_excerpt1'
# little_name = "28_Coffee"
# start = 293
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()

name = 'cat129_rec1_jackhammer_excerpt1'
# little_name = "29_Jackhammer"
# start = 299
# one_sec = W[start:start + int((len(W) / 4))]
# one_sec = one_sec.copy()




full_spec(ROOT, name, kind, 441)

#Define little_name

filepath = ROOT + name + '.wav'
SAVE_PATH = f'/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Clips/Tests/{kind}/{little_name}_{start}.wav'
start = start * factor
fs, W = wavfile.read(filepath)

#INSERT CODE HERE

little_name = "29_Jackhammer"
start = 299
one_sec = W[start:start + int((len(W) / 4))]
one_sec = one_sec.copy()



#FINAL:one_sec[19845:] = one_sec[18963:-882]
##

spec = gtgram(one_sec, fs, 0.02, 0.01, 48, 100, 8000)
# get_z(spec, name, threshold=0.15)
get_norm(spec, name, threshold=0.025)

wavfile.write(SAVE_PATH, fs, one_sec)

def get_norm(spec, name, threshold=0.05):
    '''Plots the spectrogram you're working with, with a panel below of the z-score
    and below that the difference in adjacent z-scores. Vertical lines are placed
    when the z-score difference between two bins exceeds a defined threshold.'''
    import numpy as np
    av = spec.mean(axis=0)
    big = np.max(av)
    norm = av/big
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].imshow(spec, aspect='auto', origin='lower')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[1].plot(norm)
    ax[1].set_ylabel('normalized')
    ax[0].vlines(50,3,45,color='white', ls=':')
    ticks = np.arange(0, spec.shape[1], 5)
    ax[1].set_xticks(ticks)
    di = np.diff(norm)
    ax[2].plot(di)
    ax[2].set_ylabel('difference')
    goods = []
    # for x in range(len(norm) - 1):
    #     if norm[x] < threshold and norm[x+1] > threshold:
    #         goods.append(x)
    for x in range(len(di)):
        if x == 0 and di[x] > threshold:
            goods.append(x)
        if x != 0:
            if di[x] > threshold and di[x-1] < threshold:
                goods.append(x)
    min,max = ax[1].get_ylim()
    ax[1].vlines(goods, min, max, ls=':')
    ax[2].vlines(goods, min, max, ls=':')
    fig.suptitle(f'{name} - threshold: {threshold}')

def get_z(spec, name, threshold=0.15):
    '''Plots the spectrogram you're working with, with a panel below of the z-score
    and below that the difference in adjacent z-scores. Vertical lines are placed
    when the z-score difference between two bins exceeds a defined threshold.'''
    import numpy as np
    av = spec.mean(axis=0)
    me = av.mean()
    sd = np.std(av)
    zz = (av - me) / sd
    di = np.diff(zz)
    fig, ax = plt.subplots(3,1, sharex=True)
    ax[0].imshow(spec, aspect='auto', origin='lower')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[1].plot(zz)
    ax[1].set_ylabel('z-score')
    ax[0].vlines(50,3,45,color='white', ls=':')
    ticks = np.arange(0, spec.shape[1], 5)
    ax[1].set_xticks(ticks)
    ax[2].plot(di)
    ax[2].set_ylabel('z-score difference')
    goods = np.where(di > threshold)[0].tolist()
    min,max = ax[1].get_ylim()
    ax[1].vlines(goods, min, max, ls=':')
    ax[2].vlines(goods, min, max, ls=':')
    fig.suptitle(f'{name}')


def full_spec(ROOT, name, kind, factor=441):
    '''Displays an entire 4s spectrogram and z-score for looking around to
    decide what chunk you want'''
    filepath = ROOT + name + '.wav'
    fs, W = wavfile.read(filepath)
    spec = gtgram(W, fs, 0.02, 0.01, 48, 100, 8000)
    # get_z(spec, name, threshold=0.15)
    get_norm(spec, name, threshold=0.025)