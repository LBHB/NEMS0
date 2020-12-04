import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from nems.analysis.gammatone.gtgram import gtgram
import pathlib as pl

GREG_ROOT = f"/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Figure Examples/"
filenames = os.listdir("/Users/grego/OneDrive/Documents/Sounds/Pilot Sounds/Figure Examples/")
titles = ['Full BG','Full FG','Full BG/Full FG','Half BG/Full FG','Half BG',
          'Half FG','Half BG/Half FG','Full BG/Half FG']

filepaths = [(GREG_ROOT + i) for i in filenames]

order = [0,6,1,4,3,7,5,2]
filepaths_reorder = [filepaths[i] for i in order]

fig, axes = plt.subplots(4, 2, sharex=True, sharey=True, squeeze=True)
axes = np.ravel(axes, order='F')

for count,(ax, file) in enumerate(zip(axes, filepaths_reorder)):
    fs, W = wavfile.read(file)
    spec = gtgram(W, fs, 0.02, 0.01, 64, 100, 8000)
    ax.imshow(spec, aspect='auto', extent=(0,99,100,8000), origin='lower')
    ax.set_xticks([0,49,99])
    ax.set_xticklabels([0,0.5,1])
    ax.set_yticks([250,8000])
    ax.set_yticklabels([100,8000])
    ax.set_title(titles[count], fontweight='bold')

plt.tight_layout()


