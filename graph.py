import numpy as np
import pandas as pd
from copy import deepcopy
import os
import bisect
from astropy.io import fits

from scipy.optimize import fmin

import matplotlib.pyplot as plt

bands = ['h','j','v','z']


for morph in ['disk', 'spheroid']:
    fig, axes = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,10))
    axes = [element for tupl in axes for element in tupl]
    for b in bands:
        band_vals = np.loadtxt('{}_{}_fittedvalues.txt'.format(morph, b))
        axes[bands.index(b)].scatter(band_vals[0,:], band_vals[1,:], color='rgby'[bands.index(b)], label='{} band'.format(b))
        axes[bands.index(b)].set_title('{} band'.format(b))
        #axes[bands.index(b)].set_ylim([0,5])
        #axes[bands.index(b)].set_xlim([-.90,.5])

    plt.suptitle(morph, fontsize=24)
    fig.text(0.5, 0.04, 'Re', ha='center', va='center', fontsize=24)
    fig.text(0.06, 0.5, 'Ie', ha='center', va='center', rotation='vertical', fontsize=24)
   

plt.show()
