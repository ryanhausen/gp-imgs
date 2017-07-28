import os
import json

import numpy as np
import pandas as pd

from DataTools import safe_fits_open
import ImageTools as it

def rms(collection):
    try:
        return np.sqrt((1/len(collection))*(np.square(collection).sum()))
    except ZeroDivisionError as e:
        print(f'{e} lenth of array {len(collection)}')
        return 0.0

def pretty_label(lbl):
    if lbl=='ClSph':
        return 'spheroid'
    elif lbl=='ClDk':
        return 'disk'
    elif lbl=='ClIr':
        return 'irregular'
    elif lbl=='ClPS':
        return 'pointsource'
    elif lbl=='ClUn':
        return 'unknown'
    else:
        raise Exception("label must be 'ClSph','ClDk','ClIr','ClPS','ClUn'")

home = os.getenv('HOME')
img_dir = 'Documents/astro_data/orig_images'
img_dir = os.path.join(home, img_dir)

img_data = dict()
sn_data = dict()

# create the json file with all of the sn rations
img_files = os.listdir(img_dir)
total = len(img_files)
count = 1
if 'sr_vals.json' not in os.listdir():
    for img_file in sorted(img_files):
        print(count/total, end='\r')
        count += 1

        if '.fits' in img_file:
            name_split = img_file.replace('.fits', '').split('_')
            img_key = '_'.join(name_split[:-1])
            band_key = name_split[-1]
            raw = safe_fits_open(os.path.join(img_dir, img_file))
            if (raw.shape[0] != 84 or raw.shape[1] != 84):
                continue

            img_data.setdefault(img_key, dict())[band_key] = raw

            # do we have all of the bands and segmap
            if len(img_data[img_key].keys()) == 5:
                src_map = img_data[img_key]['segmap'] == int(img_key.split('_')[2])
                noise_map = img_data[img_key]['segmap'] == 0

                for b in ['h', 'j', 'v', 'z']:
                    img = img_data[img_key][b]
                    re = it.effective_radius(img, src_map)
                    cx, cy = np.where(img==np.max(img[src_map]))
                    cx, cy = cx[0], cy[0]
                    radial_grid = it.radial_frame(img.shape[0], img.shape[1], cx, cy)

                    # signal is the sum of the signal within the effective radius
                    s = img[radial_grid <= re].sum()
                    if (s <= 0):
                        print(s)

                    # noise is the RMS of the noise vlaues at >= 3*Re
                    nmap = np.logical_and(noise_map, (radial_grid >= 3*re))
                    n = rms(img[nmap].flatten())

                    sn_data.setdefault(img_key, dict())

                    sn_data[img_key][b] = str(s/n)

                img_data[img_key] = None

    with open('sr_vals.json', 'w') as f:
        json.dump(sn_data, f)
else:
    with open('sr_vals.json', 'r') as f:
        sn_data = json.load(f)

# create a dataframe with the s/n ratios and the labels
sn_df = pd.DataFrame.from_dict(sn_data, orient='index')

labels = pd.read_csv('labels.csv', header=0)
labels['label'] = labels.loc[:, ['ClSph','ClDk','ClIr','ClPS','ClUn']].idxmax(axis=1)
labels['label'] = labels['label'].apply(lambda x: pretty_label(x))

signal_to_noise = labels.join(sn_df, on='ID')
signal_to_noise = signal_to_noise.loc[:, ['ID', 'h', 'j', 'v', 'z', 'label']]

signal_to_noise.replace([np.inf, -np.inf], np.nan, inplace=True)
signal_to_noise.dropna(inplace=True)

signal_to_noise.to_csv('sr_dataframe.csv')











