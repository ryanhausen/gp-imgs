import os

import numpy as np
import matplotlib.pyplot as plt

import ImageGenerator as ig
import DataTools as dt
import ImageTools as it

epsilon = lambda a: 0 if np.min(a) > 0 else (abs(np.min(a)) + 1e-3)

home = os.getenv('HOME')
img_dir = os.path.join(home, 'Documents/astro_data/our_images')
img_name = 'deep2_10042'
img_name = 'ers2_12827'
bands = ['h', 'j', 'v', 'z']
log_scale = False
orig_images = False

img_id, img, segmap = dt._id_img_seg(img_dir, img_name, bands, log_scale, orig_images)
src_map = segmap==img_id

whole_sbp = lambda i: it.radial_bin_image(i, src_map, re_normed=True, ie_normed=True)
whole_ar = lambda i: round(it.axis_ratio(i, src_map), 3)
whole_re = lambda i: round(it.effective_radius(i, src_map), 3)
# measure the effective radius and the axis ratio for the image
re = it.effective_radius(img[0,:,:], src_map)
ar = it.axis_ratio(img[0,:,:], src_map)
print(f'Real Measured Re:{re} Ar:{ar}')

synth = ig._generate_image('spheroid', False, axis_ratio=ar, effective_radius=re, seed=1)
re = whole_re(synth[:,:,0])
ar = whole_ar(synth[:,:,0])
print(f'Synth Measured Re:{re} Ar:{ar}')

f, axes = plt.subplots(4, 2, figsize=(10,10))
synth_col, real_col = 0, 1
plt.suptitle('<- Synth -- Real ->')
for i in range(4):
    band_synth = synth[:,:,i] + epsilon(synth[:,:,i])

    axes[i][synth_col].imshow(band_synth, cmap='gray')
    axes[i][synth_col].set_title(f'{bands[i]} Re:{whole_re(band_synth)} Axis-Ratio{whole_ar(band_synth)} ')

    band_img = img[i,:,:] + epsilon(img[i,:,:])
    axes[i][real_col].imshow(band_img, cmap='gray')
    axes[i][real_col].set_title(f'{bands[i]} Re:{whole_re(band_img)} Axis-Ratio:{whole_ar(band_img)} ')

plt.tight_layout()


f, axes = plt.subplots(4, 2, figsize=(10,10))
synth_col, real_col = 0, 1
plt.suptitle('<- Synth -- Real ->')
for i in range(4):
    band_img = img[i,:,:] + epsilon(img[i,:,:])

    img_sbp = whole_sbp(band_img)
    x, y = [], []
    for r in sorted(img_sbp.keys()):
        x.append(r)
        y.append(img_sbp[r])

    axes[i][real_col].plot(x, y)
    axes[i][real_col].set_xlim((0, 2))
    axes[i][real_col].set_ylim((-1, 20))
    axes[i][real_col].set_title(f'{bands[i]} Re:{whole_re(band_img)} Axis-Ratio{whole_ar(band_img)} ')


    band_synth = synth[:,:,i] + epsilon(synth[:,:,i])

    synth_sbp = whole_sbp(band_synth)
    x, y = [], []
    for r in sorted(synth_sbp.keys()):
        x.append(r)
        y.append(synth_sbp[r])

    axes[i][synth_col].plot(x, y)
    axes[i][synth_col].set_xlim((0, 2))
    axes[i][synth_col].set_ylim((-1, 20))
    axes[i][synth_col].set_title(f'{bands[i]} Re:{whole_re(band_synth)} Axis-Ratio:{whole_ar(band_synth)} ')


plt.tight_layout()

plt.show()






