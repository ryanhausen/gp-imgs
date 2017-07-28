import numpy as np

import ImageTools as it
import matplotlib.pyplot as plt

from GaussianProcess import GP, RBF
from DataTools import disk_sbp, spheroid_sbp, get_noise, get_sr
from PIL import Image

# get H data
x, y16, y, y84 = disk_sbp('h')

# get an ar and re from the measured distribution

# restore a pretrained GP and kernel
l, f = 4.22574589, 2.82598596
kernel = RBF((l,f))
gp_sbp = GP(kernel)
gp_sbp.restore()

# a radially frame, that is normalized to the desired effective radius
_re = 5 * 0.06
rad_frame = (it.radial_frame(84,84,41,41))
rad_frame /= _re

# extract an x to predict on
_x = np.sort(np.unique(rad_frame))
#_x = _x[_x < 4.0]
_x = _x[:,np.newaxis]

# A random seed needs to be set so that the same GP sample can be drawn again if needed
seed = 0
np.random.seed(seed)
sample = gp_sbp.sample(_x, 1)
sample = np.exp(sample[0,:])

# create the disk image in spheroid form for H band
sbp = {r:v for r, v in zip(_x[:,0], sample)}
h = it.fill_radial_frame(rad_frame.copy(), sbp)

# measure the re and and if it is different change the adjust the radially image
# to make drawn sbp fit the input re
re = it.effective_radius(h, np.ones_like(h, dtype=bool))
print(f'Input re:{_re} Measured re:{re}')
if _re != re:
    re_ratio = re / _re
    print(f'f:{re_ratio}')
    rad_frame *= re_ratio
    _x = np.sort(np.unique(rad_frame))
    _x = _x[_x < 4.0]
    _x = _x[:,np.newaxis]

    np.random.seed(seed)
    sample = gp_sbp.sample(_x, 1)
    sample = np.exp(sample[0,:])

    sbp = {r:v for r, v in zip(_x[:,0], sample)}
    h = it.fill_radial_frame(rad_frame.copy(), sbp)
    re = it.effective_radius(h, np.ones_like(h, dtype=bool))
    print(f'Input re:{_re} Remeasured re:{re}')

# apply color ratios before reshaping into disk
l, f = 1.0, 3.0
kernel = RBF((l, f))
gp_color = GP(kernel)
gp_color.restore('gp_colors.json')

color_sample, _ = gp_color.predict(_x)
j, v, z = h.copy(), h.copy(), h.copy()

for color, img in [(color_sample[:,0], j),
                   (color_sample[:,1], v),
                   (color_sample[:,2], z)]:
    multiplier = {r:v for r, v in zip(_x[:,0], color)}
    ratio = it.fill_radial_frame(rad_frame.copy(), multiplier)
    img *= ratio

# transform to disk
axis_ratio = 0.5
r = _re
a = np.sqrt(r**2 * axis_ratio) / r
b = np.sqrt(r**2 / axis_ratio) / r

print(f'a={a} b={b}')

trans = it.trans_to_origin().dot(it.scale_image(a, b)).dot(it.trans_from_origin())
trans = it.PIL_tuple(trans)

sr_ratio = get_sr('disk')
for n, i in [('h', h), ('j', j), ('v', v), ('z', z)]:
#    plt.figure()
#    plt.title(f'{n} band disk image')
#    plt.imshow(i, cmap='gray')
#    plt.axis('off')
#    plt.gca().invert_yaxis()

    tmp = Image.fromarray(i)
    tmp = tmp.transform(i.shape, Image.AFFINE, trans, resample=Image.BICUBIC)
    tmp = np.asarray(tmp)

    if n == 'h':
        print(f'Disk re:{it.effective_radius(tmp, np.ones(i.shape, dtype=np.bool))}')
        print(f'Disk ar:{it.axis_ratio(tmp, np.ones(i.shape, dtype=np.bool))}')

    plt.figure()
    plt.title(f'{n} band disk image')
    plt.imshow(tmp, cmap='gray')
    plt.axis('off')
    plt.gca().invert_yaxis()

    # add noise
    noise = get_noise(n, [84,84])
    n_mean = noise.mean()
    i_mean = tmp.max()
    sr = i_mean / n_mean
    #tmp = tmp / (sr/sr_ratio[n])
    noise = noise * (sr/sr_ratio[n])
    plt.figure()
    plt.title(f'{n} band disk image with noise. Ratio:{sr_ratio[n]}')
    plt.imshow(tmp + noise, cmap='gray')
    plt.axis('off')
    plt.gca().invert_yaxis()

    n_mean = noise.mean()
    i_mean = tmp.max()
    sr = i_mean / n_mean
    print(sr, sr_ratio[n])


plt.figure()
plt.title('Color Ratio GP Predicted Mean')
not_h = ['j','v','z']
for i in range(3):
    plt.plot(_x, color_sample[:,i], label=not_h[i])
plt.legend()

plt.figure()
plt.title('H Band Surface Brightness Profile Sample')
plt.plot(x, y, label='Measured Data', color='b')


plt.plot(_x, sample, label='Sample', color='r')
plt.legend()

plt.show()