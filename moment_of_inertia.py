# =============================================================================
# https://alyssaq.github.io/2015/computing-the-axes-or-orientation-of-a-blob/
# https://en.wikipedia.org/wiki/Image_moment
# http://stackoverflow.com/questions/9005659/compute-eigenvectors-of-image-in-python
# =============================================================================

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

import ImageTools as it

def get_img(band):
    return fits.getdata(f'./imgs/GDS_deep2_10064_{band}.fits'), \
           fits.getdata(f'./imgs/GDS_deep2_10064_segmap.fits')

def raw_moment(data, x, y, i_order, j_order):
    return (data * x**i_order * y**j_order).sum()

def moments_cov(data, x, y):
    m00 = data.sum()
    m10 = raw_moment(data, x, y, 1, 0)
    m01 = raw_moment(data, x, y, 0, 1)
    x_centroid = m10 / m00
    y_centroid = m01 / m00
    u11 = (raw_moment(data, x, y, 1, 1) - x_centroid * m01) / m00
    u20 = (raw_moment(data, x, y, 2, 0) - x_centroid * m10) / m00
    u02 = (raw_moment(data, x, y, 0, 2) - y_centroid * m01) / m00
    cov = np.array([[u20, u11], [u11, u02]])
    return cov

def get_re_pix(img, xs, ys, cx, cy):
    r_vals = {}
    r_coords = {}
    for x, y in zip(xs, ys):
        r = np.sqrt((x-cx)**2 + (y-cy)**2)*0.06
        r_vals.setdefault(r, []).append(img[x, y])
        r_coords.setdefault(r, []).append((x, y))

    tmp = 0.0
    coords = []
    img_map = np.zeros(img.shape, dtype=int)
    Re = 0.0
    for r in sorted(r_vals.keys()):
        tmp += sum(r_vals[r])
        coords.extend(r_coords[r])
        for x, y in r_coords[r]:
            img_map[x, y] = 1
        if tmp >= I_e:
            Re = r
            break

    x, y = zip(*coords)
    x, y = np.array(x), np.array(y)

    return x, y, img_map.astype(bool), Re

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
axes = [ax1, ax2, ax3, ax4]

img_id = 10064
bands = ['h', 'j', 'v', 'z']
for band in bands:
    img, segmap = get_img(band)
    I_tot = img[segmap == img_id].sum()
    I_e = I_tot / 2

    y, x = np.where(segmap == img_id)

    # find the effective radius
    cy, cx = np.where(img == img[segmap == img_id].max())
    cy, cx = cy[0], cx[0]

    _, _, Re_map, Re = get_re_pix(img, x, y, cx, cy)
    img_map = segmap == img_id


    cov = moments_cov(img[img_map], x, y)
    evals, evecs = np.linalg.eig(cov)
    pairs = {}
    for i in range(2):
        pairs[evals[i]] = evecs[:, i]

    eval_maj = evals.max()
    eval_min = evals.min()

    major_x, major_y = pairs[eval_maj]  # Eigenvector with largest eigenvalue
    minor_x, minor_y = pairs[eval_min]

    max_coords = np.where(img == img.max())
    x, y = max_coords[0][0], max_coords[1][0]

    print(f'Cov:{cov}')
    print(f'Major Eigenvector{pairs[evals.max()]}')
    print(f'Minor Eigenvector{pairs[evals.min()]}')
    print(f'Axis Ratio:{np.sqrt(evals.min()/evals.max())}')
    print(f'Axis Ratio:{it.axis_ratio(img, segmap==img_id)}')

    scale = np.sqrt(eval_maj) * np.array([-1.0, 1.0])
    major_x_line = scale * major_x + x
    major_y_line = scale * major_y + y

    scale = np.sqrt(eval_min) * np.array([-1.0, 1.0])
    minor_x_line = scale * minor_x + x
    minor_y_line = scale * minor_y + y

    theta = 0.5 * np.arctan((2 * cov[0,1])/(cov[0,0] - cov[1,1]))
    theta = np.rad2deg(theta)
    print(f'Theta:{theta}')

    r = Re/0.06
    Q = np.sqrt(evals.max())/np.sqrt(evals.min())
    a = np.sqrt(r**2 * Q)
    b = np.sqrt(r**2 / Q)

    print(f'Re:{Re} R:{Re/0.06}')
    print(f'a:{a} b:{b}')

    ell = Ellipse((cx,cy), 2*a, 2*b, angle=theta, lw=2.0)
    ell.set_alpha(.4)
    ell.set_edgecolor('g')
    ell.set_facecolor('none')

    ax = axes[bands.index(band)]
    ax.axis('off')
    ax.imshow(img, cmap='gray', alpha=0.5)
    ax.add_patch(ell)
    ax.invert_yaxis()

plt.tight_layout()


plt.show()
