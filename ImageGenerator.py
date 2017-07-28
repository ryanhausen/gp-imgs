#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:48:45 2017

@author: ryanhausen
"""
import numpy as np
import json

from GaussianProcess import GP, RBF
from DataTools import _nmpy_decode, get_noise, get_sr, get_param_from_dist
from ImageTools import radial_frame, fill_radial_frame, effective_radius, img_center
from ImageTools import trans_to_origin, trans_from_origin, scale_image, PIL_tuple
from ImageTools import axis_ratio
import ImageTools as it
from PIL import Image
from astropy.io import fits

def generate_batch(batch_size):
    # x [batch_size, 84, 84, 1]
    # y [batch_size, 5]

    # disk, spheroid distribution in the measured set - .785
    morph = np.random.binomial(1, .5, batch_size)

    batch_x, batch_y = [], []
    for i in range(batch_size):
        if morph[i]:
            batch_x.append(generate_disk())
            batch_y.append(np.array([1.0, 0.0, 0.0, 0.0, 0.0]))
        else:
            batch_x.append(generate_spheroid())
            batch_y.append(np.array([0.0, 1.0, 0.0, 0.0, 0.0]))

    return np.array(batch_x), np.array(batch_y)

def generate_disk(return_fits=False):
    return _generate_image('disk', return_fits)

def generate_spheroid(return_fits=False):
    return _generate_image('spheroid', return_fits)

def generate_unknown(return_fits=False):
    img = [get_noise(c, (84,84)) for c in ['h', 'j', 'v', 'z']]
    return np.dstack(img)

def _sbp_predict_on(morph, x):
    # restore a pretrained GP and kernel
    disk_params = [4.22574589, 2.82598596]
    sph_params = [4.12495443, 2.74516939]
    params = disk_params if morph=='disk' else sph_params
    kernel = RBF(params)
    gp_sbp = GP(kernel)
    gp_sbp.restore(f'{morph}_gp_model.json')

    return gp_sbp.sample(x)

def _generate_image_from(morph, ar, re, ie, seed=None):
    shape = (84, 84)

    seed = seed if seed else np.random.randint(0, 2**32-1)
    # create a matrix in units in effective radaii
    img = radial_frame(shape[0], shape[1], shape[0] // 2, shape[1] // 2) / re


    # the unique radial values to be predicted on by the gaussian process
    x = np.sort(np.unique(img))
    x = x[x < 10.0]
    x = x[:, np.newaxis]

    np.random.seed(seed)
    sbp = _sbp_predict_on(morph, x)
    sbp = 10**sbp[0, :]
    sbp = {r:v for r, v in zip(x[:,0], sbp)}

    h = fill_radial_frame(img.copy(), sbp) * ie

    # validate and adjust the image if the effective radius doesn't match the
    # desired effective radius
    #src_map = (img<(3*re)).astype(np.bool)
    src_map = np.ones_like(h, dtype=np.bool)
    measured_re, measured_ie = effective_radius(h, src_map, return_ie=True)

    import matplotlib.pyplot as plt
    x, y = it.sbp(h, src_map)
    plt.plot(x, y)
    plt.show()
    print(f'Wanted ie: {ie} got {measured_ie}\nWanted re: {re} got {measured_re}')

    if measured_re != re:
        re_ratio = measured_re / re
        img *= re_ratio

        x = np.sort(np.unique(img))
        x = x[x < 10.0]
        x = x[:, np.newaxis]

        np.random.seed(seed)
        sbp = _sbp_predict_on(morph, x)
        sbp = 10**sbp[0, :]
        sbp = {r:v for r, v in zip(x[:,0], sbp)}

        h = fill_radial_frame(img.copy(), sbp) * ie

    x, y = it.sbp(h, src_map)
    plt.plot(x, y)
    plt.show()


    measured_re, measured_ie = effective_radius(h, src_map, return_ie=True)
    print(f'Wanted ie: {ie} got {measured_ie}\nWanted re: {re} got {measured_re}')

    h *= ie / measured_ie

    x, y = it.sbp(h, src_map)
    plt.plot(x, y)
    plt.show()

    measured_re, measured_ie = effective_radius(h, src_map, return_ie=True)
    print(f'Wanted ie: {ie} got {measured_ie}\nWanted re: {re} got {measured_re}')

    # set the axis ratio
    # apply the drawn axis ratio
    a = np.sqrt(re**2 * ar) / re
    b = np.sqrt(re**2 / ar) / re

    trans = trans_to_origin().dot(scale_image(a, b)).dot(trans_from_origin())
    trans = PIL_tuple(trans)
    tmp = Image.fromarray(h)
    tmp = tmp.transform(h.shape, Image.AFFINE, trans, resample=Image.BICUBIC)
    h = np.asarray(tmp)
    h.setflags(write=1)

    print(f'Wanted ar: {ar} got {axis_ratio(h, src_map)}')

    return h, x, img

def _get_other_bands(h, x, img_re, morph):
    # get the color profile for the image
    params = [1.0, 3.0]
    kernel = RBF(params)
    gp_color = GP(kernel)
    gp_color.restore(f'{morph}_gp_colors.json')

    color_ratios, _ = gp_color.predict(x)
    j, v, z, = h.copy(), h.copy(), h.copy()

    # apply the color profile to make the image in the other bands
    # apply the axis ratio
    for color, img in [(color_ratios[:, 0], j),
                       (color_ratios[:, 1], v),
                       (color_ratios[:, 2], z)]:
        color = np.exp(color)
        multiplier = {r:v for r, v in zip(x[:,0], color)}
        ratio = fill_radial_frame(img_re.copy(), multiplier)
        img *= ratio

    return j, v, z

def _generate_image(morph, return_fits,
                    axis_ratio=None,
                    effective_radius=None,
                    intensity_re=None,
                    snr=None,
                    seed=None):
    header = {'morph':morph}

    ie = get_param_from_dist(morph, 'ie')
    re = get_param_from_dist(morph, 're')
    ar = get_param_from_dist(morph, 'ar')

    ie = intensity_re if intensity_re else ie
    re = effective_radius if effective_radius else re
    ar = axis_ratio if axis_ratio else ar

    header['axisr'] = ar
    header['Re'] = re

    h, x, img_re = _generate_image_from(morph, ar, re, ie, seed)

    j, v, z = _get_other_bands(h, x, img_re, morph)

    # add noise
    # scale noise
    sr_ratio = snr if snr else get_sr(morph)
    for color, img in [('h', h), ('j', j), ('v', v), ('z', z)]:
            # not sure I want this in the larger namespace
        def rms(collection):
            try:
                return np.sqrt((1/len(collection))*(np.square(collection).sum()))
            except ZeroDivisionError as e:
                print(f'{e} lenth of array {len(collection)}')
                return 0.0


        header[f'sr-{color}'] = sr_ratio[color]
        noise = get_noise(color, img.shape)
        n_mean = rms(noise.flatten())

        cx, cy = np.where(img==np.max(img))
        cx, cy = cx[0], cy[0]
        radial_grid = radial_frame(img.shape[0], img.shape[1], cx, cy)
        signal = img[radial_grid <= re].sum()


        sr = signal / n_mean
        img *= (sr_ratio[color] /  sr)
        #noise = noise * (sr / sr_ratio[color])
        img += noise

    if return_fits:
        header['dim0'] = 'h,j,v,z'
        hdr = fits.Header()
        for k in header.keys():
            hdr[k] = header[k]
        img = np.dstack((h, j, v, z))

        return fits.PrimaryHDU(header=hdr, data=img)
    else:
        return np.dstack((h, j, v, z))
