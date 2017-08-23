#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
import os
import io
import json
import base64
from copy import deepcopy

import numpy as np
import pandas as pd
from astropy.io.fits import getdata
from scipy.stats import alpha, expon, norm

import ImageTools as it

bands = ['h', 'j', 'v', 'z']

# http://stackoverflow.com/questions/30698004/how-can-i-serialize-a-numpy-array-while-preserving-matrix-dimensions
def _nmpy_encode(a):
    with io.BytesIO() as mem:
        np.save(mem, a)
        mem.seek(0)
        return base64.b64encode(mem.read()).decode('utf-8')

def _nmpy_decode(a_str):
    with io.BytesIO() as mem:
        mem.write(base64.b64decode(a_str.encode('utf-8')))
        mem.seek(0)
        return np.load(mem)

def _id_img_seg(path, name, bands, log_scale, orig):
    _id = int(name.split('_')[1])
    _img = []
    _segmap = None

    if orig:
        _segmap = safe_fits_open(os.path.join(path, f'GDS_{name}_segmap.fits'))

        bands = np.atleast_1d(bands)
        if bands.shape[0] > 1:
            for b in bands:
                _img.append(safe_fits_open(os.path.join(path, f'GDS_{name}_{b}.fits')))

            _img = np.array(_img)
        else:
            _img = safe_fits_open(os.path.join(path, f'GDS_{name}_{bands[0]}.fits'))
    else:
        home = os.getenv('HOME')
        img_dir = 'Documents/astro_data/orig_images'
        seg_dir = os.path.join(home, img_dir)
        _segmap = safe_fits_open(os.path.join(seg_dir, f'GDS_{name}_segmap.fits'))
        _img = safe_fits_open(os.path.join(path, f'{name}.fits'))
        _img = np.array([_img[:,:,i] for i in range(4)])

    if log_scale:
        if np.any(_img <= 0):
            _img =_img + abs(_img.min()) + 1e-5
        _img = np.log(_img)

    return (_id, _img, _segmap)

def _data_grab(morph, path, count, silent, bands, log_scale, orig):
    if path is None:
        sub_dir = 'orig_images' if orig else 'our_images'
        my_rel = f'Documents/astro_data/{sub_dir}'
        path = os.path.join(os.getenv('HOME'), my_rel)

    count = count if count else 0

    data = None
    with open(f'./{morph}', 'r') as f:
        data = [d.strip() for d in f.readlines()[-count:]]

    if silent:
        return [_id_img_seg(path, d, bands, log_scale, orig) for d in data]
    else:
        tmp = []
        count = 1
        for d in data:
            print(f'{count/len(data)}...', end='\r')
            count += 1
            tmp.append(_id_img_seg(path, d, bands, log_scale, orig))

        return tmp

def _sbp_grab(morph, band, log_scale):
    data = None

    if 'orig_sbp.json' not in os.listdir('.'):
        all_rs, all_is = [], []
        for spheroid in spheroids:
            _id, img, segmap = spheroid
            rvals, ivals = it.sbp(img, segmap==_id)

            all_rs.append(rvals)
            all_is.append(ivals)





    with open('orig_sbp.json', 'r') as f:
        data = json.load(f)

    data = data[morph][band]

    if log_scale:
        vals = [_nmpy_decode(data['x'])]
        for d in data.keys():
            if d != 'x':
                vals.append(np.log(_nmpy_decode(data[d])))
        return tuple(vals)
    else:
        return tuple([_nmpy_decode(data[d]) for d in data.keys()])

def _ar_re_grab(morph, band):
    data = None
    if 'ar_re_bins.json' not in os.listdir('.'):
        print('Data not found. Processing...')

        data = {'ClSph':{}, 'ClDk':{}}
        for b in ['h', 'j', 'v', 'z']:
            print('Getting spheroids...')
            sph = spheroids(bands=b, count=None)
            print('Getting disks...')
            dsk = disks(bands=b, count=None)

            print('Processing spheroids...')
            s_ar, s_re = [], []
            for _id, _img, _seg in sph:
                print(_id, _img[_seg==_id].sum())
                if _img[_seg==_id].sum() == 0:
                    print(f'Invalid band data, check id:{_id} band:{b}')
                    continue

                s_ar.append(it.axis_ratio(_img, _seg==_id))
                s_re.append(it.effective_radius(_id, _img, _seg))

            data['ClSph'][b] = {}
            data['ClSph'][b]['ar'] = _nmpy_encode(np.array(s_ar))
            data['ClSph'][b]['re'] = _nmpy_encode(np.array(s_re))

            print('Processing disks...')
            d_ar, d_re = [], []
            for _id, _img, _seg in dsk:
                print(_id, _img[_seg==_id].sum())
                if _img[_seg==_id].sum() == 0:
                    print(f'Invalid band data, check id:{_id} band:{b}')
                    continue

                d_ar.append(it.axis_ratio(_img, _seg==_id))
                d_re.append(it.effective_radius(_id, _img, _seg))

            data['ClDk'][b] = {}
            data['ClDk'][b]['ar'] = _nmpy_encode(np.array(d_ar))
            data['ClDk'][b]['re'] = _nmpy_encode(np.array(d_re))

        with open('ar_re_bins.json', 'w') as f:
            json.dump(data, f)
    else:
        with open('ar_re_bins.json', 'r') as f:
            data = json.load(f)

    data = data[morph][band]

    return (_nmpy_decode(data['ar']), _nmpy_decode(data['re']))


def draw_expon(loc, scale, size=1, rejection=lambda x: False):
    vals = []

    while len(vals) < size:
        v = expon.rvs(loc=loc, scale=scale)
        if rejection(v)==False:
            vals.append(v)

    return vals[0] if size==1 else vals

def draw_norm(loc, scale, size=1, rejection=lambda x: False):
    vals = []

    while len(vals) < size:
        v = norm.rvs(loc=loc, scale=scale)
        if rejection(v)==False:
            vals.append(v)

    return vals[0] if size==1 else vals

def draw_alpha(shape, loc, scale, size=1,
               rejection=lambda x: False,
               massage=None):
    vals = []

    while len(vals) < size:
        v = alpha.rvs(shape, loc=loc, scale=scale)
        if rejection(v)==False:
            vals.append(v)
        elif massage:
            vals.append(massage(v))


    return vals[0] if size==1 else vals

def loessc(x,y,dx=None):
    #dx = 0.1*(max(x)-min(x))
    if dx is None:
        dx = 1.0e-3*np.abs(max(x)-min(x))
    y_l = np.zeros(len(y),dtype=np.float32)
    w_l = np.zeros(len(y),dtype=np.float32)

    for i in range(len(x)):
        flag = 0
        dxt = dx
        idx = (np.abs(x-x[i])<dxt).nonzero()
        xx = x[idx]
        yy = y[idx]

        wx = (1.-(np.abs(xx-x[i])/dxt)**3)**3
        nj = len(wx)

        y_l[i] = 0.0
        if(nj>0):
            for j in range(nj):
                y_l[i] += wx[j]*yy[j]
                w_l[i] += wx[j]
        else:
            y_l[i] = y[i]
            w_l[i] = 1.
        y_l[i] = y_l[i]/w_l[i]

        if((x[i]-x.min())<dx):
            dxta = np.abs(x[i]-x.min())
            idx = (np.abs(x-x[i])<dxta).nonzero()
            xx = x[idx]
            yy = y[idx]
            wx = (1.-(np.abs(xx-x[i])/dxta)**3)**3
            nj = len(wx)

            y_l[i] = 0.0
            w_l[i] = 0.0
            if(nj>0):
                for j in range(nj):
                    y_l[i] += wx[j]*yy[j]
                    w_l[i] += wx[j]
            else:
                y_l[i] += y[i]
                w_l[i] += 1.

            y_l[i] = y_l[i]/w_l[i]

        if((x.max()-x[i])<dx):
            dxta = np.abs(x.max()-x[i])
            idx = (np.abs(x-x[i])<dxta).nonzero()
            xx = x[idx]
            yy = y[idx]
            wx = (1.-(np.abs(xx-x[i])/dxta)**3)**3
            nj = len(wx)

            y_l[i] = 0.0
            w_l[i] = 0.0
            if(nj>0):
                for j in range(nj):
                    y_l[i] += wx[j]*yy[j]
                    w_l[i] += wx[j]
            else:
                y_l[i] += y[i]
                w_l[i] += 1.

            y_l[i] = y_l[i]/w_l[i]
    return y_l

def get_possible_re():
    return np.unique(it.radial_frame(84, 84, 42, 42)).flatten()

def get_param_from_dist(morph, param):
    with open('ierear_dist_params.json', 'r') as f:
        pdf_params = json.load(f)


    # rejection criteria
    if param=='ie':
        rej = lambda x: False
    elif param=='re':
        res = get_possible_re()
        rej = lambda x: (x==res).astype(np.int).sum()==0
        mass = lambda x: res[np.argmin(np.abs(res-x))]
    elif param=='ar':
        rej = lambda x: np.logical_or(x<=0, x>1)


    dist_params = pdf_params[morph][param]
    dist = dist_params['dist']
    if dist=='norm':
        return draw_norm(dist_params['loc'], dist_params['scale'], rejection=rej)
    elif dist=='expon':
        return draw_expon(dist_params['loc'], dist_params['scale'], rejection=rej)
    elif dist=='alpha':
        return draw_alpha(dist_params['shape'], dist_params['loc'], dist_params['scale'], rejection=rej, massage=mass)
    else:
        raise Exception('Distribution Not Supported')



def normalize(collection):
    denom = 0.0
    for c in collection:
        if c == 0:
            continue
        denom += np.exp(c)

    new_collection = []
    for c in collection:
        if c == 0:
            new_collection.append(0)
        else:
            new_collection.append(np.exp(c)/denom)

    return new_collection

def safe_fits_open(path):
    tmp = getdata(path)
    data = deepcopy(tmp)
    del tmp
    return data

# returns true if all bands have non zero data in the source area of the img
def all_bands_have_valid_data(src_map, img):
    for i in range(4):
        if img[i,src_map].sum() == 0:
            return False
    return True

# returns true if all of the centroids are close to each other
def all_centroids_near_each_other(src_map, img):
    # max allowed dist in units of pixels
    max_dist = 8

    # get the centroid for a single band
    y, x = np.where(img[0, :,:,]==img[0, src_map].max())
    y, x = x[0], y[0]

    for i in range(1,4):
        _y, _x, = np.where(img[i, :,:,]==img[i, src_map].max())
        _y, _x = _y[0], _x[0]

        dist = np.sqrt((x-_x)**2 + (y-_y)**2)
        if dist > max_dist:
            print(f'Measured Distance:{dist}')
            return False
    return True

def get_noise(band, shape):
    _noise = getdata(f'./noise/{band}_noise.fits')
    total = int(np.prod(shape))
    noise = []
    for i in range(total):
        n = _noise[np.random.randint(0, _noise.shape[0])]

        while (n < -0.1 or n > 0.1):
            n = _noise[np.random.randint(0, _noise.shape[0])]

        noise.append(n)

    return np.array(noise).reshape(shape)

def get_sr(morph):
    all_sr = pd.read_csv('sr_dataframe.csv', header=0, index_col=0)
    morph_sr = all_sr.loc[all_sr['label']==morph, ['h', 'j', 'v', 'z']]
    morph_sr.reset_index(inplace=True, drop=True)
    sr = morph_sr.loc[np.random.randint(0, len(morph_sr)), :]
    return sr.to_dict()


def spheroids(bands=['h', 'j', 'v', 'z'], path=None, count=None, silent=False, log_scale=False, orig=True):
    return _data_grab('spheroids', path, count, silent, bands, log_scale, orig)

def disks(bands=['h', 'j', 'v', 'z'], path=None, count=None, silent=False, log_scale=False, orig=True):
    return _data_grab('disks', path, count, silent, bands, log_scale, orig)

def spheroid_sbp(band, log_scale=False):
    return _sbp_grab('ClSph', band, log_scale)

def disk_sbp(band, log_scale=False):
    return _sbp_grab('ClDk', band, log_scale)

def disk_ar_re(band):
    return _ar_re_grab('ClDk', band)

def spheroid_ar_re(band):
    return _ar_re_grab('ClSph', band)











